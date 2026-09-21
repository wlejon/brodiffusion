// QwenImage21Denoiser — QwenImage21Transformer2DModel behind brodiffusion's
// model-agnostic Denoiser interface.
//
// Two things this wrapper owns that the transformer does not:
//
//   * latent layout. The Pipeline speaks the flat NCHW row
//     (1, 64*H_lat*W_lat); the transformer speaks (H_lat*W_lat, 64) tokens.
//     patch_size is 1, so the conversion is a plain (C, HW) <-> (HW, C)
//     transpose — nchw_to_sequence / sequence_to_nchw, on device.
//   * the per-branch prefix KV cache. Each CFG branch has its own prompt and
//     therefore its own prefix; prepare() creates both (empty), the first
//     forward after prepare extracts, and later forwards decode. A latent
//     grid change resets the affected cache automatically.

#include "brodiffusion/dit/qwenimage21.h"

#include "qwenimage21_detail.h"

#include "brodiffusion/dit/common.h"
#include "brodiffusion/detail/compute.h"
#include "brodiffusion/detail/device.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace brodiffusion::dit {

namespace bt = ::brotensor;
namespace st = ::brotensor::safetensors;

namespace {

[[noreturn]] void fail_den(const std::string& msg) {
    throw std::runtime_error("dit::QwenImage21Denoiser: " + msg);
}

// Per-branch prepared payload: txt_in's output rows plus that branch's prefix
// KV cache. The caches are `mutable` because Denoiser::forward takes the
// prepared conditioning by const reference while the cache is, by design,
// written on the first step and read on every later one.
struct QwenImage21Prepared : PreparedConditioning::Impl {
    bt::Tensor txt;           // (n_valid_pos, hidden)
    bt::Tensor uncond_txt;    // (n_valid_neg, hidden)
    bool has_uncond = false;
    mutable QwenImage21PrefixCache cache;
    mutable QwenImage21PrefixCache uncond_cache;
    // Image-conditioned prefixes, one per branch. Empty segments = the plain
    // text-to-image layout, which forward() then builds on the fly from the
    // branch's row count.
    QwenImage21EditPrefix prefix;
    QwenImage21EditPrefix uncond_prefix;
};

// The text-to-image prefix: a single Text run covering every conditioning row.
std::vector<QwenImage21Segment> text_only_prefix(int n_rows) {
    std::vector<QwenImage21Segment> segs(1);
    segs[0].kind     = QwenImage21Segment::Kind::Text;
    segs[0].n_tokens = n_rows;
    return segs;
}

// Cross-check a branch's prefix against the rows it will consume, so a
// mismatch is reported here — where the caller can see which branch and which
// count disagreed — rather than as a shape error 32 blocks deep.
void validate_prefix(const QwenImage21EditPrefix& p, int txt_rows,
                     const char* branch) {
    if (p.empty()) return;
    int n_text = 0, n_cond = 0;
    for (const QwenImage21Segment& s : p.segments) {
        if (s.kind == QwenImage21Segment::Kind::Text) {
            n_text += s.n_tokens;
        } else {
            n_cond += s.h * s.w;
        }
    }
    if (n_text != txt_rows) {
        fail_den(std::string("prepare_edit: the ") + branch + " prefix's text "
                 "segments cover " + std::to_string(n_text) + " rows but the "
                 "branch encoded " + std::to_string(txt_rows));
    }
    const int have = p.cond_latents.size() == 0 ? 0 : p.cond_latents.rows;
    if (n_cond != have) {
        fail_den(std::string("prepare_edit: the ") + branch + " prefix's image "
                 "segments cover " + std::to_string(n_cond) + " latent tokens "
                 "but cond_latents holds " + std::to_string(have));
    }
}

// Drop right-padded text rows. Qwen-Image 2.1's causal structure has no other
// way to exclude a padded key, and at batch 1 the reference pipeline pads
// nothing at all, so a mask that marks every row valid (or no mask) is the
// normal case and this is a no-op. Trailing pads are safe to remove: RoPE
// positions are assigned in order, so dropping the tail leaves every
// surviving row's position unchanged.
bt::Tensor compact_valid_rows(const bt::Tensor& embeds,
                              const bt::Tensor& mask) {
    if (mask.size() == 0) return embeds;
    bt::Tensor m32 = mask;
    if (m32.dtype != bt::Dtype::FP32) {
        bt::Tensor t;
        bt::cast(mask, t, bt::Dtype::FP32);
        m32 = std::move(t);
    }
    std::vector<float> h = m32.to(bt::Device::CPU).to_host_vector();
    const int n = static_cast<int>(h.size());
    if (n != embeds.rows) {
        fail_den("prepare: text mask length does not match the embedding rows");
    }
    int last = -1;
    for (int i = 0; i < n; ++i) {
        if (h[static_cast<std::size_t>(i)] > 0.5f) last = i;
    }
    if (last < 0) fail_den("prepare: text mask has no valid tokens");
    for (int i = 0; i <= last; ++i) {
        if (h[static_cast<std::size_t>(i)] <= 0.5f) {
            fail_den("prepare: the text mask has a mid-sequence pad — "
                     "Qwen-Image 2.1 only supports right-padded prompts "
                     "(interior pads would shift every later RoPE position)");
        }
    }
    const int n_valid = last + 1;
    if (n_valid == embeds.rows) return embeds;
    bt::Tensor out;
    detail::resize_like(out, n_valid, embeds.cols, embeds.dtype,
                        embeds.device);
    bt::copy_d2d(embeds, 0, out, 0, n_valid * embeds.cols);
    return out;
}

}  // namespace

QwenImage21Denoiser::QwenImage21Denoiser(const QwenImage21Config& cfg)
    : model_(cfg) {}

QwenImage21Denoiser::~QwenImage21Denoiser() = default;

void QwenImage21Denoiser::load_weights(const st::File& f,
                                       const std::string& prefix) {
    model_.load_weights(f, prefix);
}

void QwenImage21Denoiser::load_weights(
    const std::vector<const st::File*>& shards, const std::string& prefix,
    const std::function<bool()>& should_cancel) {
    model_.load_weights(shards, prefix, should_cancel);
}

namespace {

// The shared body of prepare() / prepare_edit(): the per-branch text encode.
std::unique_ptr<QwenImage21Prepared> encode_branches(
    QwenImage21Transformer2DModel& model, const Conditioning& cond) {
    auto prep = std::make_unique<QwenImage21Prepared>();
    if (cond.text_embeddings.size() == 0) {
        fail_den("prepare: text_embeddings is empty");
    }
    model.encode_text(
        compact_valid_rows(cond.text_embeddings, cond.text_embeddings_mask),
        prep->txt);
    prep->has_uncond = cond.has_uncond;
    if (cond.has_uncond) {
        if (cond.uncond_embeddings.size() == 0) {
            fail_den("prepare: has_uncond but uncond_embeddings is empty");
        }
        model.encode_text(
            compact_valid_rows(cond.uncond_embeddings,
                               cond.uncond_embeddings_mask),
            prep->uncond_txt);
    }
    return prep;
}

}  // namespace

PreparedConditioning QwenImage21Denoiser::prepare(const Conditioning& cond) {
    return PreparedConditioning(encode_branches(model_, cond));
}

PreparedConditioning QwenImage21Denoiser::prepare_edit(
    const Conditioning& cond, const QwenImage21EditPrefix& cond_prefix,
    const QwenImage21EditPrefix* uncond_prefix) {
    auto prep = encode_branches(model_, cond);
    validate_prefix(cond_prefix, prep->txt.rows, "cond");
    prep->prefix = cond_prefix;
    if (prep->has_uncond) {
        // With no explicit uncond prefix, reuse the cond one: the condition
        // images are shared between the branches and only the text runs'
        // lengths differ. That is fine exactly when the two prompts encoded
        // to the same number of rows; otherwise validate_prefix says which
        // count disagreed and the caller supplies a second description.
        const QwenImage21EditPrefix& up =
            uncond_prefix != nullptr ? *uncond_prefix : cond_prefix;
        validate_prefix(up, prep->uncond_txt.rows, "uncond");
        prep->uncond_prefix = up;
    }
    return PreparedConditioning(std::move(prep));
}

void QwenImage21Denoiser::reset_cache(PreparedConditioning& prepared) {
    auto* prep = dynamic_cast<QwenImage21Prepared*>(prepared.get());
    if (!prep) fail_den("reset_cache: prepared conditioning has the wrong type");
    prep->cache.reset();
    prep->uncond_cache.reset();
}

bt::Tensor& QwenImage21Denoiser::text_rows(PreparedConditioning& prepared,
                                           bool uncond) {
    auto* prep = dynamic_cast<QwenImage21Prepared*>(prepared.get());
    if (!prep) fail_den("text_rows: prepared conditioning has the wrong type");
    if (uncond) {
        if (!prep->has_uncond) {
            fail_den("text_rows: uncond requested but no uncond conditioning "
                     "was prepared");
        }
        return prep->uncond_txt;
    }
    return prep->txt;
}

QwenImage21PrefixCache& QwenImage21Denoiser::prefix_cache(
    PreparedConditioning& prepared, bool uncond) {
    auto* prep = dynamic_cast<QwenImage21Prepared*>(prepared.get());
    if (!prep) fail_den("prefix_cache: prepared conditioning has the wrong type");
    if (uncond) {
        if (!prep->has_uncond) {
            fail_den("prefix_cache: uncond requested but no uncond "
                     "conditioning was prepared");
        }
        return prep->uncond_cache;
    }
    return prep->cache;
}

bool QwenImage21Denoiser::has_uncond(const PreparedConditioning& prepared) const {
    const auto* prep = dynamic_cast<const QwenImage21Prepared*>(prepared.get());
    return prep != nullptr && prep->has_uncond;
}

void QwenImage21Denoiser::forward(const bt::Tensor& latent, int H_lat,
                                  int W_lat, float timestep,
                                  const PreparedConditioning& prepared,
                                  Branch branch, bt::Tensor& out) {
    if (!prepared) fail_den("forward: prepared conditioning is empty");
    const auto* prep = dynamic_cast<const QwenImage21Prepared*>(prepared.get());
    if (!prep) fail_den("forward: prepared conditioning has the wrong type");

    const bt::Tensor* txt = &prep->txt;
    QwenImage21PrefixCache* cache = &prep->cache;
    const QwenImage21EditPrefix* edit = &prep->prefix;
    if (branch == Branch::Uncond) {
        if (!prep->has_uncond) {
            fail_den("forward: Uncond branch requested but no uncond "
                     "conditioning was prepared");
        }
        txt = &prep->uncond_txt;
        cache = &prep->uncond_cache;
        edit = &prep->uncond_prefix;
    }

    const int LC = model_.config().latent_channels();
    const std::size_t expect =
        static_cast<std::size_t>(LC) * H_lat * W_lat;
    if (static_cast<std::size_t>(latent.size()) != expect) {
        fail_den("forward: latent has unexpected element count");
    }

    // The joint prefix: either the interleaved edit layout or, with no
    // condition images, one text run over every conditioning row.
    const std::vector<QwenImage21Segment> t2i_segments =
        edit->empty() ? text_only_prefix(txt->rows)
                      : std::vector<QwenImage21Segment>{};
    const std::vector<QwenImage21Segment>& segments =
        edit->empty() ? t2i_segments : edit->segments;
    int prefix_len = 0;
    for (const QwenImage21Segment& s : segments) {
        prefix_len += (s.kind == QwenImage21Segment::Kind::Text) ? s.n_tokens
                                                                 : s.h * s.w;
    }

    // A grid change invalidates the cache (its target block, and therefore
    // every RoPE position after the prefix, differs). The prompt cannot change
    // without a new prepare(), which brings a fresh cache with it.
    if (cache->valid() && !cache->matches(prefix_len, H_lat, W_lat)) {
        cache->reset();
    }

    // patch_size == 1: packing the latent is the (C, HW) -> (HW, C) transpose.
    bt::nchw_to_sequence(latent, 1, LC, H_lat, W_lat, packed_);

    // The scheduler passes the continuous timestep sigma*num_train_timesteps
    // (sigma*1000); the transformer wants the flow time in [0,1] and
    // re-multiplies by 1000 for its own sinusoid.
    const float t_flow = timestep / 1000.0f;
    model_.forward_joint(packed_, H_lat, W_lat, *txt,
                         edit->empty() ? nullptr : &edit->cond_latents,
                         segments, t_flow, cache, tf_out_);

    bt::Tensor raw;
    bt::sequence_to_nchw(tf_out_, 1, LC, H_lat, W_lat, raw);
    const bt::Dtype dt = brodiffusion::compute_dtype();
    if (raw.dtype != dt) {
        bt::cast(raw, out, dt);
    } else {
        out = std::move(raw);
    }
}

}  // namespace brodiffusion::dit
