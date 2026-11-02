#include "brodiffusion/dit/krea2.h"

#include "brodiffusion/dit/common.h"

#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace brodiffusion::dit {

namespace bt = ::brotensor;
namespace st = ::brotensor::safetensors;

// ═══════════════════════════════════════════════════════════════════════
// Krea2Denoiser — Denoiser-interface wrapper
// ═══════════════════════════════════════════════════════════════════════

namespace {

[[noreturn]] void fail_den(const std::string& msg) {
    throw std::runtime_error("dit::Krea2Denoiser: " + msg);
}

// Per-model prepared payload: the text_fusion output rows per CFG branch.
// encode_text (compaction + the whole fusion stack + txt_in) is timestep-
// independent, so it runs ONCE here instead of inside every denoise step —
// the reference pipeline re-runs it per forward, which is pure waste.
struct Krea2Prepared : PreparedConditioning::Impl {
    bt::Tensor txt;          // (n_valid_pos, hidden)
    bt::Tensor uncond_txt;   // (n_valid_neg, hidden)
    bool has_uncond = false;
};

}  // namespace

Krea2Denoiser::Krea2Denoiser(const Krea2Config& cfg, int patch_size)
    : model_(cfg), patch_size_(patch_size) {
    // The 2x2 pack/unpack helpers (dit::pack_latents) hardcode a 2x2 patch;
    // Krea 2 is always patch_size=2. Reject anything else at the boundary
    // rather than silently mis-packing.
    if (patch_size_ != 2) {
        fail_den("only patch_size=2 is supported (got " +
                 std::to_string(patch_size_) + ")");
    }
}

Krea2Denoiser::~Krea2Denoiser() = default;

void Krea2Denoiser::load_weights(const st::File& f, const std::string& prefix) {
    model_.load_weights(f, prefix);
}

void Krea2Denoiser::load_weights(const std::vector<const st::File*>& shards,
                                 const std::string& prefix,
                                 const std::function<bool()>& should_cancel) {
    model_.load_weights(shards, prefix, should_cancel);
}

PreparedConditioning Krea2Denoiser::prepare(const Conditioning& cond) {
    auto prep = std::make_unique<Krea2Prepared>();
    if (cond.text_embeddings.size() == 0 ||
        cond.text_embeddings_mask.size() == 0) {
        fail_den("prepare: text_embeddings / text_embeddings_mask are empty");
    }
    model_.encode_text(cond.text_embeddings, cond.text_embeddings_mask,
                       prep->txt);
    prep->has_uncond = cond.has_uncond;
    if (cond.has_uncond) {
        if (cond.uncond_embeddings.size() == 0 ||
            cond.uncond_embeddings_mask.size() == 0) {
            fail_den("prepare: has_uncond but uncond embeddings/mask are empty");
        }
        model_.encode_text(cond.uncond_embeddings, cond.uncond_embeddings_mask,
                           prep->uncond_txt);
    }
    return PreparedConditioning(std::move(prep));
}

brotensor::Tensor& Krea2Denoiser::fused_text(PreparedConditioning& prepared,
                                             bool uncond) {
    auto* prep = dynamic_cast<Krea2Prepared*>(prepared.get());
    if (!prep) fail_den("fused_text: prepared conditioning has the wrong type");
    if (uncond) {
        if (!prep->has_uncond) {
            fail_den("fused_text: uncond requested but no uncond conditioning "
                     "was prepared");
        }
        return prep->uncond_txt;
    }
    return prep->txt;
}

void Krea2Denoiser::forward(const bt::Tensor& latent, int H_lat, int W_lat,
                            float timestep,
                            const PreparedConditioning& prepared,
                            Branch branch, bt::Tensor& out) {
    if (!prepared) fail_den("forward: prepared conditioning is empty");
    const auto* prep = dynamic_cast<const Krea2Prepared*>(prepared.get());
    if (!prep) fail_den("forward: prepared conditioning has the wrong type");
    if (H_lat % patch_size_ != 0 || W_lat % patch_size_ != 0) {
        fail_den("forward: H_lat and W_lat must be multiples of patch_size");
    }

    const bt::Tensor* txt = &prep->txt;
    if (branch == Branch::Uncond) {
        if (!prep->has_uncond) {
            fail_den("forward: Uncond branch requested but no uncond "
                     "conditioning was prepared");
        }
        txt = &prep->uncond_txt;
    }

    const int LC = model_.config().latent_channels();
    const int hp = H_lat / patch_size_;
    const int wp = W_lat / patch_size_;

    // The Denoiser contract passes the flat NCHW latent; the transformer wants
    // 2x2-packed tokens. Pack (pipeline dtype) → transformer (casts to BF16
    // internally) → unpack the packed velocity back to NCHW (pipeline dtype).
    bt::Tensor packed = pack_latents(latent, LC, H_lat, W_lat);

    // The scheduler passes the continuous timestep sigma*num_train_timesteps
    // (sigma*1000); Krea2Transformer2DModel::forward expects the flow-time
    // sigma in [0,1] and re-multiplies by 1000 for its own embedding (matching
    // Flux's embedded value). So divide by 1000 here to invert that.
    const float t_flow = timestep / 1000.0f;

    model_.forward_with_text(packed, hp, wp, *txt, t_flow, tf_out_);
    unpack_latents(tf_out_, LC, H_lat, W_lat, out);
}

}  // namespace brodiffusion::dit
