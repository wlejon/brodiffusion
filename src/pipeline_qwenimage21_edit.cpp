// Qwen-Image 2.1 image-conditioned (edit / reference-image) priming.
//
// What makes this different from img2img: a condition image is CONTEXT, not a
// starting point. It never carries noise and the sampler never touches it.
// The picture enters the joint sequence twice over —
//
//   * through the vision-language encoder, as rows of the prompt stream (the
//     tower's post-merger tokens stand in for the template's `<|image_pad|>`
//     placeholders), and
//   * through the autoencoder, as latent tokens sitting in the joint
//     sequence's PREFIX, ahead of the noise being denoised.
//
// Both views have to describe the same pixels at the same geometry, which is
// why one resize feeds both: calculate_dimensions() rounds the image to a
// multiple of 32, the vision tower then emits one token per 32 pixels and the
// autoencoder one latent per 16, so each vision slot covers exactly four
// latents — the relation the DiT's joint layout is built on.
//
// Everything downstream is the ordinary machinery: the prefix is
// timestep-independent under causal_condition, so the first step extracts the
// KV cache over text AND condition-image rows together and later steps decode
// the target block against it, exactly as text-to-image does.

#include "brodiffusion/pipeline.h"

#include "brodiffusion/denoiser.h"
#include "brodiffusion/dit/qwenimage21.h"
#include "brodiffusion/image_io.h"
#include "brodiffusion/qwenimage21_text.h"

#include "brolm/qwen3vl_vl.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace brodiffusion::pipeline {

namespace bt = ::brotensor;
namespace q3 = ::brolm::qwen3vl;
namespace qi = ::brodiffusion::qwenimage21;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("pipeline::Pipeline: " + msg);
}

// One condition image, resized and split into the two views the model needs.
struct PreparedCondition {
    std::vector<float> rgb;    // (3, H, W) composited over white, for the VLM
    HostImage rgba;            // (4, H, W), for the autoencoder
    int W = 0;
    int H = 0;
};

// Decode (or adopt) a condition image at its native size.
HostImage read_condition(const ConditionImage& ci, std::size_t index) {
    const bool has_path = !ci.path.empty();
    const bool has_pixels = !ci.pixels.empty();
    if (has_path == has_pixels) {
        fail("condition image " + std::to_string(index) +
             ": set exactly one of `path` and `pixels`");
    }
    if (has_path) return brodiffusion::load_image_rgba(ci.path);

    if (ci.channels != 3 && ci.channels != 4) {
        fail("condition image " + std::to_string(index) +
             ": channels must be 3 (RGB) or 4 (RGBA)");
    }
    if (ci.H <= 0 || ci.W <= 0) {
        fail("condition image " + std::to_string(index) +
             ": pixels were supplied without H/W");
    }
    const std::size_t plane =
        static_cast<std::size_t>(ci.H) * static_cast<std::size_t>(ci.W);
    if (ci.pixels.size() != static_cast<std::size_t>(ci.channels) * plane) {
        fail("condition image " + std::to_string(index) + ": pixels holds " +
             std::to_string(ci.pixels.size()) + " values, expected " +
             std::to_string(static_cast<std::size_t>(ci.channels) * plane) +
             " for a planar " + std::to_string(ci.channels) + "x" +
             std::to_string(ci.H) + "x" + std::to_string(ci.W) + " image");
    }
    HostImage img;
    img.channels = 4;
    img.H = ci.H;
    img.W = ci.W;
    img.planes.resize(4 * plane);
    std::copy(ci.pixels.begin(),
              ci.pixels.begin() +
                  static_cast<std::ptrdiff_t>(3 * plane),
              img.planes.begin());
    if (ci.channels == 4) {
        std::copy(ci.pixels.begin() + static_cast<std::ptrdiff_t>(3 * plane),
                  ci.pixels.end(), img.planes.begin() + 3 * plane);
    } else {
        // An RGB caller means opaque. The autoencoder is RGBA on both ends and
        // reconstructs a fully opaque alpha for a generation, so this is the
        // value a round trip would have produced anyway.
        std::fill(img.planes.begin() + 3 * plane, img.planes.end(), 1.0f);
    }
    return img;
}

// Resize one image to the geometry calculate_dimensions gives for `area`, and
// take both views of it.
PreparedCondition prepare_one(const ConditionImage& ci, std::size_t index,
                              double area) {
    HostImage native = read_condition(ci, index);
    int w = 0, h = 0;
    qi::calculate_dimensions(area,
                             static_cast<double>(native.W) /
                                 static_cast<double>(native.H),
                             w, h);
    PreparedCondition out;
    out.rgba = brodiffusion::resize_rgba(native, w, h);
    out.rgb  = brodiffusion::composite_over_white(out.rgba);
    out.W = w;
    out.H = h;
    return out;
}

// VAE-encode one condition image. All four channels reach the autoencoder —
// only the vision tower's copy was flattened over white — and `encode` itself
// takes the mode (argmax, no sampling) and applies latents_mean/latents_std,
// which is what the reference's _encode_vae_image does.
bt::Tensor encode_rgba(vae_qwenimage21::Encoder& enc, const HostImage& img) {
    const int in_ch = enc.config().in_channels;
    if (in_ch != 3 && in_ch != 4) {
        fail("condition image: unexpected VAE input channel count");
    }
    const std::size_t plane = img.plane_stride();
    std::vector<float> nchw(static_cast<std::size_t>(in_ch) * plane);
    for (int c = 0; c < in_ch; ++c) {
        const float* s = img.planes.data() + static_cast<std::size_t>(c) * plane;
        float* d = nchw.data() + static_cast<std::size_t>(c) * plane;
        // [0,1] -> [-1,1], the range the encoder was trained on.
        for (std::size_t p = 0; p < plane; ++p) d[p] = s[p] * 2.0f - 1.0f;
    }
    bt::Tensor dev =
        bt::Tensor::from_host(nchw.data(), 1, static_cast<int>(nchw.size()))
            .to(bt::default_device());
    bt::Tensor latent;
    enc.encode(dev, img.H, img.W, nullptr, latent);
    return latent;
}

// Drop the `<|image_pad|>` rows, keeping the text rows in order.
//
// The DiT's txt_in only ever sees text: the placeholder rows are slots it
// fills from the condition latents (through img_in, a different projection),
// so feeding them through the text path would both waste the work and, worse,
// leave the joint sequence one row-supply short. Stripping here rather than
// after the projection is exact, because txt_in is row-independent — a
// zero-centre RMS norm plus a linear, both per row.
bt::Tensor strip_image_rows(const qi::TextConditioning& tc) {
    if (tc.image_runs.empty()) return tc.embeds;
    const int n = tc.n_valid();
    const int cols = tc.embeds.cols;
    int n_text = n;
    for (const qi::ImagePadRun& r : tc.image_runs) n_text -= r.n_slots;
    if (n_text <= 0) fail("prime: the prompt encoded to image rows only");

    bt::Tensor out = bt::Tensor::zeros_on(tc.embeds.device, n_text, cols,
                                          tc.embeds.dtype);
    int src = 0, dst = 0;
    std::size_t next_run = 0;
    while (src < n) {
        int end = n;
        if (next_run < tc.image_runs.size()) end = tc.image_runs[next_run].row;
        if (end > src) {
            bt::copy_d2d(tc.embeds, src * cols, out, dst * cols,
                         (end - src) * cols);
            dst += end - src;
        }
        src = end;
        if (next_run < tc.image_runs.size()) {
            src += tc.image_runs[next_run].n_slots;
            ++next_run;
        }
    }
    if (dst != n_text) fail("prime: text-row compaction lost rows");
    return out;
}

// Turn a TextConditioning's image_pad_mask into the DiT's prefix segments:
// alternating Text runs (the rows that are NOT image slots) and Image runs
// (one per condition image, carrying that image's latent grid).
std::vector<dit::QwenImage21Segment> segments_from(
    const qi::TextConditioning& tc) {
    std::vector<dit::QwenImage21Segment> segs;
    const int n = tc.n_valid();
    std::size_t next_run = 0;
    int row = 0;
    while (row < n) {
        if (next_run < tc.image_runs.size() &&
            tc.image_runs[next_run].row == row) {
            const qi::ImagePadRun& r = tc.image_runs[next_run];
            dit::QwenImage21Segment s;
            s.kind     = dit::QwenImage21Segment::Kind::Image;
            s.h        = r.h_lat;
            s.w        = r.w_lat;
            s.n_tokens = r.h_lat * r.w_lat;
            segs.push_back(s);
            row += r.n_slots;
            ++next_run;
            continue;
        }
        // Text runs to the start of the next image slot, or to the end.
        int end = n;
        if (next_run < tc.image_runs.size()) end = tc.image_runs[next_run].row;
        dit::QwenImage21Segment s;
        s.kind     = dit::QwenImage21Segment::Kind::Text;
        s.n_tokens = end - row;
        if (s.n_tokens <= 0) {
            fail("prime: two condition-image runs are adjacent with no "
                 "separating row — the template always writes a label "
                 "between them");
        }
        segs.push_back(s);
        row = end;
    }
    return segs;
}

}  // namespace

void Pipeline::qi21_resolve_size(const GenerateOptions& opts, int& width,
                                 int& height) {
    if (opts.width > 0 && opts.height > 0) {
        width  = opts.width;
        height = opts.height;
        return;
    }
    if (model_class_ != ModelClass::QwenImage21 ||
        opts.condition_images.empty()) {
        fail("qi21_resolve_size: height/width may only be 0 for a Qwen-Image "
             "2.1 generation with condition images");
    }
    if (opts.output_resolution <= 0) {
        fail("qi21_resolve_size: output_resolution must be positive");
    }
    // The reference derives the canvas from the LAST condition image: with
    // several references it is the final one that sets the shape of what
    // comes out.
    const ConditionImage& last = opts.condition_images.back();
    HostImage img = read_condition(last, opts.condition_images.size() - 1);
    const double area = static_cast<double>(opts.output_resolution) *
                        static_cast<double>(opts.output_resolution);
    qi::calculate_dimensions(
        area, static_cast<double>(img.W) / static_cast<double>(img.H), width,
        height);
}

qi::TextConditioning Pipeline::qi21_encode_prompt_images(
    std::string_view prompt, const std::vector<ConditionImage>& images,
    int output_resolution) {
    if (model_class_ != ModelClass::QwenImage21) {
        fail("qi21_encode_prompt_images: Qwen-Image 2.1 only");
    }
    if (images.empty()) {
        fail("qi21_encode_prompt_images: no condition images — use "
             "qi21_encode_prompt() for text only");
    }
    if (!qwen3vl_model_ || !qwen3vl_tokenizer_) {
        fail("qi21_encode_prompt_images: the Qwen3-VL text encoder is not "
             "resident (it was released — reload it first)");
    }
    if (!qwen3vl_vision_) {
        fail("qi21_encode_prompt_images: this checkpoint's text_encoder "
             "carries no vision tower, so a condition image cannot reach the "
             "prompt stream");
    }
    if (output_resolution <= 0) {
        fail("qi21_encode_prompt_images: output_resolution must be positive");
    }
    const double area = static_cast<double>(output_resolution) *
                        static_cast<double>(output_resolution);

    std::vector<PreparedCondition> prepared;
    prepared.reserve(images.size());
    for (std::size_t i = 0; i < images.size(); ++i) {
        prepared.push_back(prepare_one(images[i], i, area));
    }
    std::vector<q3::ImageInput> inputs(prepared.size());
    for (std::size_t i = 0; i < prepared.size(); ++i) {
        inputs[i].pixels = prepared[i].rgb.data();
        inputs[i].H = prepared[i].H;
        inputs[i].W = prepared[i].W;
    }
    return qi::encode_prompt_with_images(*qwen3vl_tokenizer_, *qwen3vl_model_,
                                         *qwen3vl_vision_, qwen3vl_pp_,
                                         std::string(prompt), inputs);
}

void Pipeline::qi21_prime_edit_(std::string_view prompt,
                                const GenerateOptions& opts, bool do_cfg) {
    if (!qwen3vl_vision_) {
        fail("prime: condition_images were supplied but this checkpoint's "
             "text_encoder carries no vision tower");
    }
    if (!qwen3vl_model_ || !qwen3vl_tokenizer_) {
        fail("prime: condition_images require the Qwen3-VL text encoder to be "
             "resident — it was released; reload it first");
    }
    if (qi21_text_override_ || qi21_uncond_text_override_) {
        fail("prime: qi21_prime_from_text() rows and condition_images cannot "
             "both drive one generation — caller-supplied rows already fix "
             "the prefix layout");
    }
    if (opts.output_resolution <= 0) {
        fail("prime: output_resolution must be positive");
    }
    const double area = static_cast<double>(opts.output_resolution) *
                        static_cast<double>(opts.output_resolution);

    // 1. One resize per image, feeding both the vision tower and the VAE.
    std::vector<PreparedCondition> prepared;
    prepared.reserve(opts.condition_images.size());
    for (std::size_t i = 0; i < opts.condition_images.size(); ++i) {
        prepared.push_back(prepare_one(opts.condition_images[i], i, area));
    }
    std::vector<q3::ImageInput> inputs(prepared.size());
    for (std::size_t i = 0; i < prepared.size(); ++i) {
        inputs[i].pixels = prepared[i].rgb.data();
        inputs[i].H = prepared[i].H;
        inputs[i].W = prepared[i].W;
    }

    // 2. Prompt + images through the vision-language encoder, once per CFG
    //    branch. The images are the same both times; only the prompt differs,
    //    so only the text runs' lengths change.
    qi::TextConditioning pos = qi::encode_prompt_with_images(
        *qwen3vl_tokenizer_, *qwen3vl_model_, *qwen3vl_vision_, qwen3vl_pp_,
        std::string(prompt), inputs);
    // No mask: nothing is padded at batch 1, and the encoder's all-ones mask
    // is indexed by the FULL row run — it would no longer line up with the
    // text-only rows that reach the denoiser.
    conditioning_.text_embeddings      = strip_image_rows(pos);
    conditioning_.text_embeddings_mask = bt::Tensor{};
    cond_control_.apply(conditioning_.text_embeddings, /*row_end=*/-1,
                        /*row_start=*/0);

    qi::TextConditioning neg;
    if (do_cfg) {
        neg = qi::encode_prompt_with_images(
            *qwen3vl_tokenizer_, *qwen3vl_model_, *qwen3vl_vision_,
            qwen3vl_pp_, std::string(opts.negative_prompt), inputs);
        conditioning_.uncond_embeddings      = strip_image_rows(neg);
        conditioning_.uncond_embeddings_mask = bt::Tensor{};
        conditioning_.has_uncond = true;
    } else {
        conditioning_.has_uncond = false;
        conditioning_.uncond_embeddings      = bt::Tensor{};
        conditioning_.uncond_embeddings_mask = bt::Tensor{};
    }
    conditioning_.guidance = 0.0f;

    // 3. Condition latents: VAE-encode each image and concatenate them in
    //    template order. The encoder already applies argmax (the mode, not a
    //    sample) and the latents_mean / latents_std normalisation, which is
    //    what the reference's _encode_vae_image does.
    const int LC = denoiser_->latent_channels();
    int n_cond_tokens = 0;
    for (const qi::ImagePadRun& r : pos.image_runs) {
        n_cond_tokens += r.h_lat * r.w_lat;
    }
    if (n_cond_tokens <= 0) fail("prime: condition images produced no latents");

    bt::Tensor cond_latents = bt::Tensor::zeros_on(
        bt::default_device(), n_cond_tokens, LC, bt::Dtype::FP32);
    int row = 0;
    for (std::size_t i = 0; i < prepared.size(); ++i) {
        bt::Tensor lat = encode_rgba(*vae_qi21_encoder_, prepared[i].rgba);
        const int h_lat = prepared[i].H / vae_scale_factor();
        const int w_lat = prepared[i].W / vae_scale_factor();
        if (h_lat != pos.image_runs[i].h_lat ||
            w_lat != pos.image_runs[i].w_lat) {
            fail("prime: condition image " + std::to_string(i) +
                 " encodes to a " + std::to_string(h_lat) + "x" +
                 std::to_string(w_lat) +
                 " latent grid but the vision tower reserved " +
                 std::to_string(pos.image_runs[i].h_lat) + "x" +
                 std::to_string(pos.image_runs[i].w_lat));
        }
        // (1, LC*h*w) NCHW -> (h*w, LC) tokens, the layout the DiT consumes.
        bt::Tensor tokens;
        bt::nchw_to_sequence(lat, 1, LC, h_lat, w_lat, tokens);
        bt::Tensor f32 = tokens;
        if (f32.dtype != bt::Dtype::FP32) {
            bt::Tensor t;
            bt::cast(tokens, t, bt::Dtype::FP32);
            f32 = std::move(t);
        }
        bt::copy_d2d(f32, 0, cond_latents, row * LC, h_lat * w_lat * LC);
        row += h_lat * w_lat;
    }
    bt::sync_all();

    // 4. The joint prefix each branch's conditioning implies.
    qi21_edit_prefix_.segments     = segments_from(pos);
    qi21_edit_prefix_.cond_latents = cond_latents;
    if (do_cfg) {
        // The two branches share the pixels but each carries its own copy:
        // the prepared conditioning keeps one per branch, and a research hook
        // that rewrites one branch's condition latents must not move the
        // other's ground under it.
        qi21_edit_uncond_prefix_.segments     = segments_from(neg);
        qi21_edit_uncond_prefix_.cond_latents = cond_latents;
    } else {
        qi21_edit_uncond_prefix_ = dit::QwenImage21EditPrefix{};
    }
    qi21_edit_active_ = true;
}

}  // namespace brodiffusion::pipeline
