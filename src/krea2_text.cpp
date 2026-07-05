// Krea 2 prompt → Qwen3-VL tapped-hidden-states text conditioning.
// See krea2_text.h.

#include "brodiffusion/krea2_text.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace brodiffusion::krea2 {

namespace bt = ::brotensor;

namespace {

// Exact token ids the Krea 2 checkpoint's HF tokenizer emits for the fixed
// template prefix / suffix (captured from the reference tokenizer — see header).
const std::vector<int> kPrefixIds = {
    151645, 8948, 198, 74785, 279, 2168, 553, 44193, 279, 1894, 11, 6083, 11,
    1379, 11, 10434, 11, 12194, 11, 1467, 11, 27979, 11871, 315, 279, 6171, 323,
    4004, 25, 151643, 198, 151645, 872, 198};
const std::vector<int> kSuffixIds = {151643, 198, 151645, 77091, 198};

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("krea2::encode_prompt: " + msg);
}

}  // namespace

const std::vector<int>& text_encoder_select_layers() {
    static const std::vector<int> kLayers = {2,  5,  8,  11, 14, 17,
                                             20, 23, 26, 29, 32, 35};
    return kLayers;
}

TextConditioning encode_prompt(const brolm::qwen3vl::Tokenizer& tokenizer,
                               brolm::qwen3vl::TextModel& model,
                               const std::string& prompt,
                               int max_sequence_length) {
    if (max_sequence_length <= 0) fail("max_sequence_length must be positive");

    const int prefix_len = static_cast<int>(kPrefixIds.size());   // 34
    const int suffix_len = static_cast<int>(kSuffixIds.size());   // 5
    // diffusers: content budget = max_sequence_length - num_suffix_tokens.
    const int content_budget = max_sequence_length - suffix_len;  // 507
    if (content_budget <= 0) fail("max_sequence_length too small for suffix");

    // Content tokens: BPE-encode the prompt alone (no specials — identical
    // between brolm and HF and to the reference's tokenize(prefix+prompt)[34:]),
    // truncated to the content budget (matches truncation=True).
    std::vector<int32_t> content32 = tokenizer.encode(prompt, /*add_special=*/false);
    int actual_len = static_cast<int>(content32.size());
    if (actual_len > content_budget) actual_len = content_budget;

    // Compacted encoder sequence: [prefix | content | suffix], no padding.
    std::vector<int> ids;
    ids.reserve(static_cast<std::size_t>(prefix_len + actual_len + suffix_len));
    ids.insert(ids.end(), kPrefixIds.begin(), kPrefixIds.end());
    for (int i = 0; i < actual_len; ++i) ids.push_back(content32[static_cast<std::size_t>(i)]);
    ids.insert(ids.end(), kSuffixIds.begin(), kSuffixIds.end());
    const int compacted_L = static_cast<int>(ids.size());  // 34 + actual_len + 5

    // Plain sequential positions on all three mRoPE axes (pure text → t=h=w).
    std::vector<int64_t> pos(static_cast<std::size_t>(compacted_L));
    for (int i = 0; i < compacted_L; ++i) pos[static_cast<std::size_t>(i)] = i;

    bt::Tensor embeds = model.embed_tokens(ids);
    std::vector<bt::Tensor> taps;
    model.forward_capture_hidden_states(embeds, pos, pos, pos,
                                        text_encoder_select_layers(), taps);
    if (taps.size() != static_cast<std::size_t>(kNumTextLayers)) {
        fail("unexpected number of captured layers");
    }

    const int D = kTextHiddenDim;
    if (taps[0].cols != D || taps[0].rows != compacted_L) {
        fail("captured hidden-state shape mismatch");
    }
    const bt::Dtype dt = taps[0].dtype;
    const bt::Device dev = taps[0].device;

    // Assemble the fixed (max_seq_len * num_layers, D) output, token-major /
    // layer-minor. Filler rows stay zero (masked out downstream).
    const int rows = max_sequence_length * kNumTextLayers;
    bt::Tensor prompt_embeds = bt::Tensor::zeros_on(dev, rows, D, dt);

    const int content_src = prefix_len;                 // first content row in a tap
    const int suffix_src = prefix_len + actual_len;     // first suffix row in a tap
    const int suffix_dst_tok = max_sequence_length - suffix_len;  // 507
    const int dst_pitch = kNumTextLayers * D;           // rows for one token span 12*D
    for (int l = 0; l < kNumTextLayers; ++l) {
        const bt::Tensor& tap = taps[static_cast<std::size_t>(l)];
        if (actual_len > 0) {
            bt::copy_d2d_strided(tap, content_src * D, D,
                                 prompt_embeds, l * D, dst_pitch,
                                 D, actual_len);
        }
        bt::copy_d2d_strided(tap, suffix_src * D, D,
                             prompt_embeds, (suffix_dst_tok * kNumTextLayers + l) * D,
                             dst_pitch, D, suffix_len);
    }

    // Validity mask: 1.0 for content [0, actual_len) and suffix [507, 512).
    std::vector<float> mask_h(static_cast<std::size_t>(max_sequence_length), 0.0f);
    for (int t = 0; t < actual_len; ++t) mask_h[static_cast<std::size_t>(t)] = 1.0f;
    for (int s = 0; s < suffix_len; ++s) {
        mask_h[static_cast<std::size_t>(suffix_dst_tok + s)] = 1.0f;
    }
    bt::Tensor prompt_embeds_mask =
        bt::Tensor::from_host(mask_h.data(), max_sequence_length, 1)
            .to(dev);

    bt::sync_all();
    TextConditioning out;
    out.prompt_embeds = std::move(prompt_embeds);
    out.prompt_embeds_mask = std::move(prompt_embeds_mask);
    return out;
}

TextConditioning encode_image_prompt(const brolm::qwen3vl::Tokenizer& tokenizer,
                                     brolm::qwen3vl::TextModel& model,
                                     brolm::qwen3vl::VisionTower& vision,
                                     const brolm::qwen3vl::PreprocessConfig& pp,
                                     const brolm::qwen3vl::ImageInput& image,
                                     int max_sequence_length) {
    if (max_sequence_length <= 0) fail("max_sequence_length must be positive");

    const int vision_start_id = tokenizer.vision_start_id();
    const int vision_end_id   = tokenizer.vision_end_id();
    const int image_pad_id    = tokenizer.image_pad_id();
    if (vision_start_id < 0 || vision_end_id < 0 || image_pad_id < 0) {
        fail("tokenizer is missing a vision special token id");
    }

    const int prefix_len = static_cast<int>(kPrefixIds.size());   // 34
    const int suffix_len = static_cast<int>(kSuffixIds.size());   // 5
    const int content_budget = max_sequence_length - suffix_len;  // 507

    // Run the vision tower: post-merger tokens + the DeepStack feature list.
    brolm::qwen3vl::PreprocessedImage pp_out;
    std::vector<bt::Tensor> deepstack_one;
    bt::Tensor vis_tokens =
        brolm::qwen3vl::run_vision_one(vision, pp, image, pp_out, deepstack_one);
    const int n_img = pp_out.num_image_tokens();

    // Content = <|vision_start|> + N image tokens + <|vision_end|>, mirroring
    // encode_prompt()'s plain-text content but framed the same way HF wraps an
    // image placeholder run. Unlike text, image tokens can't be truncated
    // mid-splice, so this must fit whole.
    const int content_len = n_img + 2;
    if (content_len > content_budget) {
        fail("image token count (" + std::to_string(n_img) +
             ") does not fit the content budget (" +
             std::to_string(content_budget - 2) + ")");
    }

    std::vector<int> ids;
    ids.reserve(static_cast<std::size_t>(prefix_len + content_len + suffix_len));
    ids.insert(ids.end(), kPrefixIds.begin(), kPrefixIds.end());
    const int content_start = prefix_len;   // index of <|vision_start|>
    ids.push_back(vision_start_id);
    for (int k = 0; k < n_img; ++k) ids.push_back(image_pad_id);
    ids.push_back(vision_end_id);
    ids.insert(ids.end(), kSuffixIds.begin(), kSuffixIds.end());
    const int compacted_L = static_cast<int>(ids.size());

    // M-RoPE positions: build_mrope_position_ids scans `ids` for the
    // <|image_pad|> run and assigns the standard Qwen2-VL 3-axis spatial
    // positions there, sequential text positions (all three axes equal)
    // everywhere else, continuing the running position after the image —
    // exactly the convention brolm::qwen3vl::VLM uses for live generation.
    brolm::qwen3vl::MRopePositions mp = brolm::qwen3vl::build_mrope_position_ids(
        ids, {pp_out}, image_pad_id, vision_start_id);

    bt::Tensor embeds = model.embed_tokens(ids);
    // image tokens start right after <|vision_start|> at content_start+1.
    brolm::qwen3vl::splice_vision(embeds, content_start + 1, vis_tokens);

    brolm::qwen3vl::DeepstackSplice splice;
    splice.row_start = content_start + 1;
    splice.per_layer  = std::move(deepstack_one);

    std::vector<bt::Tensor> taps;
    model.forward_capture_hidden_states(embeds, mp.t, mp.h, mp.w,
                                        text_encoder_select_layers(), taps,
                                        {splice});
    if (taps.size() != static_cast<std::size_t>(kNumTextLayers)) {
        fail("unexpected number of captured layers");
    }

    const int D = kTextHiddenDim;
    if (taps[0].cols != D || taps[0].rows != compacted_L) {
        fail("captured hidden-state shape mismatch");
    }
    const bt::Dtype dt = taps[0].dtype;
    const bt::Device dev = taps[0].device;

    const int rows = max_sequence_length * kNumTextLayers;
    bt::Tensor prompt_embeds = bt::Tensor::zeros_on(dev, rows, D, dt);

    const int suffix_src = prefix_len + content_len;
    const int suffix_dst_tok = max_sequence_length - suffix_len;  // 507
    const int dst_pitch = kNumTextLayers * D;
    for (int l = 0; l < kNumTextLayers; ++l) {
        const bt::Tensor& tap = taps[static_cast<std::size_t>(l)];
        bt::copy_d2d_strided(tap, content_start * D, D,
                             prompt_embeds, l * D, dst_pitch,
                             D, content_len);
        bt::copy_d2d_strided(tap, suffix_src * D, D,
                             prompt_embeds, (suffix_dst_tok * kNumTextLayers + l) * D,
                             dst_pitch, D, suffix_len);
    }

    std::vector<float> mask_h(static_cast<std::size_t>(max_sequence_length), 0.0f);
    for (int t = 0; t < content_len; ++t) mask_h[static_cast<std::size_t>(t)] = 1.0f;
    for (int s = 0; s < suffix_len; ++s) {
        mask_h[static_cast<std::size_t>(suffix_dst_tok + s)] = 1.0f;
    }
    bt::Tensor prompt_embeds_mask =
        bt::Tensor::from_host(mask_h.data(), max_sequence_length, 1).to(dev);

    bt::sync_all();
    TextConditioning out;
    out.prompt_embeds = std::move(prompt_embeds);
    out.prompt_embeds_mask = std::move(prompt_embeds_mask);
    return out;
}

}  // namespace brodiffusion::krea2
