// Qwen-Image 2.1 image-conditioned prompt encoding — the edit path's front
// end. See qwenimage21_text.h.
//
// This is the same Qwen3-VL-8B prefill the text-only encoder runs, with three
// additions:
//
//   1. the template gains one "<imageN><|vision_start|>...<|vision_end|>" run
//      per condition image, ahead of the prompt's own tokens;
//   2. each run's `<|image_pad|>` input embeddings are replaced by that
//      image's post-merger vision tokens, and the tower's DeepStack features
//      are added into the first decoder layers at the same rows;
//   3. M-RoPE positions come from the 3-axis spatial layout instead of a flat
//      counter, so the image rows carry their grid geometry.
//
// Everything after the prefill — the drop of the leading system rows, the
// (n_valid, 4096) output — is identical, which is the point: the DiT reads one
// interleaved text/image stream and cannot tell where the rows came from.

#include "brodiffusion/qwenimage21_text.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace brodiffusion::qwenimage21 {

namespace bt = ::brotensor;
namespace q3 = ::brolm::qwen3vl;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("qwenimage21::encode_prompt_with_images: " + msg);
}

}  // namespace

std::vector<int> image_label_ids(const q3::Tokenizer& tokenizer,
                                 int one_based_index) {
    if (one_based_index < 1) {
        fail("image label index is 1-based");
    }
    // The first label sits straight after the user header's "\n"; every later
    // one is separated from the preceding <|vision_end|> by a space, which the
    // reference writes into the template string itself.
    std::string label = one_based_index == 1 ? "<" : " <";
    label += "image";
    label += std::to_string(one_based_index);
    label += ">";
    std::vector<std::int32_t> ids32 =
        tokenizer.encode(label, /*add_special=*/false);
    std::vector<int> ids;
    ids.reserve(ids32.size());
    for (std::int32_t v : ids32) ids.push_back(static_cast<int>(v));
    return ids;
}

std::vector<int> build_ti2i_tokens(const q3::Tokenizer& tokenizer,
                                   const std::string& prompt,
                                   const std::vector<int>& n_slots,
                                   std::vector<int>& slot_rows_out,
                                   int max_content_tokens) {
    if (max_content_tokens <= 0) fail("max_content_tokens must be positive");
    if (n_slots.empty()) fail("build_ti2i_tokens: no condition images");

    const std::string text = prompt.empty() ? std::string(" ") : prompt;
    std::vector<std::int32_t> content =
        tokenizer.encode(text, /*add_special=*/false);
    int n_content = static_cast<int>(content.size());
    if (n_content > max_content_tokens) n_content = max_content_tokens;

    std::vector<int> ids = system_prefix_ids();
    const std::vector<int>& header = user_header_ids();
    ids.insert(ids.end(), header.begin(), header.end());

    slot_rows_out.clear();
    slot_rows_out.reserve(n_slots.size());
    for (std::size_t i = 0; i < n_slots.size(); ++i) {
        if (n_slots[i] <= 0) fail("an image contributes no vision tokens");
        const std::vector<int> label =
            image_label_ids(tokenizer, static_cast<int>(i) + 1);
        ids.insert(ids.end(), label.begin(), label.end());
        ids.push_back(kVisionStartId);
        slot_rows_out.push_back(static_cast<int>(ids.size()));
        for (int k = 0; k < n_slots[i]; ++k) ids.push_back(kImagePadId);
        ids.push_back(kVisionEndId);
    }

    for (int i = 0; i < n_content; ++i) {
        ids.push_back(static_cast<int>(content[static_cast<std::size_t>(i)]));
    }
    const std::vector<int>& suffix = assistant_suffix_ids();
    ids.insert(ids.end(), suffix.begin(), suffix.end());
    return ids;
}

void calculate_dimensions(double target_area, double ratio, int& width_out,
                          int& height_out) {
    if (!(target_area > 0.0) || !(ratio > 0.0)) {
        fail("calculate_dimensions: target_area and ratio must be positive");
    }
    const double w = std::sqrt(target_area * ratio);
    const double h = w / ratio;
    // The reference rounds (round(x / 32) * 32), it does not floor — a 32-wide
    // difference here changes the token count and therefore the whole RoPE
    // layout, so the rule has to match exactly.
    int wi = static_cast<int>(std::lround(w / 32.0)) * 32;
    int hi = static_cast<int>(std::lround(h / 32.0)) * 32;
    if (wi < 32) wi = 32;
    if (hi < 32) hi = 32;
    width_out  = wi;
    height_out = hi;
}

TextConditioning encode_prompt_with_images(
    const q3::Tokenizer& tokenizer, q3::TextModel& model,
    q3::VisionTower& vision, const q3::PreprocessConfig& pp,
    const std::string& prompt,
    const std::vector<q3::ImageInput>& images, int max_content_tokens) {
    if (images.empty()) {
        fail("no condition images — use encode_prompt() for text only");
    }
    const auto& tcfg = model.config();
    if (tcfg.hidden_size != kTextHiddenDim) {
        fail("the text backbone has hidden_size " +
             std::to_string(tcfg.hidden_size) + ", expected " +
             std::to_string(kTextHiddenDim));
    }

    // 1. Vision tower, one image at a time. Each pass yields the post-merger
    //    token block plus the per-layer DeepStack features for the same rows.
    const std::size_t n_img = images.size();
    std::vector<bt::Tensor> vis_tokens(n_img);
    std::vector<std::vector<bt::Tensor>> deepstack(n_img);
    std::vector<q3::PreprocessedImage> pp_out(n_img);
    std::vector<int> n_slots(n_img);
    for (std::size_t i = 0; i < n_img; ++i) {
        if (images[i].pixels == nullptr || images[i].H <= 0 ||
            images[i].W <= 0) {
            fail("condition image " + std::to_string(i) +
                 " has no pixels — pass FP32 CHW RGB in [0,1]");
        }
        vis_tokens[i] =
            q3::run_vision_one(vision, pp, images[i], pp_out[i], deepstack[i]);
        n_slots[i] = pp_out[i].num_image_tokens();

        // The caller sized the image; smart_resize must have left it alone,
        // or the vision grid and the VAE's latent grid would describe
        // different pictures and the DiT's 4-latents-per-slot arithmetic
        // would not close. Both sides being a multiple of 32 (which
        // calculate_dimensions guarantees) and the pixel count sitting inside
        // [min_pixels, max_pixels] is exactly the condition for that.
        if (pp_out[i].grid_h * pp.patch_size != images[i].H ||
            pp_out[i].grid_w * pp.patch_size != images[i].W) {
            fail("condition image " + std::to_string(i) + " is " +
                 std::to_string(images[i].W) + "x" +
                 std::to_string(images[i].H) +
                 ", which the preprocessor rescaled to " +
                 std::to_string(pp_out[i].grid_w * pp.patch_size) + "x" +
                 std::to_string(pp_out[i].grid_h * pp.patch_size) +
                 " — size it with calculate_dimensions() so the vision grid "
                 "and the VAE latent grid agree");
        }
    }

    // 2. Template tokens with one <|image_pad|> run per image.
    std::vector<int> slot_rows;
    std::vector<int> ids = build_ti2i_tokens(tokenizer, prompt, n_slots,
                                             slot_rows, max_content_tokens);

    // 3. Input embeddings with the vision tokens spliced over the placeholders.
    bt::Tensor embeds = model.embed_tokens(ids);
    std::vector<q3::DeepstackSplice> splices;
    splices.reserve(n_img);
    for (std::size_t i = 0; i < n_img; ++i) {
        q3::splice_vision(embeds, slot_rows[i], vis_tokens[i]);
        if (!deepstack[i].empty()) {
            q3::DeepstackSplice s;
            s.row_start = slot_rows[i];
            s.per_layer = std::move(deepstack[i]);
            splices.push_back(std::move(s));
        }
    }

    // 4. M-RoPE. build_mrope_position_ids finds the <|image_pad|> runs itself
    //    and lays the standard 3-axis spatial positions over them, advancing
    //    the shared text position past each image — the same convention
    //    brolm's VLM uses for live generation, and what the checkpoint was
    //    trained with.
    q3::MRopePositions mp =
        q3::build_mrope_position_ids(ids, pp_out, kImagePadId, kVisionStartId);

    EncodeOverrides ov;
    ov.inputs_embeds = &embeds;
    ov.mrope_t = &mp.t;
    ov.mrope_h = &mp.h;
    ov.mrope_w = &mp.w;
    ov.deepstack = std::move(splices);

    TextConditioning tc = encode_tokens(model, ids, kDropIdx, ov);

    // 5. Where the image rows ended up, in post-drop coordinates.
    const int n_valid = tc.n_valid();
    tc.image_pad_mask.assign(static_cast<std::size_t>(n_valid), 0u);
    tc.image_runs.resize(n_img);
    for (std::size_t i = 0; i < n_img; ++i) {
        const int row = slot_rows[i] - kDropIdx;
        if (row < 0 || row + n_slots[i] > n_valid) {
            fail("an image run falls outside the surviving rows");
        }
        for (int k = 0; k < n_slots[i]; ++k) {
            tc.image_pad_mask[static_cast<std::size_t>(row + k)] = 1u;
        }
        tc.image_runs[i].row     = row;
        tc.image_runs[i].n_slots = n_slots[i];
        tc.image_runs[i].h_lat   = pp_out[i].grid_h;
        tc.image_runs[i].w_lat   = pp_out[i].grid_w;
        // The slot count is the merged grid; four latents per slot is the
        // reference's _IMG_TOKENS_PER_SLOT, and this is where that identity
        // is checked rather than assumed downstream.
        if (pp_out[i].grid_h * pp_out[i].grid_w != 4 * n_slots[i]) {
            fail("image " + std::to_string(i) + " has " +
                 std::to_string(n_slots[i]) + " vision slots but a " +
                 std::to_string(pp_out[i].grid_h) + "x" +
                 std::to_string(pp_out[i].grid_w) +
                 " latent grid — the four-latents-per-slot layout requires "
                 "grid_h*grid_w == 4*slots");
        }
    }
    return tc;
}

}  // namespace brodiffusion::qwenimage21
