// Qwen-Image 2.1 prompt → Qwen3-VL-8B last-hidden-state text conditioning.
// See qwenimage21_text.h.

#include "brodiffusion/qwenimage21_text.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace brodiffusion::qwenimage21 {

namespace bt = ::brotensor;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("qwenimage21::encode_prompt: " + msg);
}

}  // namespace

const std::vector<int>& system_prefix_ids() {
    // "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
    static const std::vector<int> kIds = {
        151644, 8948, 198, 1092, 30782, 408, 323, 23643, 279, 3897, 9934, 13,
        151645, 198};
    return kIds;
}

const std::vector<int>& user_header_ids() {
    // "<|im_start|>user\n"
    static const std::vector<int> kIds = {151644, 872, 198};
    return kIds;
}

const std::vector<int>& assistant_suffix_ids() {
    // "<|im_end|>\n<|im_start|>assistant\n"
    static const std::vector<int> kIds = {151645, 198, 151644, 77091, 198};
    return kIds;
}

std::vector<int> build_t2i_tokens(const brolm::qwen3vl::Tokenizer& tokenizer,
                                  const std::string& prompt,
                                  int max_content_tokens) {
    if (max_content_tokens <= 0) fail("max_content_tokens must be positive");

    // Qwen has no BOS token, so an empty prompt would leave the encoder with
    // nothing between the user header and the suffix. diffusers substitutes a
    // single space; do the same so an empty prompt (the usual negative) lands
    // on the same conditioning.
    const std::string text = prompt.empty() ? std::string(" ") : prompt;

    std::vector<std::int32_t> content =
        tokenizer.encode(text, /*add_special=*/false);
    int n_content = static_cast<int>(content.size());
    if (n_content > max_content_tokens) n_content = max_content_tokens;

    const std::vector<int>& prefix = system_prefix_ids();
    const std::vector<int>& header = user_header_ids();
    const std::vector<int>& suffix = assistant_suffix_ids();

    std::vector<int> ids;
    ids.reserve(prefix.size() + header.size() +
                static_cast<std::size_t>(n_content) + suffix.size());
    ids.insert(ids.end(), prefix.begin(), prefix.end());
    ids.insert(ids.end(), header.begin(), header.end());
    for (int i = 0; i < n_content; ++i) {
        ids.push_back(static_cast<int>(content[static_cast<std::size_t>(i)]));
    }
    ids.insert(ids.end(), suffix.begin(), suffix.end());
    return ids;
}

TextConditioning encode_tokens(brolm::qwen3vl::TextModel& model,
                               const std::vector<int>& ids,
                               int drop_idx,
                               const EncodeOverrides& ov) {
    const int L = static_cast<int>(ids.size());
    if (L <= 0) fail("token sequence is empty");
    if (drop_idx < 0 || drop_idx >= L) {
        fail("drop_idx (" + std::to_string(drop_idx) +
             ") must be in [0, token count = " + std::to_string(L) + ")");
    }

    const auto& tcfg = model.config();
    if (tcfg.hidden_size != kTextHiddenDim) {
        fail("the text backbone has hidden_size " +
             std::to_string(tcfg.hidden_size) + ", expected " +
             std::to_string(kTextHiddenDim) +
             " — Qwen-Image 2.1 ships the Qwen3-VL 8B encoder");
    }

    // M-RoPE positions. Pure text advances all three axes together, one per
    // token; an image-conditioned caller supplies its own (the <|image_pad|>
    // run gets the 2-D spatial layout and the shared position resumes after).
    std::vector<std::int64_t> seq;
    const std::vector<std::int64_t>* pt = ov.mrope_t;
    const std::vector<std::int64_t>* ph = ov.mrope_h;
    const std::vector<std::int64_t>* pw = ov.mrope_w;
    if (pt == nullptr || ph == nullptr || pw == nullptr) {
        if (pt != nullptr || ph != nullptr || pw != nullptr) {
            fail("EncodeOverrides: supply all three M-RoPE axes or none");
        }
        seq.resize(static_cast<std::size_t>(L));
        for (int i = 0; i < L; ++i) seq[static_cast<std::size_t>(i)] = i;
        pt = ph = pw = &seq;
    }
    if (static_cast<int>(pt->size()) != L ||
        static_cast<int>(ph->size()) != L ||
        static_cast<int>(pw->size()) != L) {
        fail("EncodeOverrides: M-RoPE position arrays must have one entry "
             "per token");
    }

    bt::Tensor embeds;
    if (ov.inputs_embeds != nullptr) {
        if (ov.inputs_embeds->rows != L ||
            ov.inputs_embeds->cols != tcfg.hidden_size) {
            fail("EncodeOverrides: inputs_embeds must be "
                 "(token count, hidden_size)");
        }
        embeds = *ov.inputs_embeds;
    } else {
        embeds = model.embed_tokens(ids);
    }

    // hidden_states[-1] in HF == the residual stream after the LAST decoder
    // layer, BEFORE the final RMSNorm. brolm's capture indices follow the same
    // 1-based convention, so that is capture layer num_hidden_layers.
    const std::vector<int> capture = {tcfg.num_hidden_layers};
    std::vector<bt::Tensor> taps;
    model.forward_capture_hidden_states(embeds, *pt, *ph, *pw, capture, taps,
                                        ov.deepstack);
    if (taps.size() != 1) fail("expected exactly one captured layer");
    bt::Tensor& hs = taps[0];
    if (hs.rows != L || hs.cols != kTextHiddenDim) {
        fail("captured hidden-state shape mismatch");
    }

    // Drop the leading system rows. No padding is involved at batch 1, so the
    // rest of the sequence is the conditioning verbatim.
    const int n_valid = L - drop_idx;
    bt::Tensor out = bt::Tensor::zeros_on(hs.device, n_valid, kTextHiddenDim,
                                          hs.dtype);
    bt::copy_d2d(hs, drop_idx * kTextHiddenDim, out, 0,
                 n_valid * kTextHiddenDim);

    std::vector<float> ones(static_cast<std::size_t>(n_valid), 1.0f);
    bt::Tensor mask =
        bt::Tensor::from_host(ones.data(), n_valid, 1).to(hs.device);

    bt::sync_all();
    TextConditioning tc;
    tc.embeds    = std::move(out);
    tc.mask      = std::move(mask);
    tc.token_ids = ids;
    tc.drop_idx  = drop_idx;
    return tc;
}

TextConditioning encode_prompt(const brolm::qwen3vl::Tokenizer& tokenizer,
                               brolm::qwen3vl::TextModel& model,
                               const std::string& prompt,
                               int max_content_tokens) {
    std::vector<int> ids =
        build_t2i_tokens(tokenizer, prompt, max_content_tokens);

    // Checkpoint-mismatch guard: the hard-coded prefix must really be the
    // system turn this tokenizer produces. The prompt's own tokens vary, but
    // the fixed spans do not — if the tokenizer ever permutes the chat-control
    // ids (as Krea 2's does), the first 14 ids would stop matching and every
    // downstream row would silently shift.
    const std::vector<int>& prefix = system_prefix_ids();
    if (static_cast<int>(prefix.size()) != kDropIdx) {
        fail("the system prefix is not kDropIdx tokens long");
    }
    for (std::size_t i = 0; i < prefix.size(); ++i) {
        if (ids[i] != prefix[i]) fail("system prefix id mismatch");
    }

    return encode_tokens(model, ids, kDropIdx);
}

}  // namespace brodiffusion::qwenimage21
