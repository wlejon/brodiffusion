// Sana prompt → Gemma-2 hidden-states text encoder. See sana_text.h.

#include "brodiffusion/sana_text.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cstdint>

#include <cctype>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace brodiffusion::sana {

namespace {

// Join CHI lines with '\n' — diffusers' `"\n".join(complex_human_instruction)`.
std::string join_chi(const std::vector<std::string>& chi) {
    std::string out;
    for (std::size_t i = 0; i < chi.size(); ++i) {
        if (i) out += '\n';
        out += chi[i];
    }
    return out;
}

// diffusers `_text_preprocessing(clean_caption=False)` => `text.lower().strip()`.
// ASCII lower / ASCII-whitespace strip — sufficient for caption prompts.
std::string lower_strip(const std::string& s) {
    std::size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b])))   ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    std::string out = s.substr(b, e - b);
    for (char& c : out) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return out;
}

}  // namespace

const std::vector<std::string>& default_complex_human_instruction() {
    // Verbatim from diffusers SanaPipeline (default complex_human_instruction).
    static const std::vector<std::string> kChi = {
        "Given a user prompt, generate an 'Enhanced prompt' that provides "
        "detailed visual descriptions suitable for image generation. Evaluate "
        "the level of detail in the user prompt:",
        "- If the prompt is simple, focus on adding specifics about colors, "
        "shapes, sizes, textures, and spatial relationships to create vivid and "
        "concrete scenes.",
        "- If the prompt is already detailed, refine and enhance the existing "
        "details slightly without overcomplicating.",
        "Here are examples of how to transform or refine prompts:",
        "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat "
        "curled up in a round shape, sleeping peacefully on a warm sunny "
        "windowsill, surrounded by pots of blooming red flowers.",
        "- User Prompt: A busy city street -> Enhanced: A bustling city street "
        "scene at dusk, featuring glowing street lamps, a diverse crowd of "
        "people in colorful clothing, and a double-decker bus passing by "
        "towering glass skyscrapers.",
        "Please generate only the enhanced description for the prompt below and "
        "avoid including any additional commentary or evaluations:",
        "User Prompt: ",
    };
    return kChi;
}

brotensor::Tensor encode_prompt(
    brolm::gemma::Gemma2Model& model,
    const brolm::gemma::Tokenizer& tokenizer,
    const std::string& prompt,
    int max_seq_len,
    const std::vector<std::string>& chi) {
    if (max_seq_len <= 0) {
        throw std::runtime_error(
            "sana::encode_prompt: max_seq_len must be positive");
    }

    // Preprocess the user prompt, then build the full CHI-prefixed string and
    // the diffusers truncation cap (`max_length_all`).
    const std::string user = lower_strip(prompt);
    std::string full;
    int cap;
    int num_chi = 0;
    if (chi.empty()) {
        full = user;
        cap  = max_seq_len;
    } else {
        const std::string chi_prompt = join_chi(chi);
        full = chi_prompt + user;
        // diffusers: num_chi_prompt_tokens = len(tokenizer.encode(chi_prompt))
        // (HF encode adds <bos>), max_length_all = num_chi + max_seq_len - 2.
        num_chi = static_cast<int>(tokenizer.encode(chi_prompt).size());
        cap = num_chi + max_seq_len - 2;
    }
    if (cap < 1) cap = 1;

    // Tokenize with a leading <bos>, no <eos> (HF add_special_tokens=True), then
    // right-truncate to the cap (HF truncation=True keeps the first `cap` ids).
    std::vector<int32_t> ids = tokenizer.encode(full);
    if (static_cast<int>(ids.size()) > cap) {
        ids.resize(static_cast<std::size_t>(cap));
    }
    const int L = static_cast<int>(ids.size());
    if (L <= 0) {
        throw std::runtime_error(
            "sana::encode_prompt: empty token sequence");
    }

    // Fresh causal forward over exactly the L valid tokens; keep the
    // post-final-norm hidden states (Gemma last_hidden_state).
    model.allocate_cache(L);   // sizes the KV cache and resets cache_len to 0
    brotensor::Tensor logits, hidden;
    model.forward(ids.data(), L, logits, &hidden);

    // diffusers token selection (SanaPipeline.encode_prompt, "Section 3.1"):
    //   select_index = [0] + range(-max_seq_len+1, 0)
    // applied to the max_length_all-padded sequence. With the CHI present this
    // keeps the BOS (row 0) plus the trailing window that begins at the last CHI
    // token (padded index num_chi-1) — DROPPING the CHI body so the DiT does not
    // cross-attend over the instruction's example prompts — and runs through the
    // user prompt. Padding rows in diffusers are zeroed by the attention mask;
    // here we simply omit them, since the DiT attends over exactly the returned
    // rows. Without a CHI the window covers the whole (<= max_seq_len) sequence,
    // so all rows are kept (the current return).
    if (!chi.empty()) {
        const int D = hidden.cols;
        int start = num_chi - 1;            // first kept content row after BOS
        if (start < 1) start = 1;           // degenerate CHI → keep everything
        if (start < L) {
            const int n_tail = L - start;   // rows [start .. L-1]
            brotensor::Tensor sel =
                brotensor::Tensor::empty(1 + n_tail, D, hidden.dtype);
            brotensor::copy_d2d(hidden, 0, sel, 0, D);            // BOS row
            brotensor::copy_d2d(hidden, start * D, sel, D, n_tail * D);
            hidden = std::move(sel);
        }
    }
    return hidden;             // (L_sel, 2304) at brolm::compute_dtype()
}

}  // namespace brodiffusion::sana
