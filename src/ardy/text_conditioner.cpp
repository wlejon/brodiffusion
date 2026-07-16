#include "brodiffusion/ardy/text_conditioner.h"

#include "brolm/llama3_tokenizer.h"
#include "brolm/llm2vec.h"

#include "brotensor/tensor.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace brodiffusion::ardy {

namespace {

// ardy applies `.strip()` to "!@#$%^&*()" + prompt inside
// prepare_for_tokenization; the leading '!' guards the front, so the net effect
// is trailing-whitespace removal on the prompt. Leading whitespace is kept.
std::string rstrip(const std::string& s) {
    std::size_t end = s.size();
    while (end > 0) {
        const unsigned char c = static_cast<unsigned char>(s[end - 1]);
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
            c == '\v') {
            --end;
        } else {
            break;
        }
    }
    return s.substr(0, end);
}

// The Llama-3-Instruct user turn LLM2Vec wraps the prompt in
// (prepare_for_tokenization). BOS is added by the tokenizer's add_bos, and the
// `!@#$%^&*()` instruction delimiter (empty instruction) is stripped back out by
// tokenize before the model sees it, so it is absent here.
constexpr const char* kHeader = "<|start_header_id|>user<|end_header_id|>\n\n";
constexpr const char* kEot    = "<|eot_id|>";

// Download a compute-dtype tensor to a host FP32 vector.
std::vector<float> download_f32(const brotensor::Tensor& t) {
    if (t.dtype == brotensor::Dtype::FP16) {
        std::vector<std::uint16_t> bits = t.to_host_vector_fp16();
        std::vector<float> out(bits.size());
        for (std::size_t i = 0; i < bits.size(); ++i) {
            out[i] = brotensor::fp16_bits_to_fp32(bits[i]);
        }
        return out;
    }
    return t.to_host_vector();
}

}  // namespace

void build_ardy_text_tokens(const brolm::llama3::Tokenizer& tok,
                            const std::string& prompt,
                            std::vector<std::int32_t>& ids,
                            std::vector<float>& pool_mask) {
    const std::string content = rstrip(prompt);

    // Full sequence: <|begin_of_text|> + user header + prompt + <|eot_id|>.
    // The tokenizer emits the <|...|> control tokens atomically.
    ids = tok.encode(std::string(kHeader) + content + kEot, /*add_bos=*/true);

    // The pooled span is exactly the trailing {prompt + <|eot_id|>} tokens,
    // sized by tokenizing that tail alone WITHOUT a BOS (ardy's embed_mask uses
    // add_special_tokens=False). Marking the last N positions mirrors ardy's
    // `e_m[-len(ids):] = 1` and is robust to any BPE boundary at the header
    // join (the tail-only tokenization is what ardy counts, too).
    const std::vector<std::int32_t> tail_ids =
        tok.encode(content + kEot, /*add_bos=*/false);

    pool_mask.assign(ids.size(), 0.0f);
    std::size_t n = tail_ids.size();
    if (n > ids.size()) n = ids.size();
    for (std::size_t i = ids.size() - n; i < ids.size(); ++i) {
        pool_mask[i] = 1.0f;
    }
}

void ardy_text_feat(const brolm::llama3::Tokenizer& tok,
                    brolm::llm2vec::Encoder& enc,
                    const std::string& prompt,
                    std::vector<float>& out) {
    std::vector<std::int32_t> ids;
    std::vector<float> pool_mask;
    build_ardy_text_tokens(tok, prompt, ids, pool_mask);
    if (ids.empty()) throw std::runtime_error("ardy_text_feat: empty token sequence");

    brotensor::Tensor emb;
    enc.encode_pooled(ids.data(), static_cast<int>(ids.size()), emb,
                      pool_mask.data());
    out = download_f32(emb);
}

}  // namespace brodiffusion::ardy
