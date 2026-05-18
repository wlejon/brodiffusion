#pragma once

// LoRA (Low-Rank Adaptation) loader for brodiffusion.
//
// Supports two on-disk key conventions seen in the wild:
//
//   1. kohya-ss / A1111 / civitai. Keys start with `lora_unet_` or `lora_te_`
//      and the diffusers tensor path follows with every '.' replaced by '_'.
//      Suffix is `.lora_down.weight`, `.lora_up.weight`, `.alpha`.
//
//   2. diffusers / PEFT. Keys start with `unet.` or `text_encoder.` and the
//      diffusers tensor path follows verbatim. Suffix is `.lora_A.weight`
//      ( ≡ lora_down) or `.lora_B.weight` ( ≡ lora_up). PEFT exports
//      occasionally interpose a `default.` between `lora_A` / `lora_B` and
//      `.weight` — we accept both.
//
// The merge formula folded into the base weight is:
//
//   W_base += (alpha / rank) * (lora_up @ lora_down) * user_scale
//
// where `rank = lora_down.shape[0]` and `alpha` defaults to `rank` if the
// `.alpha` scalar is absent (PEFT often omits it).
//
// Only attention and feed-forward projections are addressed — community
// LoRAs almost universally adapt these only. The dominant target is the
// `latent-consistency/lcm-lora-sdv1-5` checkpoint, which patches only the
// four UNet attention projections per Transformer2D block (attn1.{q,k,v,
// to_out.0} and attn2.{q,k,v,to_out.0}).

#include <string>
#include <vector>

namespace brodiffusion::safetensors { class File; struct TensorView; }

namespace brodiffusion::lora {

enum class Format {
    Kohya,
    Diffusers,
};

// Inspect the file's keys and decide which naming convention dominates.
// Throws if neither pattern is detected with at least one matching key.
Format detect_format(const safetensors::File& f);

// One LoRA target: a base weight in either the UNet or the CLIP text encoder
// plus the (lora_down, lora_up) raw keys in the safetensors file and the
// resolved (alpha, rank). The caller looks up the views and dispatches to
// UNet::apply_lora_delta / TextEncoder::apply_lora_delta.
struct Triple {
    std::string domain;       // "unet" or "text_encoder"
    std::string target_path;  // diffusers path *within* the domain, e.g.
                              //   "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q"
                              //   "encoder.layers.0.self_attn.q_proj"
    std::string down_key;     // raw key for lora_down / lora_A in `f`
    std::string up_key;       // raw key for lora_up   / lora_B in `f`
    float       alpha = 0.0f; // resolved (= rank if .alpha absent)
    int         rank  = 0;    // = lora_down.shape[0]
};

// Enumerate every LoRA triple in `f` whose target path matches one of the
// supported attention/FF projections. Unknown or unsupported keys are
// skipped silently if they look like LoRA bookkeeping (e.g. dora_scale,
// rank metadata), and reported as a single grouped error otherwise.
std::vector<Triple> enumerate(const safetensors::File& f);

// Test-only helper: translate a diffusers path "unet.<path>" or
// "text_encoder.text_model.<path>" into its kohya equivalent (lora_unet_<...>
// or lora_te_<...>, with every '.' inside `<path>` replaced by '_').
std::string diffusers_to_kohya_prefix(const std::string& domain,
                                      const std::string& target_path);

// Test-only helper: split a kohya key prefix and recover (domain, target_path)
// using the inverse-map approach (enumerate expected diffusers paths and
// match the underscored form). Returns true on success.
bool kohya_to_diffusers(const std::string& kohya_prefix,
                        std::string& domain,
                        std::string& target_path);

// Build the canonical list of diffusers target paths the loader recognizes,
// keyed by domain. Used internally; exposed for tests.
struct TargetPath {
    std::string domain;       // "unet" or "text_encoder"
    std::string path;         // diffusers path within the domain
};
std::vector<TargetPath> known_targets();

}  // namespace brodiffusion::lora
