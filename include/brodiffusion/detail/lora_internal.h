#pragma once

// Internal LoRA key-format helpers — kohya <-> diffusers prefix conversion.
//
// Not part of the public brodiffusion::lora API (detect_format / enumerate).
// Declared here so the LoRA test suite can exercise the round-trip directly
// without widening lora.h's surface.

#include <string>

namespace brodiffusion::lora {

// Translate a diffusers path "unet.<path>" / "text_encoder.text_model.<path>"
// into its kohya equivalent (lora_unet_<...> / lora_te_<...>, with every '.'
// inside <path> replaced by '_').
std::string diffusers_to_kohya_prefix(const std::string& domain,
                                      const std::string& target_path);

// Split a kohya key prefix back into (domain, target_path). Returns false if
// the prefix is not a recognized kohya LoRA key.
bool kohya_to_diffusers(const std::string& kohya_prefix,
                        std::string& domain,
                        std::string& target_path);

}  // namespace brodiffusion::lora
