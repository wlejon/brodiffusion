#pragma once

// model_config — read a diffusers model directory's JSON config files into
// brodiffusion's architecture/hyper-parameter structs.
//
// A diffusers model directory has `model_index.json` at the root plus one
// subdirectory per component (`unet/`, `vae/`, `text_encoder/`, `scheduler/`,
// …), each with a `config.json` — except the scheduler component, whose file
// is named `scheduler_config.json`.
//
// load_model_config() reads `_class_name` from model_index.json to determine
// the model class, then parses whichever component configs are present and
// populates a ModelConfig. The Pipeline factory consumes this to build the
// right sub-modules.

#include "brodiffusion/unet.h"
#include "brodiffusion/vae.h"
#include "brodiffusion/vae_dcae.h"
#include "brolm/clip.h"
#include "brodiffusion/scheduler.h"
#include "brodiffusion/lcm_scheduler.h"
#include "brodiffusion/flow_match_scheduler.h"
#include "brodiffusion/scm_scheduler.h"
#include "brodiffusion/dpm_solver.h"
#include "brodiffusion/dit/flux.h"
#include "brodiffusion/dit/sana.h"
#include "brodiffusion/dit/pixart.h"
#include "brolm/t5.h"
#include "brolm/gemma2_config.h"

#include <string>
#include <variant>

namespace brodiffusion {

enum class ModelClass { StableDiffusion, Flux, Sana, PixArt, Unknown };

// Architecture + hyper-parameters of a diffusers model directory, read from
// its JSON config files. The Pipeline factory consumes this to build the
// right sub-modules.
struct ModelConfig {
    ModelClass  model_class = ModelClass::Unknown;
    std::string model_dir;            // absolute/relative root passed in

    unet::UNetConfig        unet;          // populated for StableDiffusion
    vae::DecoderConfig      vae;           // always populated
    brolm::clip::TextEncoderConfig text_encoder;  // populated when a CLIP encoder exists

    dit::FluxConfig flux;   // populated for ModelClass::Flux
    brolm::t5::T5Config t5; // populated for ModelClass::Flux (the second text encoder)
    int             t5_max_length = 512;  // T5 sequence length (flux-schnell 256, dev 512)

    dcae::DecoderConfig dcae;          // populated for ModelClass::Sana (DC-AE f32c32 VAE)
    dit::SanaConfig     sana;          // populated for ModelClass::Sana (the transformer)
    brolm::gemma::Gemma2Config gemma;  // populated for ModelClass::Sana (the Gemma-2 text encoder)
    int             sana_max_seq_len = 300;  // Gemma caption sequence length

    dit::PixArtConfig pixart;          // populated for ModelClass::PixArt (the DiT)
    // PixArt-Sigma shares Flux's T5-XXL text encoder (the `t5` field above) but
    // caps captions at 300 tokens. The encoder weights are NOT bundled in a
    // PixArt model dir — from_model_dir resolves them from a sibling t5-xxl dir.

    std::variant<scheduler::DDIMConfig,
                 scheduler::LCMConfig,
                 scheduler::FlowMatchConfig,
                 scheduler::SCMConfig,
                 scheduler::DPMSolverConfig> scheduler;
};

// Read `model_index.json` + each component config from `model_dir`.
// Throws std::runtime_error on a missing/malformed `model_index.json`. A
// component subdirectory that does not exist is simply skipped.
ModelConfig load_model_config(const std::string& model_dir);

}  // namespace brodiffusion
