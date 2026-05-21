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
#include "brodiffusion/clip.h"
#include "brodiffusion/scheduler.h"
#include "brodiffusion/lcm_scheduler.h"
#include "brodiffusion/flow_match_scheduler.h"

#include <string>
#include <variant>

namespace brodiffusion {

enum class ModelClass { StableDiffusion, Flux, Unknown };

// Architecture + hyper-parameters of a diffusers model directory, read from
// its JSON config files. The Pipeline factory consumes this to build the
// right sub-modules.
struct ModelConfig {
    ModelClass  model_class = ModelClass::Unknown;
    std::string model_dir;            // absolute/relative root passed in

    unet::UNetConfig        unet;          // populated for StableDiffusion
    vae::DecoderConfig      vae;           // always populated
    clip::TextEncoderConfig text_encoder;  // populated when a CLIP encoder exists

    std::variant<scheduler::DDIMConfig,
                 scheduler::LCMConfig,
                 scheduler::FlowMatchConfig> scheduler;

    // NOTE: the Flux transformer config and the T5 text-encoder config are
    // populated by later phases (they extend this struct + loader). Phase 3
    // detects ModelClass::Flux and populates `vae` + `scheduler` for it.
};

// Read `model_index.json` + each component config from `model_dir`.
// Throws std::runtime_error on a missing/malformed `model_index.json`. A
// component subdirectory that does not exist is simply skipped.
ModelConfig load_model_config(const std::string& model_dir);

}  // namespace brodiffusion
