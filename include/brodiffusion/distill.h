#pragma once

// In-tree distillation trainer for the SD1.5 UNet Inlet.
//
// Loads captured teacher activations (one safetensors file per
// (prompt, step, branch) emitted by `brodiffusion capture-inlet`), runs the
// inlet forward, computes per-tap MSE against the captured skips, backprops
// through the inlet graph, and steps an Adam optimiser. FP16 weights with an
// FP32 master copy + FP32 Adam state.

#include "brodiffusion/inlet.h"
#include "brotensor/tensor.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace brodiffusion::distill {

struct TrainOptions {
    int   steps        = 5000;
    float lr           = 1e-3f;
    float beta1        = 0.9f;
    float beta2        = 0.999f;
    float eps          = 1e-8f;
    std::uint64_t shuffle_seed = 0;
    int   log_every    = 50;
    int   ckpt_every   = 1000;
    std::array<float, 12> loss_weights = {1,1,1,1,1,1,1,1,1,1,1,1};
};

// Train an inlet against captures in `capture_dir`, write final weights to
// `out_path`. Intermediate checkpoints written to `out_path + ".step<N>"`.
// If `init_path` is non-empty, warm-starts from that safetensors file.
int run_distill(const std::string& capture_dir,
                const std::string& out_path,
                const std::string& init_path,
                const TrainOptions& opts);

// Save the inlet's weights to `path` as safetensors with the key convention
// that Inlet::load_from_safetensors(f, prefix="") accepts.
void save_inlet_safetensors(const inlet::Inlet& net, const std::string& path);

}  // namespace brodiffusion::distill
