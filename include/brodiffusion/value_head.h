#pragma once

// Tiny FP32 MLP value head — the "internalise MCTS" experiment.
//
// Input  : flattened latent at decision t (C_lat * H_lat * W_lat floats),
//          concatenated with a B-dim one-hot action encoding.
// Output : a scalar prediction of the terminal CLIP score (or whatever
//          scorer produced the trace labels).
//
// Architecture is deliberately small (one hidden layer, ReLU). This is
// experimental scaffolding: if the head can't fit the training set we
// delete the files and try something else, not bolt on attention.
//
// All compute is FP32. We bypass AdamFP16 and call brotensor::adam_step_gpu
// directly on FP32 parameters — saves the FP16↔FP32 fold-back round-trip
// that's pointless at this scale.

#include "brotensor/tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace brodiffusion::value_head {

struct Config {
    int latent_dim     = 4 * 64 * 64;   // 4 * H_lat * W_lat for 512x512 SD
    int branching      = 3;             // B (action one-hot width)
    int hidden_dim     = 128;
    float lr           = 1e-3f;
    float beta1        = 0.9f;
    float beta2        = 0.999f;
    float eps          = 1e-8f;
    float init_std_W1  = 0.01f;         // small init — input is ~16K-wide
    float init_std_W2  = 0.1f;
    std::uint64_t seed = 1;
};

class ValueHead {
public:
    explicit ValueHead(const Config& cfg);

    int in_dim() const { return cfg_.latent_dim + cfg_.branching; }

    // Forward over a minibatch. X_BD is (B_batch, in_dim) FP32.
    // Resizes Y_B1 to (B_batch, 1). Caches activations for backward.
    void forward(const brotensor::GpuTensor& X_BD, brotensor::GpuTensor& Y_B1);

    // Backward + one Adam step. dY_B1 is (B_batch, 1) FP32 (typically
    // (pred - target) for MSE). Internally zeros gradient scratch, computes
    // dW1/db1/dW2/db2 over the cached forward, then calls adam_step_gpu.
    void backward_and_step(const brotensor::GpuTensor& dY_B1);

    // Save/load FP32 weights as a flat binary blob:
    //   header: i32 magic = 'BDVH', i32 version=1, i32 latent_dim,
    //           i32 branching, i32 hidden_dim
    //   then: W1 (hidden, in_dim), b1 (hidden, 1), W2 (1, hidden), b2 (1, 1)
    //   all FP32 row-major.
    void save(const std::string& path) const;
    void load(const std::string& path);

    int step_count() const { return step_n_; }

private:
    Config cfg_;

    // Parameters (FP32, on device).
    brotensor::GpuTensor W1_, b1_;
    brotensor::GpuTensor W2_, b2_;

    // Adam state (FP32, on device, matched shape).
    brotensor::GpuTensor m_W1_, v_W1_, m_b1_, v_b1_;
    brotensor::GpuTensor m_W2_, v_W2_, m_b2_, v_b2_;

    // Gradient scratch (FP32, on device, accumulator-style: zeroed each step).
    brotensor::GpuTensor g_W1_, g_b1_, g_W2_, g_b2_;

    // Forward cache (re-used across forwards).
    brotensor::GpuTensor X_cached_;     // (B, in_dim)  — non-owning view? owned for simplicity
    brotensor::GpuTensor h_pre_;        // (B, hidden)  pre-ReLU
    brotensor::GpuTensor h_;            // (B, hidden)  post-ReLU
    brotensor::GpuTensor dH_, dHpre_;   // backward scratch (B, hidden)
    brotensor::GpuTensor dX_unused_;    // (B, in_dim)  — required by op, discarded

    int step_n_ = 0;
};

}  // namespace brodiffusion::value_head
