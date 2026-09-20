#pragma once

// Internal helpers shared by the three QwenImage21 translation units
// (qwenimage21.cpp — loader, qwenimage21_forward.cpp — forward,
// qwenimage21_denoiser.cpp — Denoiser wrapper). Not a public header.

#include "brotensor/tensor.h"

#include <string>
#include <vector>

namespace brotensor::safetensors { class File; struct TensorView; }

namespace brodiffusion::dit::qi21 {

// Throw with the model's name prefixed.
[[noreturn]] void fail(const std::string& msg);

// First shard carrying `key`; throws when no shard has it.
const brotensor::safetensors::TensorView& need(
    const std::vector<const brotensor::safetensors::File*>& shards,
    const std::string& key);

// Download a checked view to host FP32 regardless of storage dtype.
std::vector<float> view_to_fp32(const brotensor::safetensors::TensorView& v,
                                int rows, int cols, const std::string& name);

// Upload host FP32 values at the given dtype, on the default device.
brotensor::Tensor upload_as(const std::vector<float>& h, int rows, int cols,
                            brotensor::Dtype dt);

// Non-owning view over rows [start, start+n) of a row-major tensor. Safe as a
// brotensor op OUTPUT only when the op's result shape is exactly (n, t.cols)
// at t.dtype — a matching resize() is a no-op, a mismatching one throws.
brotensor::Tensor row_view(const brotensor::Tensor& t, int start, int n);

}  // namespace brodiffusion::dit::qi21
