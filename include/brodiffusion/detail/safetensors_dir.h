#pragma once

// safetensors_dir — open every *.safetensors file in a diffusers component
// directory.
//
// A diffusers component directory holds either a single `*.safetensors` file
// or a *sharded* set (`*-00001-of-0000N.safetensors` plus a `*.index.json`
// weight-map). brotensor::safetensors::File opens exactly one file; this
// helper opens all of them so a component loader can search every shard.
//
// We deliberately do NOT parse the `.index.json` weight-map: searching every
// shard by tensor name makes the index unnecessary, and the shard count is
// small (Flux ships 3 transformer shards, T5-XXL 2).

#include "brotensor/safetensors.h"

#include <string>
#include <vector>

namespace brodiffusion::detail {

// Open every `*.safetensors` file in `component_dir` (non-recursive). A
// single-file component yields a one-element vector; a sharded component
// yields one File per shard, sorted by filename for determinism. Throws
// std::runtime_error if the directory contains no `*.safetensors` file.
std::vector<brotensor::safetensors::File>
open_component_files(const std::string& component_dir);

}  // namespace brodiffusion::detail
