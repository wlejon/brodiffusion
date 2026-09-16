#pragma once

#include "embed/embed.h"
#include "host_class.h"
#include "object_builder.h"
#include "arg_reader.h"

#include <brodiffusion/pipeline.h>
#include <brodiffusion/controlnet.h>
#include <brodiffusion/vae.h>
#include <brodiffusion/triposplat/flow_model.h>
#include <brodiffusion/triposplat/octree_decoder.h>
#include <brodiffusion/triposplat/sampler.h>
#include <brodiffusion/triposplat/vae_encoder.h>
#include <brotensor/tensor.h>
#include <brotensor/safetensors.h>
#include <brotensor/runtime.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace brodiffusion::api {

namespace ev = bronze::embed;
using Value = bronze::Value;

inline constexpr uint32_t kHostPipelineTag      = 0x44494650u; // 'DIFP'
inline constexpr uint32_t kHostPipelineStateTag = 0x44494653u; // 'DIFS'
inline constexpr uint32_t kHostTripoSplatTag    = 0x5453504Cu; // 'TSPL'
inline constexpr uint32_t kHostVaeTag           = 0x44564145u; // 'DVAE'

struct PipelineWrapper {
    uint32_t tag = kHostPipelineTag;
    std::unique_ptr<brodiffusion::pipeline::Pipeline> pipeline;
    bool weights_loaded = false;
};

struct PipelineStateWrapper {
    uint32_t tag = kHostPipelineStateTag;
    brodiffusion::pipeline::PipelineState state;
    brodiffusion::pipeline::GenerateOptions opts;
};

struct TripoSplatWrapper {
    uint32_t tag = kHostTripoSplatTag;
    std::unique_ptr<brodiffusion::triposplat::FlowDiT> flow;
    std::unique_ptr<brodiffusion::triposplat::OctreeGaussianDecoder> decoder;
    std::unique_ptr<brodiffusion::triposplat::Flux2VaeEncoder> vae;
    brotensor::Device device = brotensor::Device::CPU;
    brodiffusion::triposplat::GaussianSplats lastSplats;
};

struct VaeWrapper {
    uint32_t tag = kHostVaeTag;
    std::unique_ptr<brodiffusion::vae::Decoder> decoder;
    std::unique_ptr<brodiffusion::vae::Encoder> encoder;
    bool weights_loaded = false;
};

extern std::atomic<bool> g_diffusionCancelRequested;
extern std::atomic<bool> g_triposplatCancelRequested;

extern HostClass g_pipelineClass;
extern HostClass g_pipelineStateClass;
extern HostClass g_tripoSplatClass;
extern HostClass g_vaeClass;

PipelineWrapper* unwrapPipeline(Value v);
PipelineStateWrapper* unwrapPipelineState(Value v);
TripoSplatWrapper* unwrapTripoSplat(Value v);
VaeWrapper* unwrapVae(Value v);

void ensureDiffusionClassesInstalled();
void ensureTriposplatClassesInstalled();
void ensureVaeClassesInstalled();

Value makeDiffusionNamespace();
Value makeTriposplatNamespace();

Value makeFloat32Array(const float* data, size_t count);
Value makeUint8ClampedArray(const uint8_t* data, size_t count);
Value makeImageResult(const std::vector<float>& nchw, int H, int W, bool includeFp32 = false);

bool readFloat32Array(Value val, const float*& outData, size_t& outCount);
bool readUint8Array(Value val, const uint8_t*& outData, size_t& outCount);
bool readImageInput(Value val, std::vector<uint8_t>& rgba, int& w, int& h, std::string& err);

bool saveSplatPLY(const brodiffusion::triposplat::GaussianSplats& splats, const std::string& path);
bool saveSplatBinary(const brodiffusion::triposplat::GaussianSplats& splats, const std::string& path);

} // namespace brodiffusion::api
