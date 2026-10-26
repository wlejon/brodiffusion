#pragma once

#include "embed/embed.h"
#include "host_class.h"
#include "object_builder.h"
#include "arg_reader.h"

#include <brodiffusion/pipeline.h>
#include <brodiffusion/controlnet.h>
#include <brodiffusion/denoiser.h>
#include <brodiffusion/krea2_text.h>
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
#include <initializer_list>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace brovisionml::dinov3 {
class Backbone;
}
namespace brovisionml::birefnet {
class BiRefNet;
}

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
    std::string scheduler_name;
    std::atomic<bool> cancel_requested{false};
    // True while a background generate owns the pipeline: the synchronous
    // entry points refuse it rather than drive the same weights and scratch
    // from two threads, and dispose() waits the job out.
    std::atomic<bool> busy{false};

    PipelineWrapper();
    ~PipelineWrapper();
};

struct PipelineStateWrapper {
    uint32_t tag = kHostPipelineStateTag;
    brodiffusion::pipeline::PipelineState state;
    brodiffusion::pipeline::GenerateOptions opts;
};

struct TripoSplatWrapper {
    uint32_t tag = kHostTripoSplatTag;
    std::unique_ptr<brovisionml::dinov3::Backbone> dino;
    std::unique_ptr<brodiffusion::triposplat::FlowDiT> flow;
    std::unique_ptr<brodiffusion::triposplat::OctreeGaussianDecoder> decoder;
    std::unique_ptr<brodiffusion::triposplat::Flux2VaeEncoder> vae;
    // Optional BiRefNet matte model (load({ birefnet })): generate() isolates
    // the subject before encoding, unless opts.removeBackground is false.
    std::unique_ptr<brovisionml::birefnet::BiRefNet> rmbg;
    brotensor::Device device = brotensor::Device::CPU;
    brodiffusion::triposplat::GaussianSplats lastSplats;
    std::atomic<bool> cancel_requested{false};
};

struct VaeWrapper {
    uint32_t tag = kHostVaeTag;
    std::unique_ptr<brodiffusion::vae::Decoder> decoder;
    std::unique_ptr<brodiffusion::vae::Encoder> encoder;
    bool weights_loaded = false;
};

extern HostClass g_pipelineClass;
extern HostClass g_pipelineStateClass;
extern HostClass g_tripoSplatClass;
extern HostClass g_vaeClass;

PipelineWrapper* unwrapPipeline(Value v);

inline constexpr const char* kPipelineBusy =
    "a background generate is in flight on this Pipeline (cancel() it or wait for onDone)";

// Cancel every background generate on `pw` and join its thread; the jobs'
// onDone still fires ({cancelled: true}) on the next tick.
void cancelAndJoinPipelineJobs(PipelineWrapper* pw);
PipelineStateWrapper* unwrapPipelineState(Value v);
TripoSplatWrapper* unwrapTripoSplat(Value v);
VaeWrapper* unwrapVae(Value v);

void ensureDiffusionClassesInstalled();
void ensureTriposplatClassesInstalled();
void ensureVaeClassesInstalled();

Value makeDiffusionNamespace();

// bro.diffusion.loadTerrain and its TerrainWorld class
// (native_diffusion_terrain.cpp).
void ensureTerrainClassInstalled();
Value terrainConstructor();
Value loadTerrain(Value thisVal, std::span<const Value> args);
Value makeTriposplatNamespace();

Value makeFloat32Array(const float* data, size_t count);
Value makeUint8ClampedArray(const uint8_t* data, size_t count);
Value makeImageResult(const std::vector<float>& nchw, int H, int W, bool includeFp32 = false);

bool readFloat32Array(Value val, const float*& outData, size_t& outCount);
bool readUint8Array(Value val, const uint8_t*& outData, size_t& outCount);

// A model dir / weight file / LoRA / ControlNet / control-dictionary path as
// the host's resolver sees it (api.h setPathResolver), or unchanged when no
// resolver is installed.
std::string resolveDiffusionPath(const std::string& path);

// The host's access gate (embed/embed_access.h; api.h, "Access"). A native
// asks about every path the app handed it, as the library will open it (after
// resolveDiffusionPath where the native resolves), before anything is opened
// or stat'ed: Read for what is read (a directory covers what is in it), Write
// for what is written. The gate is handed the path absolute and lexically
// normal, in UTF-8; empty paths are skipped. A set PathRefusal carries the
// gate's sentence and that path.
struct PathRefusal {
    std::string message, path;
    explicit operator bool() const { return !message.empty(); }
};
PathRefusal checkReadAccess(std::initializer_list<std::string_view> paths);
PathRefusal checkWriteAccess(std::string_view path);
// Every image a GenerateOptions reads: initImagePath, maskImagePath, each
// control's imagePath and each condition image's path.
PathRefusal checkImagesAccess(const brodiffusion::pipeline::GenerateOptions& o);
PathRefusal checkImagesAccess(const std::vector<brodiffusion::pipeline::ConditionImage>& images);
// The refusal thrown into the running program (Error, code EACCES, path).
Value throwRefusal(const PathRefusal& refusal);

// ── shared by the pipeline / control / krea2 / state translation units ─────

// Download a brotensor::Tensor to host FP32, converting FP16/BF16 bits as
// needed — brodiffusion tensors carry the compute dtype.
std::vector<float> downloadTensorFloats(const brotensor::Tensor& t);

// { rows, cols, data: Float32Array } ↔ brotensor::Tensor (host FP32).
bool tensorFromJs(Value v, brotensor::Tensor& out);
Value tensorToJs(const brotensor::Tensor& t);

// krea2::TextConditioning -> { embeds, mask }.
Value textConditioningToJs(const brodiffusion::krea2::TextConditioning& tc);

// Map a JS opts object onto GenerateOptions (defaults kept for absent keys).
brodiffusion::pipeline::GenerateOptions parseGenerateOptions(Value v);

// Fill in a canvas the caller asked to inherit: Qwen-Image 2.1 condition
// images with no explicit width/height leave both at 0, meaning "take the
// last image's aspect at outputResolution". prime() applies the same rule
// internally, but the binding needs the numbers too — they are what the
// returned image object is described by. No-op when the size is already set.
void resolveDerivedSize(brodiffusion::pipeline::Pipeline& p,
                        brodiffusion::pipeline::GenerateOptions& o);

// A PipelineState retains its owning Pipeline through a `__pipeline` property
// so the weights cannot be collected while a state (or a clone) is alive.
Value attachPipelineToState(Value stateVal, Value pipelineVal);
brodiffusion::pipeline::Pipeline* pipelineOfState(Value stateVal);
PipelineWrapper* pipelineWrapperOfState(Value stateVal);  // ALLOCATES (a property read)

// Prototype decoration, split across TUs to keep each file small.
void decoratePipelineControlProto(ObjectBuilder& proto);
void decoratePipelineKrea2Proto(ObjectBuilder& proto);
void decoratePipelineQwenImage21Proto(ObjectBuilder& proto);
void decoratePipelineStepProto(ObjectBuilder& proto);
void decoratePipelineStateProto(ObjectBuilder& proto);
bool readImageInput(Value val, std::vector<uint8_t>& rgba, int& w, int& h, std::string& err);

bool saveSplatPLY(const brodiffusion::triposplat::GaussianSplats& splats, const std::string& path);
bool saveSplatBinary(const brodiffusion::triposplat::GaussianSplats& splats, const std::string& path);

} // namespace brodiffusion::api
