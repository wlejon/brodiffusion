// bro.diffusion — the Pipeline class (one-shot + step-wise entry points) and
// the namespace it hangs off. Value translation lives in
// native_diffusion_values.cpp; the conditioning-control and Krea 2 research
// surfaces decorate the same prototype from native_diffusion_control.cpp and
// native_diffusion_krea2.cpp; PipelineState lives in native_diffusion_state.cpp.

#include "host_diffusion_internal.h"
#include <brodiffusion/version.h>
#include <brodiffusion/scheduler.h>
#include <brodiffusion/lcm_scheduler.h>
#include <brodiffusion/flow_match_scheduler.h>
#include <brodiffusion/scm_scheduler.h>
#include <brodiffusion/dpm_solver.h>
#include <brolm/tokenizer.h>
#include <brotensor/runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <variant>

namespace brodiffusion::api {

// ---------------------------------------------------------------------------
// Pipeline lifecycle tracking
// ---------------------------------------------------------------------------

static std::mutex g_pipelineRegMtx;
static std::vector<PipelineWrapper*> g_activePipelines;

PipelineWrapper::PipelineWrapper() {
    std::lock_guard<std::mutex> lock(g_pipelineRegMtx);
    g_activePipelines.push_back(this);
}

PipelineWrapper::~PipelineWrapper() {
    std::lock_guard<std::mutex> lock(g_pipelineRegMtx);
    auto it = std::find(g_activePipelines.begin(), g_activePipelines.end(), this);
    if (it != g_activePipelines.end()) g_activePipelines.erase(it);
}

// ---------------------------------------------------------------------------
// Diffusion async-job machine
// ---------------------------------------------------------------------------

struct DiffusionWork {
    std::atomic<bool> cancel{false};
    std::atomic<bool> finished{false};
    std::vector<float> nchw;
    int height = 0, width = 0;
    bool includeFp32 = false, cancelled = false;
    std::string error;
};

struct DiffusionJob {
    std::shared_ptr<DiffusionWork> work;
    std::thread th;
    ev::Persistent onDone, pipelineRef;
    PipelineWrapper* pw = nullptr;
    ~DiffusionJob() { if (th.joinable()) th.join(); }
};

static thread_local std::vector<std::unique_ptr<DiffusionJob>> s_diffusionJobs;
static thread_local std::mutex s_diffusionJobsMtx;

static void cancelJobsForPipeline(PipelineWrapper* pw) {
    if (pw) pw->cancel_requested.store(true, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(s_diffusionJobsMtx);
    for (auto& job : s_diffusionJobs) {
        if (!pw || job->pw == pw) job->work->cancel.store(true, std::memory_order_release);
    }
}

static void cancelAllDiffusionJobs() {
    {
        std::lock_guard<std::mutex> lock(g_pipelineRegMtx);
        for (auto* pw : g_activePipelines) if (pw) pw->cancel_requested.store(true, std::memory_order_relaxed);
    }
    std::lock_guard<std::mutex> lock(s_diffusionJobsMtx);
    for (auto& job : s_diffusionJobs) job->work->cancel.store(true, std::memory_order_release);
}

void tickDiffusionAsync() {
    std::vector<std::unique_ptr<DiffusionJob>> finished;
    {
        std::lock_guard<std::mutex> lock(s_diffusionJobsMtx);
        for (auto it = s_diffusionJobs.begin(); it != s_diffusionJobs.end();) {
            if ((*it)->work->finished.load(std::memory_order_acquire)) {
                finished.push_back(std::move(*it));
                it = s_diffusionJobs.erase(it);
            } else {
                ++it;
            }
        }
    }
    for (auto& job : finished) {
        if (job->th.joinable()) job->th.join();
        const bool cancelled = job->work->cancelled || job->work->cancel.load(std::memory_order_acquire);
        const std::string& err = job->work->error;

        ev::Persistent info(ev::createObject());
        {
            ObjectBuilder b(info.get());
            b.set("cancelled", ev::fromBool(cancelled));
            if (!err.empty()) b.set("error", ev::fromUtf8(err));
            info.set(b.get());
        }

        ev::Persistent result(ev::null());
        if (!cancelled && err.empty() && !job->work->nchw.empty()) {
            result.set(makeImageResult(job->work->nchw, job->work->height, job->work->width, job->work->includeFp32));
        } else if (cancelled) {
            ObjectBuilder b;
            b.set("cancelled", true);
            result.set(b.build());
        }

        Value onDone = job->onDone.get();
        if (ev::isFunction(onDone)) {
            const Value args[2] = {result.get(), info.get()};
            ev::CallResult r = ev::call(onDone, ev::undefined(), std::span<const Value>(args, 2));
            (void)r;
        }

        job->onDone.set(ev::undefined());
        job->pipelineRef.set(ev::undefined());
    }
}

void shutdownDiffusionAsync() {
    std::vector<std::unique_ptr<DiffusionJob>> all;
    {
        std::lock_guard<std::mutex> lock(s_diffusionJobsMtx);
        all.swap(s_diffusionJobs);
    }
    for (auto& job : all) {
        job->work->cancel.store(true, std::memory_order_release);
        if (job->th.joinable()) job->th.join();
        job->onDone.set(ev::undefined());
        job->pipelineRef.set(ev::undefined());
    }
}

static Value dispatchGenerateAsync(Value thisVal, PipelineWrapper* w, std::string prompt,
                                  brodiffusion::pipeline::GenerateOptions opts,
                                  bool includeFp32, Value onDoneVal) {
    auto work = std::make_shared<DiffusionWork>();
    work->height = opts.height;
    work->width = opts.width;
    work->includeFp32 = includeFp32;

    auto job = std::make_unique<DiffusionJob>();
    job->work = work;
    job->pw = w;
    job->pipelineRef = ev::Persistent(thisVal);
    if (ev::isFunction(onDoneVal)) {
        job->onDone = ev::Persistent(onDoneVal);
    }

    w->cancel_requested.store(false, std::memory_order_relaxed);

    job->th = std::thread([work, w, prompt = std::move(prompt), opts = std::move(opts)]() mutable {
        try {
            opts.should_cancel = [w, work]() {
                return w->cancel_requested.load(std::memory_order_relaxed) ||
                       work->cancel.load(std::memory_order_relaxed);
            };
            work->nchw = w->pipeline->generate(prompt, opts);
        } catch (const brodiffusion::pipeline::GenerateCancelled&) {
            work->cancelled = true;
        } catch (const std::exception& e) {
            work->error = e.what();
        } catch (...) {
            work->error = "unknown error in generate";
        }
        work->finished.store(true, std::memory_order_release);
    });

    {
        std::lock_guard<std::mutex> lock(s_diffusionJobsMtx);
        s_diffusionJobs.push_back(std::move(job));
    }

    ObjectBuilder h;
    h.def("cancel", 0, [work, w](Value, std::span<const Value>) -> Value {
        work->cancel.store(true, std::memory_order_release);
        if (w) w->cancel_requested.store(true, std::memory_order_relaxed);
        return ev::undefined();
    });
    h.accessor("done", [work](Value, std::span<const Value>) -> Value {
        return ev::fromBool(work->finished.load(std::memory_order_acquire));
    }, nullptr);
    h.def("wait", 0, [work](Value, std::span<const Value>) -> Value {
        while (!work->finished.load(std::memory_order_acquire)) {
            tickDiffusionAsync();
            if (work->finished.load(std::memory_order_acquire)) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        tickDiffusionAsync();
        return ev::undefined();
    });
    return h.build();
}

namespace {

Value pipelineGenerate(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.generate: not a loaded Pipeline");
    if (!w->weights_loaded) return ev::throwError("Pipeline.generate: call loadWeights() first");
    if (args.empty() || !ev::isString(args[0])) {
        return ev::throwTypeError("Pipeline.generate(prompt, opts?): string prompt required");
    }

    std::string prompt = ev::toUtf8(args[0]);
    Value optVal = args.size() > 1 ? args[1] : ev::undefined();
    ev::Persistent opt(optVal);
    const bool includeFp32 = propBool(opt.get(), "includeFp32");
    auto opts = parseGenerateOptions(opt.get());
    try {
        resolveDerivedSize(*w->pipeline, opts);
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.generate failed: ") + e.what());
    }

    Value onDoneVal = ev::isObject(opt.get()) ? ev::getProperty(opt.get(), "onDone") : ev::undefined();
    const bool isAsync = ev::isFunction(onDoneVal) || propBool(opt.get(), "async");

    if (isAsync) {
        return dispatchGenerateAsync(thisVal, w, std::move(prompt), std::move(opts), includeFp32, onDoneVal);
    }

    try {
        w->cancel_requested.store(false, std::memory_order_relaxed);
        opts.should_cancel = [w]() {
            return w->cancel_requested.load(std::memory_order_relaxed);
        };
        std::vector<float> nchw = w->pipeline->generate(prompt, opts);
        return makeImageResult(nchw, opts.height, opts.width, includeFp32);
    } catch (const brodiffusion::pipeline::GenerateCancelled&) {
        ObjectBuilder b;
        b.set("cancelled", true);
        return b.build();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.generate failed: ") + e.what());
    }
}

Value pipelineGenerateAsync(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.generateAsync: not a loaded Pipeline");
    if (!w->weights_loaded) return ev::throwError("Pipeline.generateAsync: call loadWeights() first");
    if (args.empty() || !ev::isString(args[0])) {
        return ev::throwTypeError("Pipeline.generateAsync(prompt, opts?): string prompt required");
    }

    std::string prompt = ev::toUtf8(args[0]);
    Value optVal = args.size() > 1 ? args[1] : ev::undefined();
    ev::Persistent opt(optVal);
    const bool includeFp32 = propBool(opt.get(), "includeFp32");
    auto opts = parseGenerateOptions(opt.get());
    try {
        resolveDerivedSize(*w->pipeline, opts);
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.generateAsync failed: ") + e.what());
    }
    Value onDoneVal = ev::isObject(opt.get()) ? ev::getProperty(opt.get(), "onDone") : ev::undefined();

    return dispatchGenerateAsync(thisVal, w, std::move(prompt), std::move(opts), includeFp32, onDoneVal);
}

Value pipelineTextToImage(Value thisVal, std::span<const Value> args) {
    return pipelineGenerate(thisVal, args);
}

// Shared body of imageToImage()/inpaint(): both are generate() with the init
// (and mask) image path forced on top of the caller's opts.
Value generateWithImages(Value thisVal, PipelineWrapper* w, const std::string& label,
                         const std::string& prompt, Value optVal,
                         const std::string& initPath, const std::string& maskPath) {
    ev::Persistent opt(optVal);
    const bool includeFp32 = propBool(opt.get(), "includeFp32");
    auto opts = parseGenerateOptions(opt.get());
    opts.init_image_path = initPath;
    if (!maskPath.empty()) opts.mask_image_path = maskPath;

    Value onDoneVal = ev::isObject(opt.get()) ? ev::getProperty(opt.get(), "onDone") : ev::undefined();
    const bool isAsync = ev::isFunction(onDoneVal) || propBool(opt.get(), "async");

    if (isAsync) {
        return dispatchGenerateAsync(thisVal, w, prompt, std::move(opts), includeFp32, onDoneVal);
    }

    try {
        w->cancel_requested.store(false, std::memory_order_relaxed);
        opts.should_cancel = [w]() {
            return w->cancel_requested.load(std::memory_order_relaxed);
        };
        std::vector<float> nchw = w->pipeline->generate(prompt, opts);
        return makeImageResult(nchw, opts.height, opts.width, includeFp32);
    } catch (const brodiffusion::pipeline::GenerateCancelled&) {
        ObjectBuilder b;
        b.set("cancelled", true);
        return b.build();
    } catch (const std::exception& e) {
        return ev::throwError(label + " failed: " + e.what());
    }
}

Value pipelineImageToImage(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.imageToImage: not a loaded Pipeline");
    if (args.size() < 2 || !ev::isString(args[0]) || !ev::isString(args[1])) {
        return ev::throwTypeError("Pipeline.imageToImage(imagePath, prompt, opts?): string imagePath and prompt required");
    }
    return generateWithImages(thisVal, w, "Pipeline.imageToImage", ev::toUtf8(args[1]),
                              args.size() > 2 ? args[2] : ev::undefined(),
                              ev::toUtf8(args[0]), std::string());
}

Value pipelineInpaint(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.inpaint: not a loaded Pipeline");
    if (args.size() < 3 || !ev::isString(args[0]) || !ev::isString(args[1]) || !ev::isString(args[2])) {
        return ev::throwTypeError("Pipeline.inpaint(imagePath, maskPath, prompt, opts?): string imagePath, maskPath, prompt required");
    }
    return generateWithImages(thisVal, w, "Pipeline.inpaint", ev::toUtf8(args[2]),
                              args.size() > 3 ? args[3] : ev::undefined(),
                              ev::toUtf8(args[0]), ev::toUtf8(args[1]));
}

// loadWeights(path)                                    — single-file checkpoint
// loadWeights(path, {textPrefix,unetPrefix,vaePrefix}) — single file, custom prefixes
// loadWeights(textPath, unetPath, vaePath)             — diffusers 3-file export
//
// An absent prefix stays "" (the root of the file), as it did before the
// bronze port: substituting the SD1.5 defaults would silently ignore a caller
// that asked for a root-prefixed module.
Value pipelineLoadWeights(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.loadWeights: not a loaded Pipeline");
    if (args.empty() || !ev::isString(args[0])) {
        return ev::throwTypeError("Pipeline.loadWeights(path, ...): path string required");
    }

    std::string p0 = resolveDiffusionPath(ev::toUtf8(args[0]));
    try {
        if (args.size() >= 3) {
            if (!ev::isString(args[1]) || !ev::isString(args[2])) {
                return ev::throwTypeError(
                    "Pipeline.loadWeights(textPath, unetPath, vaePath): all three must be strings");
            }
            auto tf = brotensor::safetensors::File::open(p0);
            auto uf = brotensor::safetensors::File::open(resolveDiffusionPath(ev::toUtf8(args[1])));
            auto vf = brotensor::safetensors::File::open(resolveDiffusionPath(ev::toUtf8(args[2])));
            w->pipeline->load_weights(tf, uf, vf);
        } else if (args.size() == 2 && ev::isObject(args[1])) {
            ev::Persistent cfg(args[1]);
            std::string tp, up, vp;
            propStr(cfg.get(), "textPrefix", tp);
            propStr(cfg.get(), "unetPrefix", up);
            propStr(cfg.get(), "vaePrefix", vp);
            auto f = brotensor::safetensors::File::open(p0);
            w->pipeline->load_weights(f, tp, up, vp);
        } else {
            auto f = brotensor::safetensors::File::open(p0);
            w->pipeline->load_weights(f);
        }
        w->weights_loaded = true;
        return ev::undefined();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.loadWeights failed: ") + e.what());
    }
}

// reloadTextEncoder(modelDir, textEncoderPath, opts?) — Krea 2 only. Swap just
// the Qwen3-VL-4B text backbone, keeping the resident DiT / VAE / vision tower.
// textEncoderPath "" restores the model dir's bundled encoder.
// opts.quantizeWeights defaults to true, matching loadModel.
Value pipelineReloadTextEncoder(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.reloadTextEncoder: not a loaded Pipeline");
    if (args.empty() || !ev::isString(args[0])) {
        return ev::throwTypeError("Pipeline.reloadTextEncoder(modelDir, textEncoderPath, opts?)");
    }
    std::string dir = resolveDiffusionPath(ev::toUtf8(args[0]));
    std::string tePath;
    if (args.size() >= 2 && ev::isString(args[1])) {
        std::string raw = ev::toUtf8(args[1]);
        if (!raw.empty()) tePath = resolveDiffusionPath(raw);   // "" → bundled
    }
    bool quantize = true;
    if (args.size() >= 3 && ev::isObject(args[2])) {
        quantize = propBool(args[2], "quantizeWeights", true);
    }
    try {
        w->pipeline->reload_krea2_text_encoder(dir, tePath, quantize);
        return ev::undefined();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.reloadTextEncoder failed: ") + e.what());
    }
}

// applyLora(path, scale=1.0). SD1.5 merges the deltas (undefined); Krea 2
// attaches a runtime-adapter group and returns its index. A non-numeric second
// argument leaves the scale at 1.0 rather than coercing it to 0.
Value pipelineApplyLora(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.applyLora: not a loaded Pipeline");
    if (!w->weights_loaded) return ev::throwError("Pipeline.applyLora: call loadWeights() first");
    if (args.empty() || !ev::isString(args[0])) {
        return ev::throwTypeError("Pipeline.applyLora(path, scale?): path string required");
    }

    std::string path = resolveDiffusionPath(ev::toUtf8(args[0]));
    float scale = 1.0f;
    if (args.size() > 1 && ev::isNumber(args[1])) scale = static_cast<float>(ev::toDouble(args[1]));
    try {
        auto f = brotensor::safetensors::File::open(path);
        int group = w->pipeline->apply_lora(f, scale);
        if (group >= 0) return ev::fromDouble(group);
        return ev::undefined();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.applyLora failed: ") + e.what());
    }
}

Value pipelineSetLoraScale(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.setLoraScale: not a loaded Pipeline");
    if (args.size() < 2) return ev::throwTypeError("Pipeline.setLoraScale(index, scale): required");
    int index = i32At(args, 0);
    float scale = static_cast<float>(numAt(args, 1));
    try {
        w->pipeline->set_lora_scale(index, scale);
        return ev::undefined();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.setLoraScale failed: ") + e.what());
    }
}

Value pipelineClearLoras(Value thisVal, std::span<const Value>) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.clearLoras: not a loaded Pipeline");
    try {
        w->pipeline->clear_loras();
        return ev::undefined();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.clearLoras failed: ") + e.what());
    }
}

Value pipelineNumLoras(Value thisVal, std::span<const Value>) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.numLoras: not a loaded Pipeline");
    try {
        return ev::fromDouble(w->pipeline->num_loras());
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.numLoras failed: ") + e.what());
    }
}

// addControlNet(path, cfg?) -> index. `cfg` carries the ControlNetConfig fields
// that differ across the SD1.5 ControlNet zoo; all five the old binding
// forwarded are forwarded again.
Value pipelineAddControlNet(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.addControlNet: not a loaded Pipeline");
    if (!w->weights_loaded) {
        return ev::throwError("Pipeline.addControlNet: call loadWeights() / loadModel() first");
    }
    if (args.empty() || !ev::isString(args[0])) {
        return ev::throwTypeError("Pipeline.addControlNet(path, cfg?): path string required");
    }

    std::string path = resolveDiffusionPath(ev::toUtf8(args[0]));
    try {
        auto f = brotensor::safetensors::File::open(path);
        int idx = 0;
        if (args.size() > 1 && ev::isObject(args[1])) {
            ev::Persistent c(args[1]);
            brodiffusion::controlnet::ControlNetConfig cfg;
            propInt(c.get(), "inChannels", cfg.in_channels);
            propInt(c.get(), "controlChannels", cfg.control_channels);
            propInt(c.get(), "layersPerBlock", cfg.layers_per_block);
            propInt(c.get(), "crossAttentionDim", cfg.cross_attention_dim);
            propInt(c.get(), "transformerNumHeads", cfg.transformer_num_heads);
            idx = w->pipeline->add_controlnet(f, cfg);
        } else {
            idx = w->pipeline->add_controlnet(f);
        }
        return ev::fromDouble(idx);
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.addControlNet failed: ") + e.what());
    }
}

// removeControlNet(index) — drop one registered net; later indices shift down.
Value pipelineRemoveControlNet(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.removeControlNet: not a loaded Pipeline");
    if (args.empty() || !ev::isNumber(args[0])) {
        return ev::throwTypeError("Pipeline.removeControlNet(index): integer index required");
    }
    try {
        w->pipeline->remove_controlnet(i32At(args, 0));
        return ev::undefined();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.removeControlNet failed: ") + e.what());
    }
}

Value pipelineClearControlNets(Value thisVal, std::span<const Value>) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.clearControlNets: not a loaded Pipeline");
    try {
        w->pipeline->clear_controlnets();
        return ev::undefined();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.clearControlNets failed: ") + e.what());
    }
}

// numXAttnBlocks() — traceable / steerable cross-attention blocks for the
// loaded denoiser (16 SD1.5 UNet, 57 Flux DiT, 0 without trace support).
// Only meaningful once weights are loaded, so query it live.
Value pipelineNumXAttnBlocks(Value thisVal, std::span<const Value>) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.numXAttnBlocks: not a loaded Pipeline");
    return ev::fromDouble(w->pipeline->num_xattn_blocks());
}

// sigmas() -> Float32Array — the flow-match sigma schedule of the most recent
// prime()/generate(): numSteps+1 entries with a trailing 0, empty for a
// non-flow-match scheduler.
Value pipelineSigmas(Value thisVal, std::span<const Value>) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.sigmas: not a loaded Pipeline");
    try {
        std::vector<float> s = w->pipeline->schedule_sigmas();
        return makeFloat32Array(s.data(), s.size());
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.sigmas failed: ") + e.what());
    }
}

// config() -> read-only snapshot of the resolved PipelineConfig.
Value pipelineConfig(Value thisVal, std::span<const Value>) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.config: not a loaded Pipeline");
    const brodiffusion::pipeline::PipelineConfig& cfg = w->pipeline->config();
    const bool lcm = std::holds_alternative<brodiffusion::scheduler::LCMConfig>(cfg.scheduler);
    const bool flow = std::holds_alternative<brodiffusion::scheduler::FlowMatchConfig>(cfg.scheduler);
    const bool scm = std::holds_alternative<brodiffusion::scheduler::SCMConfig>(cfg.scheduler);
    const bool dpm = std::holds_alternative<brodiffusion::scheduler::DPMSolverConfig>(cfg.scheduler);
    const char* schedName = !w->scheduler_name.empty() ? w->scheduler_name.c_str() :
        (scm ? "scm" : lcm ? "lcm" : (flow ? "flowmatch" : (dpm ? "dpm" : "ddim")));
    const char* modelClassName =
        cfg.model_class == brodiffusion::ModelClass::Flux   ? "Flux" :
        cfg.model_class == brodiffusion::ModelClass::Sana   ? "Sana" :
        cfg.model_class == brodiffusion::ModelClass::PixArt ? "PixArt" :
        cfg.model_class == brodiffusion::ModelClass::Krea2  ? "Krea2" :
        cfg.model_class == brodiffusion::ModelClass::QwenImage21 ? "QwenImage21"
                                                                 : "StableDiffusion";

    ObjectBuilder o;
    o.set("modelClass", modelClassName);
    o.set("scheduler", schedName);
    o.set("timeCondProjDim", static_cast<double>(cfg.unet.time_cond_proj_dim));
    o.set("quantizeWeights", cfg.unet.quantize_weights);
    o.set("numXAttnBlocks", static_cast<double>(w->pipeline->num_xattn_blocks()));
    o.set("weightsLoaded", w->weights_loaded);
    o.set("numControlNets", static_cast<double>(w->pipeline->num_controlnets()));
    o.set("hasControlNet", w->pipeline->has_controlnet());
    return o.build();
}

// prime(prompt, opts?) -> PipelineState. The opts are captured on the returned
// state so stepOnce()/decode() need none, and the state retains the owning
// Pipeline so its weights outlive the handle that made it.
Value pipelinePrime(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.prime: not a loaded Pipeline");
    if (!w->weights_loaded) return ev::throwError("Pipeline.prime: call loadWeights() first");
    if (args.empty() || !ev::isString(args[0])) {
        return ev::throwTypeError("Pipeline.prime(prompt, opts?): string prompt required");
    }

    ev::Persistent self(thisVal);
    std::string prompt = ev::toUtf8(args[0]);
    auto opts = parseGenerateOptions(args.size() > 1 ? args[1] : ev::undefined());
    try {
        resolveDerivedSize(*w->pipeline, opts);
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.prime failed: ") + e.what());
    }
    w->cancel_requested.store(false, std::memory_order_relaxed);
    opts.should_cancel = [w]() {
        return w->cancel_requested.load(std::memory_order_relaxed);
    };

    try {
        auto stateWrapper = std::make_unique<PipelineStateWrapper>();
        stateWrapper->opts = opts;
        stateWrapper->state = w->pipeline->prime(prompt, opts);
        Value st = g_pipelineStateClass.createInstance(std::move(stateWrapper));
        return attachPipelineToState(st, self.get());
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.prime failed: ") + e.what());
    }
}

// Pipeline.stepOnce(state, ctrl?) / Pipeline.decode(state) — the pipeline-side
// convenience forms; accepts optional trace and attnBias/logitBias control.
Value pipelineStepOnce(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.stepOnce: not a loaded Pipeline");
    if (args.empty()) return ev::throwTypeError("Pipeline.stepOnce(state, ctrl?): state required");

    auto* sw = unwrapPipelineState(args[0]);
    if (!sw) return ev::throwTypeError("Pipeline.stepOnce: expected PipelineState argument");

    bool wantTrace = false;
    Value biasVal = ev::undefined();
    if (args.size() > 1) {
        if (ev::isObject(args[1])) {
            wantTrace = propBool(args[1], "trace");
            biasVal = ev::getProperty(args[1], "attnBias");
            if (ev::isUndefined(biasVal) || ev::isNull(biasVal)) {
                biasVal = ev::getProperty(args[1], "logitBias");
            }
        } else if (ev::isBool(args[1])) {
            wantTrace = ev::toBool(args[1]);
            if (args.size() > 2) biasVal = args[2];
        }
    }

    std::vector<brotensor::Tensor> owned;
    std::vector<const brotensor::Tensor*> ptrs;
    bool haveBias = false;
    if (ev::isObject(biasVal)) {
        haveBias = true;
        const int n = w->pipeline->num_xattn_blocks();
        const uint32_t len = arrayLength(biasVal);
        if (static_cast<int>(len) != n) {
            return ev::throwRangeError(
                "Pipeline.stepOnce: attnBias length " + std::to_string(len) +
                " must equal numXAttnBlocks() " + std::to_string(n));
        }
        owned.reserve(static_cast<size_t>(n));
        ptrs.reserve(static_cast<size_t>(n));
        std::vector<bool> present(static_cast<size_t>(n), false);
        for (int i = 0; i < n; ++i) {
            ev::Persistent e(ev::getElement(biasVal, static_cast<uint32_t>(i)));
            if (!ev::isObject(e.get())) continue;
            present[static_cast<size_t>(i)] = true;
            int Lq = 0, Lk = 0;
            propInt(e.get(), "Lq", Lq);
            propInt(e.get(), "Lk", Lk);
            Value dv = ev::getProperty(e.get(), "data");
            const float* fp = nullptr;
            size_t cnt = 0;
            const bool ok = readFloat32Array(dv, fp, cnt) && fp && Lq > 0 && Lk > 0 &&
                            cnt == static_cast<size_t>(Lq) * static_cast<size_t>(Lk);
            if (!ok) {
                return ev::throwTypeError(
                    "Pipeline.stepOnce: attnBias[" + std::to_string(i) +
                    "] must be { data: Float32Array(Lq*Lk), Lq, Lk } or null");
            }
            owned.push_back(brotensor::Tensor::from_host(fp, Lq, Lk));
        }
        size_t next = 0;
        for (int i = 0; i < n; ++i) {
            ptrs.push_back(present[static_cast<size_t>(i)] ? &owned[next++] : nullptr);
        }
    }

    brodiffusion::AttentionTrace trace;
    brodiffusion::AttentionTrace* tracePtr = wantTrace ? &trace : nullptr;
    const std::vector<const brotensor::Tensor*>* biasPtr = haveBias ? &ptrs : nullptr;

    try {
        if (w->cancel_requested.load(std::memory_order_relaxed)) {
            ObjectBuilder b;
            b.set("cancelled", true);
            return b.build();
        }
        w->pipeline->step_once(sw->state, sw->opts, tracePtr, biasPtr);
        if (w->cancel_requested.load(std::memory_order_relaxed)) {
            ObjectBuilder b;
            b.set("cancelled", true);
            return b.build();
        }
        bool hasMore = sw->state.step_index < sw->state.n_steps;
        if (wantTrace) {
            ObjectBuilder res;
            res.set("hasMore", hasMore);
            ev::Persistent arr(hostArrayOf(trace.size(), [&trace](size_t i) -> Value {
                const brotensor::Tensor& t = trace[i];
                std::vector<float> host = downloadTensorFloats(t);
                ObjectBuilder entry;
                entry.set("Lq", static_cast<double>(t.rows));
                entry.set("Lk", static_cast<double>(t.cols));
                ev::Persistent d(makeFloat32Array(host.data(), host.size()));
                entry.set("data", d.get());
                return entry.build();
            }));
            res.set("trace", arr.get());
            return res.build();
        }
        return ev::fromBool(hasMore);
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.stepOnce failed: ") + e.what());
    }
}

Value pipelineDecode(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.decode: not a loaded Pipeline");
    if (args.empty()) return ev::throwTypeError("Pipeline.decode(state): state required");

    auto* sw = unwrapPipelineState(args[0]);
    if (!sw) return ev::throwTypeError("Pipeline.decode: expected PipelineState argument");

    const bool includeFp32 = args.size() > 1 && propBool(args[1], "includeFp32");
    try {
        std::vector<float> nchw = w->pipeline->decode(sw->state);
        int scale = w->pipeline->vae_scale_factor();
        int H = sw->state.H_lat * scale;
        int W = sw->state.W_lat * scale;
        return makeImageResult(nchw, H, W, includeFp32);
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.decode failed: ") + e.what());
    }
}

Value pipelineDispose(Value thisVal, std::span<const Value>) {
    auto* w = unwrapPipeline(thisVal);
    if (!w) return ev::throwTypeError("Pipeline.dispose: not a Pipeline");
    w->pipeline.reset();
    w->weights_loaded = false;
    return ev::undefined();
}

Value pipelineCancel(Value thisVal, std::span<const Value>) {
    auto* w = unwrapPipeline(thisVal);
    if (!w) return ev::throwTypeError("Pipeline.cancel: not a Pipeline");
    cancelJobsForPipeline(w);
    return ev::undefined();
}

void decoratePipeline(ObjectBuilder& proto) {
    proto.def("generate", 2, pipelineGenerate);
    proto.def("generateAsync", 2, pipelineGenerateAsync);
    proto.def("textToImage", 2, pipelineTextToImage);
    proto.def("imageToImage", 3, pipelineImageToImage);
    proto.def("inpaint", 4, pipelineInpaint);
    proto.def("loadWeights", 3, pipelineLoadWeights);
    proto.def("reloadTextEncoder", 3, pipelineReloadTextEncoder);
    proto.def("applyLora", 2, pipelineApplyLora);
    proto.def("setLoraScale", 2, pipelineSetLoraScale);
    proto.def("clearLoras", 0, pipelineClearLoras);
    proto.def("numLoras", 0, pipelineNumLoras);
    proto.def("addControlNet", 2, pipelineAddControlNet);
    proto.def("removeControlNet", 1, pipelineRemoveControlNet);
    proto.def("clearControlNets", 0, pipelineClearControlNets);
    proto.def("numXAttnBlocks", 0, pipelineNumXAttnBlocks);
    proto.def("sigmas", 0, pipelineSigmas);
    proto.def("config", 0, pipelineConfig);
    proto.def("prime", 2, pipelinePrime);
    proto.def("stepOnce", 2, pipelineStepOnce);
    proto.def("decode", 1, pipelineDecode);
    proto.def("dispose", 0, pipelineDispose);
    proto.def("cancel", 0, pipelineCancel);
    proto.def("tick", 0, [](Value, std::span<const Value>) -> Value {
        tickDiffusionAsync();
        return ev::undefined();
    });

    // Conditioning-control + identity anchor, and the per-model research
    // hook blocks (Krea 2, Qwen-Image 2.1).
    decoratePipelineControlProto(proto);
    decoratePipelineKrea2Proto(proto);
    decoratePipelineQwenImage21Proto(proto);
}

} // namespace

void ensureDiffusionClassesInstalled() {
    // Once per THREAD: a class's constructor and prototype are the
    // installing thread's (host_class.h), so a Worker realm installs its own.
    static thread_local bool installed = false;
    if (installed) return;
    installed = true;

    g_pipelineClass.install("Pipeline", 0, nullptr, decoratePipeline);
    g_pipelineStateClass.install("PipelineState", 0, nullptr, decoratePipelineStateProto);
}

Value makeDiffusionNamespace() {
    ensureDiffusionClassesInstalled();
    ObjectBuilder diff;

    diff.accessor("version", [](Value, std::span<const Value>) -> Value {
        return ev::fromUtf8(brodiffusion::version_string());
    });

    diff.def("init", 0, [](Value, std::span<const Value>) -> Value {
        try {
            brotensor::init();
        } catch (const std::exception& e) {
            return ev::throwError(std::string("bro.diffusion.init failed: ") + e.what());
        }
        return ev::undefined();
    });

    diff.def("loadModel", 2, [](Value, std::span<const Value> args) -> Value {
        if (args.empty()) {
            return ev::throwTypeError("bro.diffusion.loadModel: path is required");
        }
        if (!ev::isString(args[0])) {
            return ev::throwTypeError("bro.diffusion.loadModel: path must be a string");
        }

        std::string dir = resolveDiffusionPath(ev::toUtf8(args[0]));
        brodiffusion::pipeline::Pipeline::ModelDirOptions dirOpts;
        if (args.size() > 1 && ev::isObject(args[1])) {
            ev::Persistent o(args[1]);
            Value qv = ev::getProperty(o.get(), "quantizeWeights");
            if (!ev::isUndefined(qv)) dirOpts.quantize = ev::toBool(qv);
            std::string tePath;
            if (propStr(o.get(), "textEncoderPath", tePath) && !tePath.empty()) {
                dirOpts.text_encoder_path = resolveDiffusionPath(tePath);
            }
        }
        dirOpts.should_cancel = nullptr;

        if (!std::filesystem::exists(dir)) {
            return ev::throwError(std::string("loadModel failed: model dir not found: ") + dir);
        }

        try {
            brotensor::init();
            auto w = std::make_unique<PipelineWrapper>();
            w->pipeline = std::make_unique<brodiffusion::pipeline::Pipeline>(
                brodiffusion::pipeline::Pipeline::from_model_dir(dir, dirOpts));
            w->weights_loaded = true;
            return g_pipelineClass.createInstance(std::move(w));
        } catch (const brodiffusion::LoadCancelled&) {
            // Cancelled mid-load (teardown): a plain signal, no pipeline.
            ObjectBuilder b;
            b.set("cancelled", true);
            return b.build();
        } catch (const std::exception& e) {
            return ev::throwError(std::string("loadModel failed: ") + e.what());
        }
    });

    diff.def("createPipeline", 1, [](Value, std::span<const Value> args) -> Value {
        if (args.empty() || !ev::isObject(args[0])) {
            return ev::throwTypeError("bro.diffusion.createPipeline: config object is required");
        }
        ev::Persistent cfgArg(args[0]);

        std::string vocabPath;
        if (!propStr(cfgArg.get(), "vocabPath", vocabPath)) {
            return ev::throwTypeError("createPipeline: opts.vocabPath (string) required");
        }
        std::string mergesPath;
        if (!propStr(cfgArg.get(), "mergesPath", mergesPath)) {
            return ev::throwTypeError("createPipeline: opts.mergesPath (string) required");
        }

        std::string schedulerName = "ddim";
        propStr(cfgArg.get(), "scheduler", schedulerName);

        bool lcm = (schedulerName == "lcm");
        bool flowmatch = (schedulerName == "flowmatch" || schedulerName == "euler");
        bool scm = (schedulerName == "scm");
        bool dpm = (schedulerName == "dpm" || schedulerName == "dpmsolver");
        bool lcmDistilled = propBool(cfgArg.get(), "lcmDistilled");
        bool quantize = propBool(cfgArg.get(), "quantizeWeights");

        try {
            brotensor::init();
            auto tok = brolm::clip::Tokenizer::load(vocabPath, mergesPath);
            brodiffusion::pipeline::PipelineConfig cfg;
            if (lcm) {
                cfg.scheduler = brodiffusion::scheduler::LCMConfig{};
                if (lcmDistilled) cfg.unet.time_cond_proj_dim = 256;
            } else if (flowmatch) {
                cfg.scheduler = brodiffusion::scheduler::FlowMatchConfig{};
            } else if (scm) {
                cfg.scheduler = brodiffusion::scheduler::SCMConfig{};
            } else if (dpm) {
                cfg.scheduler = brodiffusion::scheduler::DPMSolverConfig{};
            }
            cfg.unet.quantize_weights = quantize;

            auto w = std::make_unique<PipelineWrapper>();
            w->scheduler_name = schedulerName;
            w->pipeline = std::make_unique<brodiffusion::pipeline::Pipeline>(cfg, std::move(tok));
            return g_pipelineClass.createInstance(std::move(w));
        } catch (const std::exception& e) {
            return ev::throwError(std::string("createPipeline failed: ") + e.what());
        }
    });

    diff.def("expandNoise", 2, [](Value, std::span<const Value> args) -> Value {
        if (args.empty()) {
            return ev::throwTypeError("bro.diffusion.expandNoise: src Float32Array is required");
        }

        int c = 4, h = 64, w = 64, k = 2;
        uint64_t seed = 0;
        if (args.size() > 1 && ev::isObject(args[1])) {
            ev::Persistent o(args[1]);
            propInt(o.get(), "channels", c);
            propInt(o.get(), "height", h);
            propInt(o.get(), "width", w);
            propInt(o.get(), "factor", k);
            propSeed(o.get(), "seed", seed);
        }
        if (c < 1 || h < 1 || w < 1 || k < 1) {
            return ev::throwTypeError("expandNoise: channels/height/width/factor must be >= 1");
        }

        // Read the view AFTER the option reads: a property read may allocate,
        // and a data pointer does not survive a moving collection.
        const float* src = nullptr;
        size_t count = 0;
        if (!readFloat32Array(args[0], src, count) || count == 0) {
            return ev::throwTypeError("expandNoise: src must be a non-empty Float32Array");
        }
        if (static_cast<size_t>(c) * static_cast<size_t>(h) * static_cast<size_t>(w) > count) {
            return ev::throwTypeError("expandNoise: src buffer smaller than c*h*w");
        }

        try {
            std::vector<float> expanded = brodiffusion::pipeline::expand_init_noise(src, c, h, w, k, seed);
            return makeFloat32Array(expanded.data(), expanded.size());
        } catch (const std::exception& e) {
            return ev::throwError(std::string("expandNoise failed: ") + e.what());
        }
    });

    diff.def("tick", 0, [](Value, std::span<const Value>) -> Value {
        tickDiffusionAsync();
        return ev::undefined();
    });

    diff.def("cancel", 0, [](Value, std::span<const Value> args) -> Value {
        if (!args.empty()) {
            if (auto* w = unwrapPipeline(args[0])) {
                cancelJobsForPipeline(w);
                return ev::undefined();
            }
            if (unwrapPipelineState(args[0])) {
                Value p = ev::getProperty(args[0], "__pipeline");
                if (auto* pw = unwrapPipeline(p)) {
                    cancelJobsForPipeline(pw);
                    return ev::undefined();
                }
            }
            if (ev::isObject(args[0])) {
                Value cFn = ev::getProperty(args[0], "cancel");
                if (ev::isFunction(cFn)) {
                    ev::call(cFn, args[0], {});
                    return ev::undefined();
                }
            }
        } else {
            cancelAllDiffusionJobs();
        }
        return ev::undefined();
    });

    diff.set("Pipeline", g_pipelineClass.constructor());
    diff.set("PipelineState", g_pipelineStateClass.constructor());
    ensureVaeClassesInstalled();
    diff.set("VAE", g_vaeClass.constructor());

    return diff.build();
}

} // namespace brodiffusion::api
