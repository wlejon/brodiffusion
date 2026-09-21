// Definitions of the internals shared by the Pipeline translation units —
// see pipeline_detail.h for what each one is for. Moved verbatim out of
// pipeline.cpp's anonymous namespace when that file was split by family.

#include "pipeline_detail.h"

#include "brodiffusion/detail/compute.h"
#include "brodiffusion/dit/flux.h"
#include "brodiffusion/dit/krea2.h"
#include "brodiffusion/dit/pixart.h"
#include "brodiffusion/dit/qwenimage21.h"
#include "brodiffusion/dit/sana.h"
#include "brodiffusion/unet.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"

#include <cstdlib>

namespace brodiffusion::pipeline::detail_pipe {

namespace bt = ::brotensor;

SchedulerVariant
make_scheduler(const std::variant<scheduler::DDIMConfig, scheduler::LCMConfig,
                                  scheduler::FlowMatchConfig,
                                  scheduler::SCMConfig,
                                  scheduler::DPMSolverConfig>& v) {
    if (std::holds_alternative<scheduler::LCMConfig>(v)) {
        return SchedulerVariant{std::in_place_type<scheduler::LCM>,
                                std::get<scheduler::LCMConfig>(v)};
    }
    if (std::holds_alternative<scheduler::FlowMatchConfig>(v)) {
        return SchedulerVariant{std::in_place_type<scheduler::FlowMatch>,
                                std::get<scheduler::FlowMatchConfig>(v)};
    }
    if (std::holds_alternative<scheduler::SCMConfig>(v)) {
        return SchedulerVariant{std::in_place_type<scheduler::SCM>,
                                std::get<scheduler::SCMConfig>(v)};
    }
    if (std::holds_alternative<scheduler::DPMSolverConfig>(v)) {
        return SchedulerVariant{std::in_place_type<scheduler::DPMSolverMultistep>,
                                std::get<scheduler::DPMSolverConfig>(v)};
    }
    return SchedulerVariant{std::in_place_type<scheduler::DDIM>,
                            std::get<scheduler::DDIMConfig>(v)};
}

float timestep_at(const SchedulerVariant& v, int i) {
    return std::visit([i](const auto& s) -> float {
        return static_cast<float>(s.timesteps()[i]);
    }, v);
}

void randn_compute(std::uint64_t key, std::uint64_t counter, int n,
                   bt::Tensor& out) {
    const bt::Dtype dt = compute_dtype();
    out = bt::Tensor::empty(1, n, dt);
    if (dt == bt::Dtype::FP16) {
        bt::Tensor fp32 = bt::Tensor::empty(1, n, bt::Dtype::FP32);
        bt::randn(key, counter, fp32);
        bt::cast(fp32, out, bt::Dtype::FP16);
    } else {
        bt::randn(key, counter, out);
    }
}

void scheduler_step(SchedulerVariant& sched, const bt::Tensor& pred,
                    int step_index, PipelineState& state, int n_lat,
                    bt::Tensor& scratch, bt::Tensor& noise_step) {
    if (std::holds_alternative<scheduler::LCM>(sched)) {
        const std::uint64_t counter =
            static_cast<std::uint64_t>(1 + step_index) *
            static_cast<std::uint64_t>(n_lat);
        randn_compute(state.rng_key, counter, n_lat, noise_step);
        std::get<scheduler::LCM>(sched).step(
            pred, step_index, state.latent, noise_step, scratch);
    } else if (std::holds_alternative<scheduler::FlowMatch>(sched)) {
        std::get<scheduler::FlowMatch>(sched).step(
            pred, step_index, state.latent, scratch);
    } else if (std::holds_alternative<scheduler::DPMSolverMultistep>(sched)) {
        // DPM-Solver multistep keeps per-generation history, so step() is
        // non-const (sched is a non-const ref here).
        std::get<scheduler::DPMSolverMultistep>(sched).step(
            pred, step_index, state.latent, scratch);
    } else {
        std::get<scheduler::DDIM>(sched).step(
            pred, step_index, state.latent, scratch);
    }
}

vae::EncoderConfig encoder_config_from_decoder(const vae::DecoderConfig& d) {
    vae::EncoderConfig e;
    e.in_channels         = d.in_channels;
    e.out_channels        = d.out_channels;
    e.block_out_channels  = d.block_out_channels;
    e.layers_per_block    = d.layers_per_block;
    e.norm_num_groups     = d.norm_num_groups;
    e.scaling_factor      = d.scaling_factor;
    e.shift_factor        = d.shift_factor;
    e.eps                 = d.eps;
    e.num_attention_heads = d.num_attention_heads;
    e.force_upcast        = d.force_upcast;
    return e;
}

std::string encoder_prefix_from_decoder(const std::string& vae_prefix) {
    const std::string tail = "decoder.";
    if (vae_prefix.size() >= tail.size() &&
        vae_prefix.compare(vae_prefix.size() - tail.size(), tail.size(),
                           tail) == 0) {
        return vae_prefix.substr(0, vae_prefix.size() - tail.size()) +
               "encoder.";
    }
    // Fallback: caller passed something unusual — just append "encoder.".
    return vae_prefix + "encoder.";
}

int img2img_t_start(int n_steps, float strength) {
    if (strength < 0.0f) strength = 0.0f;
    if (strength > 1.0f) strength = 1.0f;
    int init_t = static_cast<int>(static_cast<float>(n_steps) * strength);
    if (init_t < 1) init_t = 1;
    if (init_t > n_steps) init_t = n_steps;
    int t_start = n_steps - init_t;
    if (t_start < 0) t_start = 0;
    if (t_start >= n_steps) t_start = n_steps - 1;
    return t_start;
}

bool step_graph_disabled() {
    const char* e = std::getenv("BRODIFFUSION_DISABLE_STEP_GRAPH");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
}

std::unique_ptr<Denoiser> make_denoiser(const PipelineConfig& cfg) {
    if (cfg.model_class == ModelClass::Flux) {
        return std::make_unique<dit::FluxDenoiser>(cfg.flux);
    }
    if (cfg.model_class == ModelClass::Sana) {
        return std::make_unique<dit::SanaDenoiser>(cfg.sana);
    }
    if (cfg.model_class == ModelClass::PixArt) {
        return std::make_unique<dit::PixArtDenoiser>(cfg.pixart);
    }
    if (cfg.model_class == ModelClass::Krea2) {
        return std::make_unique<dit::Krea2Denoiser>(cfg.krea2.transformer,
                                                    cfg.krea2.patch_size);
    }
    if (cfg.model_class == ModelClass::QwenImage21) {
        return std::make_unique<dit::QwenImage21Denoiser>(
            cfg.qwenimage21.transformer);
    }
    return std::make_unique<unet::UNet>(cfg.unet);
}

bool cfg_branch_active(ModelClass model_class, const Denoiser& denoiser,
                       bool is_lcm, float guidance_scale) {
    if (!denoiser.uses_cfg() || is_lcm) return false;
    if (model_class == ModelClass::QwenImage21) return guidance_scale > 1.0f;
    return guidance_scale != 1.0f;
}

}  // namespace brodiffusion::pipeline::detail_pipe
