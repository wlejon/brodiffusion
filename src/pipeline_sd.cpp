// Stable Diffusion 1.5 specifics: the CLIP prompt encode, the ControlNet
// registry, and img2img / inpaint priming. Moved verbatim out of
// pipeline.cpp — see pipeline_detail.h for the file map.

#include "brodiffusion/pipeline.h"

#include "pipeline_detail.h"

#include "brodiffusion/controlnet.h"
#include "brodiffusion/detail/compute.h"
#include "brodiffusion/detail/device.h"
#include "brodiffusion/image_io.h"

#include "brolm/clip.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace brodiffusion::pipeline {

namespace bt = ::brotensor;

using namespace detail_pipe;

int Pipeline::add_controlnet(const brotensor::safetensors::File& f) {
    return add_controlnet(f, controlnet::ControlNetConfig{});
}

int Pipeline::add_controlnet(const brotensor::safetensors::File& f,
                             const controlnet::ControlNetConfig& cfg) {
    if (model_class_ != ModelClass::StableDiffusion) {
        fail("add_controlnet: Flux not supported (SD1.5 only)");
    }
    auto net = std::make_unique<controlnet::ControlNet>(cfg);
    net->load_weights(f, "");
    controlnets_.push_back(std::move(net));
    return static_cast<int>(controlnets_.size()) - 1;
}

void Pipeline::remove_controlnet(int index) {
    if (index < 0 || index >= static_cast<int>(controlnets_.size())) {
        fail("remove_controlnet: index " + std::to_string(index) +
             " out of range (have " + std::to_string(controlnets_.size()) +
             ")");
    }
    controlnets_.erase(controlnets_.begin() + index);
    if (index < static_cast<int>(control_images_.size())) {
        control_images_.erase(control_images_.begin() + index);
    }
    if (index < static_cast<int>(control_inputs_.size())) {
        control_inputs_.erase(control_inputs_.begin() + index);
    }
    if (controlnets_.empty()) controlnet_active_ = false;
}

void Pipeline::clear_controlnets() {
    controlnets_.clear();
    control_images_.clear();
    control_inputs_.clear();
    controlnet_active_ = false;
}

void Pipeline::encode_prompt_(std::string_view prompt, bt::Tensor& out,
                              int* content_end) {
    if (!tokenizer_) fail("encode_prompt_: CLIP tokenizer not present");
    std::vector<std::int32_t> ids = tokenizer_->encode(prompt);
    if (static_cast<int>(ids.size()) != cfg_.text_encoder.max_position) {
        fail("tokenizer returned " + std::to_string(ids.size()) +
             " ids, expected " + std::to_string(cfg_.text_encoder.max_position));
    }
    if (content_end) {
        // First EOS marks the end of the content rows (ids are BOS + content +
        // EOS + EOS-padding); default to the full length if none is found.
        auto it = std::find(ids.begin(), ids.end(), brolm::clip::eos_id);
        *content_end = static_cast<int>(it - ids.begin());
    }
    text_encoder_.forward(ids.data(), out);
}

void Pipeline::prime_img2img_(const GenerateOptions& opts, PipelineState& state,
                              int n_lat, int H_lat, int W_lat, int C_lat) {
    // ── img2img: VAE-encode the init image, then noise it to t_start ──
    // Counter layout for the img2img Philox stream:
    //   counter 0..n_lat               : add_noise noise (reuses the
    //                                    txt2img initial-latent slot,
    //                                    which is unused here).
    //   counter n_lat..2*n_lat         : VAE encoder eps when
    //                                    vae_encode_sample is true.
    //   counter (1+step)*n_lat..       : LCM per-step noise (unchanged).
    // The eps slot at offset n_lat does NOT collide with the LCM
    // per-step slots because LCM starts at counter (1+0)*n_lat = n_lat
    // — but step 0's noise is only drawn on the *first* step_once(), by
    // which time we've already consumed eps. Treat them as
    // non-overlapping in time even though the counter ranges abut.
    bt::Tensor image_nchw = load_image_as_latent_input(
        opts.init_image_path, opts.width, opts.height);

    bt::Tensor x0_latent;
    if (opts.vae_encode_sample) {
        bt::Tensor eps;
        randn_compute(opts.seed,
                      static_cast<std::uint64_t>(n_lat),
                      n_lat, eps);
        vae_encoder_.encode(image_nchw, opts.height, opts.width,
                            &eps, x0_latent);
    } else {
        vae_encoder_.encode(image_nchw, opts.height, opts.width,
                            /*eps=*/nullptr, x0_latent);
    }

    // Draw the add-noise noise (raw N(0,1)) from counter 0 — same
    // Philox slot a txt2img run would have used for its initial latent.
    bt::Tensor noise;
    randn_compute(opts.seed, 0, n_lat, noise);

    const int t_start = img2img_t_start(state.n_steps, opts.strength);

    // Inpaint: stash a clone of x0 BEFORE add_noise consumes it via
    // state.latent. Cheap (one extra latent-sized tensor per generation).
    if (!opts.mask_image_path.empty()) {
        inpaint_x0_   = x0_latent.clone();
        inpaint_mask_ = brodiffusion::load_mask_as_latent(
                            opts.mask_image_path, H_lat, W_lat);

        // Broadcast the (1, H_lat*W_lat) mask across C_lat channels on the
        // host once per generation — the device-side per-step blend then
        // reduces to plain elementwise mul + add against the cached
        // broadcast tensors. Faster than rebuilding the broadcast each
        // step, and avoids needing a broadcasting op in brotensor.
        const int mask_n = H_lat * W_lat;
        std::vector<float> mask_host(static_cast<std::size_t>(mask_n));
        if (inpaint_mask_.dtype == bt::Dtype::FP16) {
            std::vector<std::uint16_t> bits(
                static_cast<std::size_t>(mask_n));
            inpaint_mask_.copy_to_host_fp16(bits.data());
            bt::sync_all();
            for (int i = 0; i < mask_n; ++i) {
                mask_host[static_cast<std::size_t>(i)] =
                    bt::fp16_bits_to_fp32(bits[static_cast<std::size_t>(i)]);
            }
        } else {
            mask_host = inpaint_mask_.to_host_vector();
        }
        std::vector<float> mask_b_host(static_cast<std::size_t>(n_lat));
        std::vector<float> one_minus_b_host(
            static_cast<std::size_t>(n_lat));
        for (int c = 0; c < C_lat; ++c) {
            for (int i = 0; i < mask_n; ++i) {
                const std::size_t k =
                    static_cast<std::size_t>(c) *
                    static_cast<std::size_t>(mask_n) +
                    static_cast<std::size_t>(i);
                const float m =
                    mask_host[static_cast<std::size_t>(i)];
                mask_b_host[k]      = m;
                one_minus_b_host[k] = 1.0f - m;
            }
        }
        inpaint_mask_b_      = detail::upload_host(mask_b_host.data(),
                                                   1, n_lat);
        inpaint_one_minus_b_ = detail::upload_host(one_minus_b_host.data(),
                                                   1, n_lat);
        inpaint_active_ = true;
    }

    std::visit([&](auto& s) {
        using S = std::decay_t<decltype(s)>;
        if constexpr (std::is_same_v<S, scheduler::DDIM> ||
                      std::is_same_v<S, scheduler::LCM> ||
                      std::is_same_v<S, scheduler::FlowMatch>) {
            s.add_noise(x0_latent, noise, t_start, state.latent, scratch_);
        } else {
            fail("prime: img2img add_noise is not supported for the "
                 "active scheduler");
        }
    }, scheduler_);
    state.step_index = t_start;
}

}  // namespace brodiffusion::pipeline
