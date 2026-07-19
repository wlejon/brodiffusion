#include "brodiffusion/terrain/sampler.h"

#include "brodiffusion/detail/compute.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace brodiffusion::terrain {
namespace {

// Pull a device tensor back to host FP32, upconverting the CUDA build's FP16.
void download_f32(const brotensor::Tensor& t, std::vector<float>& out) {
    const std::size_t n = static_cast<std::size_t>(t.rows) * t.cols;
    if (t.dtype == brotensor::Dtype::FP16) {
        std::vector<std::uint16_t> bits(n);
        t.copy_to_host_fp16(bits.data());
        brotensor::sync_all();
        out.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = brotensor::fp16_bits_to_fp32(bits[i]);
        }
        return;
    }
    brotensor::sync_all();
    out = t.to_host_vector();
}

// One UNet evaluation from a host NCHW buffer. Batch 1 throughout, so a
// channel concat is a plain buffer append and needs no interleaving.
void unet_forward_host(MPUNet& net, const std::vector<float>& x_in, int S,
                       float noise_label,
                       const std::vector<std::vector<float>>& cond,
                       std::vector<float>& out) {
    const auto& cfg = net.config();
    const std::size_t want =
        static_cast<std::size_t>(cfg.in_channels) * S * S;
    if (x_in.size() != want) {
        throw std::runtime_error("terrain sampler: model input is " +
                                 std::to_string(x_in.size()) + " values, expected " +
                                 std::to_string(want));
    }
    brotensor::Tensor x = brodiffusion::detail::upload_host(
        x_in.data(), 1, cfg.in_channels * S * S);
    brotensor::Tensor y;
    net.forward(x, /*N=*/1, S, &noise_label, cond, y);
    download_f32(y, out);
}

// Number of extra channels the stage concatenates onto its own sample.
int extra_channels(const MPUNetConfig& cfg) {
    return cfg.in_channels - cfg.out_channels;
}

}  // namespace

std::vector<float> karras_sigmas(int num_steps, float sigma_min,
                                 float sigma_max, float rho) {
    if (num_steps < 2) {
        throw std::runtime_error("terrain sampler: karras_sigmas needs >= 2 steps");
    }
    const double inv_rho = 1.0 / static_cast<double>(rho);
    const double min_inv = std::pow(static_cast<double>(sigma_min), inv_rho);
    const double max_inv = std::pow(static_cast<double>(sigma_max), inv_rho);

    std::vector<float> sigmas(static_cast<std::size_t>(num_steps) + 1);
    for (int i = 0; i < num_steps; ++i) {
        // torch.linspace(0, 1, num_steps)
        const double ramp = static_cast<double>(i) / static_cast<double>(num_steps - 1);
        sigmas[static_cast<std::size_t>(i)] = static_cast<float>(
            std::pow(max_inv + ramp * (min_inv - max_inv), static_cast<double>(rho)));
    }
    // final_sigmas_type == "zero".
    sigmas[static_cast<std::size_t>(num_steps)] = 0.0f;
    return sigmas;
}

std::vector<float> karras_timesteps(const std::vector<float>& sigmas) {
    if (sigmas.size() < 2) return {};
    std::vector<float> t(sigmas.size() - 1);
    for (std::size_t i = 0; i < t.size(); ++i) {
        t[i] = static_cast<float>(0.25 * std::log(static_cast<double>(sigmas[i])));
    }
    return t;
}

float trigflow_precondition_noise(float sigma, float sigma_data) {
    return static_cast<float>(std::atan(static_cast<double>(sigma) /
                                        static_cast<double>(sigma_data)));
}

DPMSolverMultistep::DPMSolverMultistep(int num_steps, int solver_order,
                                       float sigma_data, float sigma_min,
                                       float sigma_max, float rho)
    : num_steps_(num_steps),
      solver_order_(solver_order),
      sigma_data_(sigma_data),
      sigmas_(karras_sigmas(num_steps, sigma_min, sigma_max, rho)) {
    if (solver_order_ < 1 || solver_order_ > 2) {
        throw std::runtime_error("terrain sampler: solver_order must be 1 or 2");
    }
}

void DPMSolverMultistep::reset() {
    model_outputs_[0].clear();
    model_outputs_[1].clear();
    step_index_ = 0;
    lower_order_nums_ = 0;
}

void DPMSolverMultistep::precondition_outputs(const float* model_output,
                                              const float* sample, std::size_t n,
                                              float sigma,
                                              std::vector<float>& x0) const {
    const double sd = static_cast<double>(sigma_data_);
    const double s  = static_cast<double>(sigma);
    const double denom = s * s + sd * sd;
    const float c_skip = static_cast<float>(sd * sd / denom);
    const float c_out  = static_cast<float>(s * sd / std::sqrt(denom));
    x0.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        x0[i] = c_skip * sample[i] + c_out * model_output[i];
    }
}

void DPMSolverMultistep::first_order_(const float* sample, std::size_t n,
                                      float sigma_t, float sigma_s0,
                                      std::vector<float>& out) const {
    const std::vector<float>& m0 = model_outputs_[1];
    out.resize(n);
    // sigma_t == 0 is the trailing schedule entry. There h = log(sigma_s0) -
    // log(0) = +inf and exp(-h) = 0, so the update degenerates to x = x0. Take
    // that branch directly rather than evaluating log(0).
    if (sigma_t == 0.0f) {
        for (std::size_t i = 0; i < n; ++i) out[i] = m0[i];
        return;
    }
    const double h = std::log(static_cast<double>(sigma_s0)) -
                     std::log(static_cast<double>(sigma_t));
    const float ratio = static_cast<float>(static_cast<double>(sigma_t) /
                                           static_cast<double>(sigma_s0));
    const float coef = static_cast<float>(std::exp(-h) - 1.0);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = ratio * sample[i] - coef * m0[i];
    }
}

void DPMSolverMultistep::second_order_(const float* sample, std::size_t n,
                                       float sigma_t, float sigma_s0,
                                       float sigma_s1,
                                       std::vector<float>& out) const {
    const std::vector<float>& m0 = model_outputs_[1];   // current x0
    const std::vector<float>& m1 = model_outputs_[0];   // previous x0
    const double h   = std::log(static_cast<double>(sigma_s0)) -
                       std::log(static_cast<double>(sigma_t));
    const double h_0 = std::log(static_cast<double>(sigma_s1)) -
                       std::log(static_cast<double>(sigma_s0));
    const double r0  = h_0 / h;
    const float ratio = static_cast<float>(static_cast<double>(sigma_t) /
                                           static_cast<double>(sigma_s0));
    const float coef  = static_cast<float>(std::exp(-h) - 1.0);
    const float inv_r0 = static_cast<float>(1.0 / r0);
    out.resize(n);
    // solver_type "midpoint": x = ratio*sample - coef*D0 - 0.5*coef*D1,
    // with D0 = m0 and D1 = (1/r0) * (m0 - m1).
    for (std::size_t i = 0; i < n; ++i) {
        const float D0 = m0[i];
        const float D1 = inv_r0 * (m0[i] - m1[i]);
        out[i] = ratio * sample[i] - coef * D0 - 0.5f * coef * D1;
    }
}

void DPMSolverMultistep::step(const float* model_output, const float* sample,
                              std::size_t n, std::vector<float>& out) {
    if (step_index_ >= num_steps_) {
        throw std::runtime_error("terrain sampler: DPM step past end of schedule");
    }
    const int i = step_index_;
    const float sigma_s0 = sigmas_[static_cast<std::size_t>(i)];
    const float sigma_t  = sigmas_[static_cast<std::size_t>(i) + 1];

    // (a) raw model output -> denoised x0 prediction.
    std::vector<float> x0;
    precondition_outputs(model_output, sample, n, sigma_s0, x0);

    // (b) rotate the 2-slot history; [1] is newest.
    model_outputs_[0] = model_outputs_[1];
    model_outputs_[1] = std::move(x0);

    // (c) order selection, mirroring diffusers' branch structure. With
    // final_sigmas_type == "zero" the last step is always driven back to first
    // order; `lower_order_second` only ever matters for solver_order 3, which
    // this port does not implement, so it is computed but inert.
    const bool lower_order_final  = (i == num_steps_ - 1);
    const bool lower_order_second = (i == num_steps_ - 2) && (num_steps_ < 15);
    (void)lower_order_second;
    const bool use_first_order =
        (solver_order_ == 1) || (lower_order_nums_ < 1) || lower_order_final;

    if (use_first_order) {
        first_order_(sample, n, sigma_t, sigma_s0, out);
    } else {
        const float sigma_s1 = sigmas_[static_cast<std::size_t>(i) - 1];
        second_order_(sample, n, sigma_t, sigma_s0, sigma_s1, out);
    }

    if (lower_order_nums_ < solver_order_) ++lower_order_nums_;
    ++step_index_;
}

float trigflow_t_init(float sigma_max, float sigma_data) {
    return static_cast<float>(std::atan(static_cast<double>(sigma_max) /
                                        static_cast<double>(sigma_data)));
}

std::vector<float> trigflow_t_list(bool two_step) {
    std::vector<float> t{trigflow_t_init()};
    if (two_step) {
        t.push_back(static_cast<float>(
            std::atan(0.35 / static_cast<double>(kSigmaData))));
    }
    return t;
}

void sample_coarse(MPUNet& net, const float* noise, const float* cond_img,
                   const std::vector<std::vector<float>>& cond, int S,
                   int num_steps, std::vector<float>& out) {
    const auto& cfg = net.config();
    const int C_out  = cfg.out_channels;
    const int C_cond = extra_channels(cfg);
    if (C_cond < 0) {
        throw std::runtime_error("terrain sampler: coarse in_channels < out_channels");
    }
    if (C_cond > 0 && cond_img == nullptr) {
        throw std::runtime_error("terrain sampler: coarse stage needs a conditioning image");
    }
    const std::size_t plane = static_cast<std::size_t>(S) * S;
    const std::size_t n     = static_cast<std::size_t>(C_out) * plane;
    const std::size_t n_c   = static_cast<std::size_t>(C_cond) * plane;

    DPMSolverMultistep solver(num_steps, /*solver_order=*/2, kSigmaData);
    const std::vector<float>& sigmas = solver.sigmas();

    std::vector<float> sample(n);
    for (std::size_t i = 0; i < n; ++i) sample[i] = noise[i] * sigmas[0];

    std::vector<float> x_in(n + n_c), model_out, next;
    if (n_c) std::copy(cond_img, cond_img + n_c, x_in.begin() + static_cast<std::ptrdiff_t>(n));

    for (int step = 0; step < num_steps; ++step) {
        const float sigma = sigmas[static_cast<std::size_t>(step)];
        // precondition_inputs: x / sqrt(sigma^2 + sigma_data^2).
        const float scale = static_cast<float>(
            1.0 / std::sqrt(static_cast<double>(sigma) * sigma +
                            static_cast<double>(kSigmaData) * kSigmaData));
        for (std::size_t i = 0; i < n; ++i) x_in[i] = sample[i] * scale;

        const float cnoise = trigflow_precondition_noise(sigma, kSigmaData);
        unet_forward_host(net, x_in, S, cnoise, cond, model_out);
        solver.step(model_out.data(), sample.data(), n, next);
        sample.swap(next);
    }

    out.resize(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = sample[i] / kSigmaData;
}

void sample_trigflow(MPUNet& net, const std::vector<float>& t_list,
                     const float* noises, const float* latents,
                     const std::vector<std::vector<float>>& cond, int S,
                     std::vector<float>& out) {
    const auto& cfg = net.config();
    const int C_out   = cfg.out_channels;
    const int C_extra = extra_channels(cfg);
    if (C_extra < 0) {
        throw std::runtime_error("terrain sampler: in_channels < out_channels");
    }
    if (C_extra > 0 && latents == nullptr) {
        throw std::runtime_error("terrain sampler: stage needs --latents");
    }
    const std::size_t plane = static_cast<std::size_t>(S) * S;
    const std::size_t n     = static_cast<std::size_t>(C_out) * plane;
    const std::size_t n_e   = static_cast<std::size_t>(C_extra) * plane;

    std::vector<float> sample(n, 0.0f);
    std::vector<float> x_t(n), model_in(n + n_e), pred;
    if (n_e) std::copy(latents, latents + n_e, model_in.begin() + static_cast<std::ptrdiff_t>(n));

    for (std::size_t s = 0; s < t_list.size(); ++s) {
        const float t = t_list[s];
        const float ct = static_cast<float>(std::cos(static_cast<double>(t)));
        const float st = static_cast<float>(std::sin(static_cast<double>(t)));
        const float* noise = noises + s * n;

        // z = noise * sigma_data;  x_t = cos(t)*sample + sin(t)*z.
        for (std::size_t i = 0; i < n; ++i) {
            x_t[i] = ct * sample[i] + st * (noise[i] * kSigmaData);
            model_in[i] = x_t[i] / kSigmaData;
        }

        unet_forward_host(net, model_in, S, t, cond, pred);

        // pred is negated (upstream's sign convention), so
        // sample = cos(t)*x_t - sin(t)*sigma_data*(-model) folds to a plus.
        for (std::size_t i = 0; i < n; ++i) {
            sample[i] = ct * x_t[i] + st * kSigmaData * pred[i];
        }
    }

    out.resize(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = sample[i] / kSigmaData;
}

}  // namespace brodiffusion::terrain
