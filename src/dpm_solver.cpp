#include "brodiffusion/dpm_solver.h"
#include "brodiffusion/detail/device.h"
#include "brodiffusion/detail/compute.h"

#include "brotensor/ops.h"
#include "brotensor/tensor.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace brodiffusion::scheduler {

namespace bt = ::brotensor;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("scheduler::DPMSolverMultistep: " + msg);
}

}  // namespace

DPMSolverMultistep::DPMSolverMultistep(const DPMSolverConfig& cfg) : cfg_(cfg) {
    if (cfg_.num_train_timesteps <= 0) fail("num_train_timesteps must be positive");
    if (cfg_.beta_start <= 0.0f || cfg_.beta_end <= 0.0f ||
        cfg_.beta_end <= cfg_.beta_start) {
        fail("beta_start/beta_end invalid");
    }
    if (cfg_.solver_order < 1 || cfg_.solver_order > 2) {
        fail("solver_order must be 1 or 2");
    }

    // Linear betas: linspace(beta_start, beta_end, N). (PixArt uses "linear",
    // NOT SD's "scaled_linear" — getting this wrong desaturates the image.)
    const int N = cfg_.num_train_timesteps;
    alphas_cumprod_.resize(N);
    double cumprod = 1.0;
    for (int t = 0; t < N; ++t) {
        const float u = (N > 1) ? static_cast<float>(t) / static_cast<float>(N - 1)
                                : 0.0f;
        const float beta  = cfg_.beta_start + (cfg_.beta_end - cfg_.beta_start) * u;
        cumprod *= static_cast<double>(1.0f - beta);
        alphas_cumprod_[t] = static_cast<float>(cumprod);
    }
}

void DPMSolverMultistep::alpha_sigma_at(int t, float& alpha, float& sigma) const {
    const int N = cfg_.num_train_timesteps;
    const int tc = (t < 0) ? 0 : (t >= N ? N - 1 : t);
    const float acp = alphas_cumprod_[tc];
    alpha = std::sqrt(acp);
    sigma = std::sqrt(1.0f - acp);
}

void DPMSolverMultistep::set_timesteps(int num_inference_steps) {
    if (num_inference_steps <= 0) fail("num_inference_steps must be positive");
    if (num_inference_steps > cfg_.num_train_timesteps) {
        fail("num_inference_steps cannot exceed num_train_timesteps");
    }
    // "linspace" spacing: np.linspace(0, N-1, n+1).round()[::-1][:-1].
    // n+1 points p_k = k*(N-1)/n for k=0..n; reverse, drop the last (p_0=0).
    const int N = cfg_.num_train_timesteps;
    const int n = num_inference_steps;
    timesteps_.resize(n);
    const double span = static_cast<double>(N - 1);
    for (int i = 0; i < n; ++i) {
        const int k = n - i;  // i=0 -> p_n = N-1; i=n-1 -> p_1
        const double p = span * static_cast<double>(k) / static_cast<double>(n);
        int t = static_cast<int>(std::lround(p)) + cfg_.steps_offset;
        if (t < 0) t = 0;
        if (t >= N) t = N - 1;
        timesteps_[i] = t;
    }
    // Reset the multistep history for a fresh generation.
    lower_order_nums_ = 0;
    have_prev_ = false;
}

void DPMSolverMultistep::step(const bt::Tensor& model_output,
                              int step_index,
                              bt::Tensor& sample,
                              bt::Tensor& scratch) {
    if (timesteps_.empty()) fail("set_timesteps() not called");
    const int n_steps = static_cast<int>(timesteps_.size());
    if (step_index < 0 || step_index >= n_steps) fail("step_index out of range");
    if (model_output.dtype != sample.dtype) {
        fail("model_output and sample must share a dtype");
    }
    if (model_output.rows != sample.rows || model_output.cols != sample.cols) {
        fail("model_output and sample shape mismatch");
    }

    const int i = step_index;
    const int n = sample.size();
    const bt::Dtype dt = sample.dtype;
    const bt::Device dev = sample.device;
    const bt::Dtype F32 = bt::Dtype::FP32;

    // DPM-Solver++ data-prediction produces huge transient x0 values (alpha_t is
    // tiny at high noise, so x0 = (x_t - sigma_t*eps)/alpha_t blows up ~100x):
    // run the whole step in FP32 regardless of the pipeline latent dtype, then
    // write the result back. `work_` is the FP32 sample, `sc32_` the FP32 scratch.
    detail::resize_like(work_, 1, n, F32, dev);
    if (dt == F32) bt::copy_d2d(sample, 0, work_, 0, n);
    else           bt::cast(sample, work_, F32);
    detail::resize_like(mo32_, 1, n, F32, dev);
    if (dt == F32) bt::copy_d2d(model_output, 0, mo32_, 0, n);
    else           bt::cast(model_output, mo32_, F32);
    (void)scratch;  // unused; FP32 scratch sc32_ is used instead

    float alpha_s, sigma_s;
    alpha_sigma_at(timesteps_[i], alpha_s, sigma_s);
    const float lambda_s = std::log(alpha_s) - std::log(sigma_s);

    // ── convert epsilon -> data prediction x0 = (work_ - sigma_s*eps)/alpha_s
    detail::resize_like(m_cur_, 1, n, F32, dev);
    bt::copy_d2d(mo32_, 0, m_cur_, 0, n);
    bt::scale_inplace(m_cur_, -sigma_s / alpha_s);
    detail::resize_like(sc32_, 1, n, F32, dev);
    bt::copy_d2d(work_, 0, sc32_, 0, n);
    bt::scale_inplace(sc32_, 1.0f / alpha_s);
    bt::add_inplace(m_cur_, sc32_);   // m_cur_ = x0 prediction at s

    // ── target alpha/sigma (the next, lower-noise timestep). The step past the
    // last inference timestep lands on the clean image (sigma_t = 0).
    float alpha_t, sigma_t;
    if (i + 1 < n_steps) {
        alpha_sigma_at(timesteps_[i + 1], alpha_t, sigma_t);
    } else {
        alpha_t = 1.0f;
        sigma_t = 0.0f;
    }

    // ── order selection (diffusers: min over solver_order, lower_order_final
    // tail, and the warm-up counter). solver_order is 1 or 2.
    int order = std::min(cfg_.solver_order, lower_order_nums_ + 1);
    if (cfg_.lower_order_final) order = std::min(order, n_steps - i);
    if (!have_prev_) order = 1;

    if (order == 1) {
        if (sigma_t == 0.0f) {
            // Final first-order step -> the data prediction itself.
            bt::copy_d2d(m_cur_, 0, work_, 0, n);
        } else {
            const float h   = (std::log(alpha_t) - std::log(sigma_t)) - lambda_s;
            const float em1 = std::exp(-h) - 1.0f;
            const float a   = sigma_t / sigma_s;   // coef on x_s
            const float b   = -(alpha_t * em1);    // coef on x0
            bt::scale_inplace(work_, a);
            bt::copy_d2d(m_cur_, 0, sc32_, 0, n);
            bt::scale_inplace(sc32_, b);
            bt::add_inplace(work_, sc32_);
        }
    } else {
        // Second-order multistep (midpoint). sigma_t > 0 here: order 2 is never
        // selected on the final step when lower_order_final is set.
        float alpha_p, sigma_p;
        alpha_sigma_at(timesteps_[i - 1], alpha_p, sigma_p);
        const float lambda_p = std::log(alpha_p) - std::log(sigma_p);
        const float lambda_t = std::log(alpha_t) - std::log(sigma_t);
        const float h   = lambda_t - lambda_s;
        const float h0  = lambda_s - lambda_p;
        const float r0  = h0 / h;
        const float em1 = std::exp(-h) - 1.0f;
        const float base = -(alpha_t * em1);
        // x_t = a*x_s0 + base*D0 + 0.5*base*D1,  D0 = m_cur_,
        //       D1 = (m_cur_ - m_prev_)/r0
        //     = a*x_s0 + (base + 0.5*base/r0)*m_cur_ - (0.5*base/r0)*m_prev_
        const float a     = sigma_t / sigma_s;
        const float cC    = 0.5f * base / r0;
        const float cCur  = base + cC;
        const float cPrev = -cC;
        bt::scale_inplace(work_, a);
        bt::copy_d2d(m_cur_, 0, sc32_, 0, n);
        bt::scale_inplace(sc32_, cCur);
        bt::add_inplace(work_, sc32_);
        bt::copy_d2d(m_prev_, 0, sc32_, 0, n);
        bt::scale_inplace(sc32_, cPrev);
        bt::add_inplace(work_, sc32_);
    }

    // ── write the FP32 result back into the (possibly 16-bit) sample.
    if (dt == F32) bt::copy_d2d(work_, 0, sample, 0, n);
    else           bt::cast(work_, sample, dt);

    // ── shift history: m_prev_ <- current x0 (m_cur_).
    std::swap(m_prev_, m_cur_);
    have_prev_ = true;
    if (lower_order_nums_ < cfg_.solver_order) ++lower_order_nums_;
}

}  // namespace brodiffusion::scheduler
