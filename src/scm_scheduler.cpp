#include "brodiffusion/scm_scheduler.h"
#include "brodiffusion/detail/device.h"

#include "brotensor/ops.h"
#include "brotensor/tensor.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace brodiffusion::scheduler {

namespace bt = ::brotensor;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("scheduler::SCM: " + msg);
}

}  // namespace

SCM::SCM(const SCMConfig& cfg) : cfg_(cfg) {
    if (cfg_.num_train_timesteps <= 0) fail("num_train_timesteps must be positive");
    if (cfg_.sigma_data <= 0.0f) fail("sigma_data must be positive");
}

void SCM::set_timesteps(int num_inference_steps, float max_timesteps,
                        float intermediate_timesteps) {
    if (num_inference_steps <= 0) fail("num_inference_steps must be positive");
    if (num_inference_steps > cfg_.num_train_timesteps) {
        fail("num_inference_steps exceeds num_train_timesteps");
    }

    const int N = num_inference_steps;
    schedule_.assign(static_cast<std::size_t>(N) + 1, 0.0f);

    if (N == 2) {
        // sCM default two-step schedule: [max, intermediate, 0].
        schedule_[0] = max_timesteps;
        schedule_[1] = intermediate_timesteps;
        schedule_[2] = 0.0f;
    } else {
        // linspace(max_timesteps, 0, N + 1).
        for (int i = 0; i <= N; ++i) {
            schedule_[static_cast<std::size_t>(i)] =
                max_timesteps * (1.0f - static_cast<float>(i) /
                                            static_cast<float>(N));
        }
        schedule_[static_cast<std::size_t>(N)] = 0.0f;
    }

    // Run angles = the schedule without its trailing 0.
    timesteps_.assign(schedule_.begin(), schedule_.end() - 1);
}

void SCM::step(const bt::Tensor& model_output, int step_index,
               bt::Tensor& sample, const bt::Tensor& noise) const {
    if (schedule_.empty()) fail("set_timesteps() not called");
    if (step_index < 0 || step_index >= static_cast<int>(timesteps_.size())) {
        fail("step_index out of range");
    }
    if (model_output.dtype != sample.dtype || noise.dtype != sample.dtype) {
        fail("model_output / noise / sample must share a dtype");
    }
    if (model_output.rows != sample.rows || model_output.cols != sample.cols ||
        noise.rows != sample.rows || noise.cols != sample.cols) {
        fail("model_output / noise / sample shape mismatch");
    }

    const float s = schedule_[static_cast<std::size_t>(step_index)];
    const float t = schedule_[static_cast<std::size_t>(step_index) + 1];

    // pred_x0 = cos(s) * sample - sin(s) * model_output   (in place on sample).
    bt::axpby_inplace(sample, model_output, std::cos(s), -std::sin(s));

    // x_t = cos(t) * pred_x0 + sin(t) * sigma_data * noise. On the final step
    // t == 0, so this leaves sample == pred_x0 (the denoised reconstruction).
    bt::axpby_inplace(sample, noise, std::cos(t),
                      std::sin(t) * cfg_.sigma_data);
}

}  // namespace brodiffusion::scheduler
