#include "brodiffusion/ardy/sampler.h"

#include "brotensor/runtime.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

namespace brodiffusion::ardy {

namespace bt = ::brotensor;

namespace {

constexpr double kPi = 3.14159265358979323846;

// g152: num_text_tokens(1) * llm_dim(4096). The denoiser always builds its
// backbones with the default llm shape, so the uncond text buffer is this size.
constexpr int kTextFeatDim = 4096;

// Read a denoiser output tensor to host FP32, up-converting from FP16 on GPU
// builds (the CFG combine + DDIM step are host-side).
std::vector<float> read_host(const bt::Tensor& t) {
    const std::size_t n = static_cast<std::size_t>(t.rows) * t.cols;
    if (t.dtype == bt::Dtype::FP16) {
        std::vector<std::uint16_t> bits(n);
        t.copy_to_host_fp16(bits.data());
        bt::sync_all();
        std::vector<float> v(n);
        for (std::size_t i = 0; i < n; ++i) v[i] = bt::fp16_bits_to_fp32(bits[i]);
        return v;
    }
    return t.to_host_vector();
}

}  // namespace

ArdyDiffusion::ArdyDiffusion(int num_base_steps) : num_base_steps_(num_base_steps) {
    // get_beta_schedule: cosine alpha_bar, betas capped at max_beta 0.999.
    auto alpha_bar = [](double u) {
        const double c = std::cos((u + 0.008) / 1.008 * kPi / 2.0);
        return c * c;
    };
    const int N = num_base_steps_;
    alphas_cumprod_base_.resize(N);
    double cum = 1.0;
    for (int i = 0; i < N; ++i) {
        const double t1 = static_cast<double>(i) / N;
        const double t2 = static_cast<double>(i + 1) / N;
        const double beta = std::min(1.0 - alpha_bar(t2) / alpha_bar(t1), 0.999);
        cum *= (1.0 - beta);
        alphas_cumprod_base_[i] = static_cast<float>(cum);
    }
}

ArdyDiffusion::Schedule ArdyDiffusion::make_schedule(int num_denoising_steps) const {
    const int N = num_base_steps_;
    Schedule s;
    s.num_denoising_steps = num_denoising_steps;

    // space_timesteps: use_timesteps = clamp(round(arange(N) * frac_stride), max=N-1).
    // frac_stride depends on the requested step count; the array is always length N.
    // std::nearbyint under the default FE_TONEAREST matches torch.round (half-to-even).
    const double frac_stride =
        static_cast<double>(N - 1) / std::max(1, num_denoising_steps - 1);
    s.base_timestep.resize(N);
    for (int k = 0; k < N; ++k) {
        int v = static_cast<int>(std::nearbyint(k * frac_stride));
        if (v > N - 1) v = N - 1;
        s.base_timestep[k] = v;
    }

    // calc_diffusion_vars: gather alphas_cumprod, derive betas, re-cumprod, clamp.
    std::vector<double> ac(N);
    for (int k = 0; k < N; ++k) ac[k] = alphas_cumprod_base_[s.base_timestep[k]];

    std::vector<double> ac2(N);  // recomputed (clamped) alphas_cumprod
    double cum = 1.0;
    for (int k = 0; k < N; ++k) {
        const double last = (k == 0) ? 1.0 : ac[k - 1];
        const double beta = 1.0 - ac[k] / last;
        const double alpha = 1.0 - beta;
        cum *= alpha;                       // running cumprod uses the unclamped value
        ac2[k] = std::max(cum, 1e-9);       // clamp is elementwise, does not feed back
    }

    s.alphas_cumprod_prev.resize(N);
    s.sqrt_recip_alphas_cumprod.resize(N);
    s.sqrt_recipm1_alphas_cumprod.resize(N);
    for (int k = 0; k < N; ++k) {
        const double acp = (k == 0) ? 1.0 : ac2[k - 1];
        s.alphas_cumprod_prev[k] = static_cast<float>(acp);
        s.sqrt_recip_alphas_cumprod[k] = static_cast<float>(1.0 / std::sqrt(ac2[k]));
        s.sqrt_recipm1_alphas_cumprod[k] =
            static_cast<float>(std::sqrt((1.0 - ac2[k]) / ac2[k]));
    }
    return s;
}

void ArdyDiffusion::ddim_step(const Schedule& s, int t, const float* x_t,
                              const float* x0, int n, float* out) {
    const float srac = s.sqrt_recip_alphas_cumprod[t];
    const float srm1 = s.sqrt_recipm1_alphas_cumprod[t];
    const float acp = s.alphas_cumprod_prev[t];
    const float sqrt_acp = std::sqrt(acp);
    const float sqrt_1macp = std::sqrt(1.0f - acp);
    for (int i = 0; i < n; ++i) {
        const float eps = (srac * x_t[i] - x0[i]) / srm1;
        out[i] = x0[i] * sqrt_acp + sqrt_1macp * eps;
    }
}

ArdyWindowSampler::ArdyWindowSampler(ArdyDenoiser& denoiser, int num_base_steps)
    : diffusion_(num_base_steps), denoiser_(denoiser) {}

void ArdyWindowSampler::sample(const float* x_init, const float* text_feat,
                               int T_tok, float first_heading_angle,
                               int num_denoising_steps, float cfg_weight,
                               std::vector<float>& out) {
    const int hyb = denoiser_.hybrid_dim();  // 148
    const int n = T_tok * hyb;
    const ArdyDiffusion::Schedule sch = diffusion_.make_schedule(num_denoising_steps);

    std::vector<float> x(x_init, x_init + n);
    const std::vector<float> zero_text(kTextFeatDim, 0.0f);
    std::vector<float> x0(n), x_next(n);

    // Reverse walk over subsampled step indices [num_denoising_steps-1 .. 0].
    for (int i = num_denoising_steps - 1; i >= 0; --i) {
        const int t = i;
        const int t_map = sch.base_timestep[i];  // original timestep for the denoiser

        // Two forwards per step: real text and zero text. For a constraint-free
        // window this is the separated-CFG collapse (constraint pass == uncond).
        bt::Tensor out_text_T, out_uncond_T;
        denoiser_.forward(x.data(), text_feat, t_map, first_heading_angle, T_tok,
                          out_text_T);
        denoiser_.forward(x.data(), zero_text.data(), t_map, first_heading_angle,
                          T_tok, out_uncond_T);
        bt::sync_all();
        const std::vector<float> ot = read_host(out_text_T);
        const std::vector<float> ou = read_host(out_uncond_T);

        for (int k = 0; k < n; ++k) x0[k] = ou[k] + cfg_weight * (ot[k] - ou[k]);

        ArdyDiffusion::ddim_step(sch, t, x.data(), x0.data(), n, x_next.data());
        x.swap(x_next);
    }

    out = std::move(x);
}

}  // namespace brodiffusion::ardy
