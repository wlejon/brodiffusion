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

void ArdyWindowSampler::sample_ar_window(const float* history,
                                         int num_history_tokens,
                                         const float* gen_noise, int num_gen_tokens,
                                         const float* text_feat,
                                         float first_heading_angle,
                                         int num_denoising_steps, float cfg_weight,
                                         std::vector<float>& out_gen) {
    const int hyb = denoiser_.hybrid_dim();  // 148
    const int H = num_history_tokens;
    const int G = num_gen_tokens;
    const int T_tok = H + G;
    const int n = T_tok * hyb;         // full window element count
    const int gn = G * hyb;            // generation-block element count
    const ArdyDiffusion::Schedule sch = diffusion_.make_schedule(num_denoising_steps);

    // x = [history | generation noise]; only the generation block is stepped.
    std::vector<float> x(static_cast<size_t>(n));
    if (H > 0) std::copy(history, history + static_cast<size_t>(H) * hyb, x.begin());
    std::copy(gen_noise, gen_noise + gn, x.begin() + static_cast<size_t>(H) * hyb);

    const std::vector<float> zero_text(kTextFeatDim, 0.0f);
    std::vector<float> x0_gen(gn), x_next_gen(gn);
    float* gen_ptr = x.data() + static_cast<size_t>(H) * hyb;

    for (int i = num_denoising_steps - 1; i >= 0; --i) {
        const int t = i;
        const int t_map = sch.base_timestep[i];

        bt::Tensor out_text_T, out_uncond_T;
        denoiser_.forward(x.data(), text_feat, t_map, first_heading_angle, T_tok,
                          out_text_T, H);
        denoiser_.forward(x.data(), zero_text.data(), t_map, first_heading_angle,
                          T_tok, out_uncond_T, H);
        bt::sync_all();
        const std::vector<float> ot = read_host(out_text_T);
        const std::vector<float> ou = read_host(out_uncond_T);

        // CFG combine + DDIM on the generation tokens only (history is held).
        const int base = H * hyb;
        for (int k = 0; k < gn; ++k)
            x0_gen[k] = ou[base + k] + cfg_weight * (ot[base + k] - ou[base + k]);
        ArdyDiffusion::ddim_step(sch, t, gen_ptr, x0_gen.data(), gn, x_next_gen.data());
        std::copy(x_next_gen.begin(), x_next_gen.end(), gen_ptr);
    }

    out_gen.assign(gen_ptr, gen_ptr + gn);
}

ArdyMotionGenerator::ArdyMotionGenerator(ArdyDenoiser& denoiser,
                                         FsqMotionDecoder& fsq,
                                         int gen_horizon_len, int num_base_steps)
    : sampler_(denoiser, num_base_steps),
      denoiser_(denoiser),
      fsq_(fsq),
      gen_horizon_len_(gen_horizon_len) {}

int ArdyMotionGenerator::num_windows(int num_frames) const {
    if (num_frames <= 0) return 0;
    return (num_frames + gen_horizon_len_ - 1) / gen_horizon_len_;  // ceil
}

int ArdyMotionGenerator::num_tokens(int num_frames) const {
    const int fpt = denoiser_.config().num_frames_per_token;
    return num_windows(num_frames) * (gen_horizon_len_ / fpt);
}

void ArdyMotionGenerator::generate_hybrid(const float* text_feat, int num_frames,
                                          float first_heading_angle,
                                          int num_denoising_steps, float cfg_weight,
                                          const float* gen_noise,
                                          std::vector<float>& out_hybrid,
                                          int& out_T_tok) {
    const int fpt = denoiser_.config().num_frames_per_token;   // 4
    const int NR  = denoiser_.config().nframe_root_dim;        // 20
    const int LB  = denoiser_.config().latent_embedding_dim;   // 128
    const int hyb = denoiser_.hybrid_dim();                    // 148
    const int G   = gen_horizon_len_ / fpt;                    // 13 tokens / window
    const int W   = num_windows(num_frames);

    std::vector<float> history;  // grows by one window (G tokens) each step
    history.reserve(static_cast<size_t>(W) * G * hyb);
    float global_transl[3] = {0.0f, 0.0f, 0.0f};

    // Scratch for the per-window recenter/requantize on the growing history.
    std::vector<float> groot, latent;

    for (int step = 0; step < W; ++step) {
        const int H = step * G;  // all prior tokens are history (no crop)
        const float* noise = gen_noise + static_cast<size_t>(step) * G * hyb;

        std::vector<float> gen;
        sampler_.sample_ar_window(H > 0 ? history.data() : nullptr, H, noise, G,
                                  text_feat, first_heading_angle,
                                  num_denoising_steps, cfg_weight, gen);
        history.insert(history.end(), gen.begin(), gen.end());

        // Recenter the whole history around the last generated frame, requantize
        // the body latents, both in place (requantize=True in _recenter_history).
        const int total_tok = H + G;
        const int F = total_tok * fpt;
        const int center_frame = step * gen_horizon_len_ + gen_horizon_len_ - 1;

        // extract global root (F,5) == per-token [0:20], recenter, write back.
        groot.resize(static_cast<size_t>(total_tok) * NR);
        for (int tk = 0; tk < total_tok; ++tk)
            std::copy(history.data() + static_cast<size_t>(tk) * hyb,
                      history.data() + static_cast<size_t>(tk) * hyb + NR,
                      groot.data() + static_cast<size_t>(tk) * NR);
        float center_pos[3];
        denoiser_.recenter_global_root(groot.data(), F, center_frame, center_pos);
        for (int tk = 0; tk < total_tok; ++tk)
            std::copy(groot.data() + static_cast<size_t>(tk) * NR,
                      groot.data() + static_cast<size_t>(tk) * NR + NR,
                      history.data() + static_cast<size_t>(tk) * hyb);

        // extract body latent (total_tok,128) == per-token [20:148], requantize.
        latent.resize(static_cast<size_t>(total_tok) * LB);
        for (int tk = 0; tk < total_tok; ++tk)
            std::copy(history.data() + static_cast<size_t>(tk) * hyb + NR,
                      history.data() + static_cast<size_t>(tk) * hyb + hyb,
                      latent.data() + static_cast<size_t>(tk) * LB);
        fsq_.requantize(latent.data(), total_tok);
        for (int tk = 0; tk < total_tok; ++tk)
            std::copy(latent.data() + static_cast<size_t>(tk) * LB,
                      latent.data() + static_cast<size_t>(tk) * LB + LB,
                      history.data() + static_cast<size_t>(tk) * hyb + NR);

        global_transl[0] += center_pos[0];
        global_transl[2] += center_pos[2];
    }

    // Apply the accumulated global translation back onto the root (world frame).
    const int total_tok = W * G;
    const int F = total_tok * fpt;
    groot.resize(static_cast<size_t>(total_tok) * NR);
    for (int tk = 0; tk < total_tok; ++tk)
        std::copy(history.data() + static_cast<size_t>(tk) * hyb,
                  history.data() + static_cast<size_t>(tk) * hyb + NR,
                  groot.data() + static_cast<size_t>(tk) * NR);
    denoiser_.translate_global_root(groot.data(), F, global_transl);
    for (int tk = 0; tk < total_tok; ++tk)
        std::copy(groot.data() + static_cast<size_t>(tk) * NR,
                  groot.data() + static_cast<size_t>(tk) * NR + NR,
                  history.data() + static_cast<size_t>(tk) * hyb);

    out_hybrid = std::move(history);
    out_T_tok = total_tok;
}

void ArdyMotionGenerator::detokenize_to_motion(const float* hybrid, int T_tok,
                                               std::vector<float>& out_motion) {
    const int fpt = denoiser_.config().num_frames_per_token;   // 4
    const int NR  = denoiser_.config().nframe_root_dim;        // 20
    const int LB  = denoiser_.config().latent_embedding_dim;   // 128
    const int hyb = denoiser_.hybrid_dim();                    // 148
    const int GR  = denoiser_.config().motion_root_dim;        // 5
    const int LR  = denoiser_.config().local_root_dim;         // 4
    const int MRD = denoiser_.config().motion_rep_dim;         // 414
    const int BD  = denoiser_.config().body_dim;               // 409
    const int F   = T_tok * fpt;
    const int pose_dim = fsq_.config().output_dim;             // 413 = local_root4 + body409
    const int fsq_lr   = fsq_.config().local_root_dim;         // 4

    // split: per token [global_root 20 == (fpt,5)] ++ [body latent 128].
    std::vector<float> groot(static_cast<size_t>(F) * GR);      // (F,5) flat
    std::vector<float> latent(static_cast<size_t>(T_tok) * LB);
    for (int t = 0; t < T_tok; ++t) {
        const float* xt = hybrid + static_cast<size_t>(t) * hyb;
        for (int i = 0; i < NR; ++i) groot[static_cast<size_t>(t) * NR + i] = xt[i];
        for (int i = 0; i < LB; ++i) latent[static_cast<size_t>(t) * LB + i] = xt[NR + i];
    }
    // external-root condition: local root (F,4) == (T_tok, fpt*4) row-major.
    std::vector<float> lroot(static_cast<size_t>(F) * LR);
    denoiser_.global_root_to_local_root(groot.data(), F, lroot.data());

    bt::Tensor decoded_T;
    fsq_.detokenize(latent.data(), lroot.data(), T_tok, decoded_T);
    bt::sync_all();
    const std::vector<float> decoded = read_host(decoded_T);   // (F, 413)

    // explicit motion (F,414) = [global_root 5, decoded body 409].
    out_motion.resize(static_cast<size_t>(F) * MRD);
    for (int f = 0; f < F; ++f) {
        float* o = out_motion.data() + static_cast<size_t>(f) * MRD;
        const float* g = groot.data() + static_cast<size_t>(f) * GR;
        const float* d = decoded.data() + static_cast<size_t>(f) * pose_dim;
        for (int i = 0; i < GR; ++i) o[i] = g[i];
        for (int i = 0; i < BD; ++i) o[GR + i] = d[fsq_lr + i];
    }
}

}  // namespace brodiffusion::ardy
