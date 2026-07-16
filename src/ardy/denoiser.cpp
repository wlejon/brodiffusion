#include "brodiffusion/ardy/denoiser.h"

#include "brodiffusion/detail/compute.h"
#include "brotensor/safetensors.h"
#include "brotensor/ops.h"
#include "brotensor/runtime.h"

#include <cmath>
#include <stdexcept>

namespace brodiffusion::ardy {

namespace bt = ::brotensor;
namespace st = ::brotensor::safetensors;

namespace {

[[noreturn]] void fail(const std::string& m) {
    throw std::runtime_error("ardy::ArdyDenoiser: " + m);
}

const st::TensorView& need(const st::File& f, const std::string& key) {
    if (const auto* v = f.find(key)) return *v;
    fail("missing tensor '" + key + "'");
}

void load_linear(const st::File& f, const std::string& key, int out, int in,
                 bt::Tensor& W, bt::Tensor& b) {
    st::upload_compute_checked(need(f, key + ".weight"), out, in, W, key);
    st::upload_compute_checked(need(f, key + ".bias"), out, 1, b, key);
}

// Read a device tensor to host FP32, up-converting from FP16 if needed. Used for
// the global->local root conversion, which is host-side per-frame geometry.
std::vector<float> to_host_f32(const bt::Tensor& t) {
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

ArdyDenoiser::ArdyDenoiser(const Config& cfg)
    : cfg_(cfg),
      root_model_([&] {
          ArdyDenoiserBackbone::Config c;
          c.latent_dim = cfg.latent_dim;
          c.output_dim = cfg.nframe_root_dim;       // 20
          return c;
      }()),
      body_model_([&] {
          ArdyDenoiserBackbone::Config c;
          c.latent_dim = cfg.latent_dim;
          c.output_dim = cfg.latent_embedding_dim;  // 128
          return c;
      }()) {}

ArdyDenoiser::~ArdyDenoiser() = default;

void ArdyDenoiser::load_weights(const st::File& f) {
    root_model_.load_weights(f, "denoiser.backbone.root_model.");
    body_model_.load_weights(f, "denoiser.backbone.body_model.");

    const int D   = cfg_.latent_dim;
    const int fpt = cfg_.num_frames_per_token;
    const int hyb = hybrid_dim();  // 148
    // constraints projections: hybrid_token + (motion_rep_dim + body_dim)*fpt.
    const int gr_ext = hyb + (cfg_.motion_rep_dim + cfg_.body_dim) * fpt;  // 3440
    const int nframe_local_root = cfg_.local_root_dim * fpt;               // 16
    const int lr_hyb = nframe_local_root + cfg_.latent_embedding_dim;      // 144
    const int lr_ext = lr_hyb + (cfg_.motion_rep_dim + cfg_.body_dim) * fpt;  // 3436

    load_linear(f, "denoiser.global_root_hybrid_constraints_proj", D, gr_ext,
                global_root_hybrid_constraints_proj_.W,
                global_root_hybrid_constraints_proj_.b);
    load_linear(f, "denoiser.local_root_hybrid_constraints_proj", D, lr_ext,
                local_root_hybrid_constraints_proj_.W,
                local_root_hybrid_constraints_proj_.b);
}

void ArdyDenoiser::set_motion_stats(const float* mean, const float* std, int n,
                                    float eps) {
    // Motion stats bundle [global_root 5, local_root 4, body 409] = 418 entries
    // (NOT the 414-dim explicit feature vector — the local-root stats are extra).
    const int stats_dim = cfg_.motion_root_dim + cfg_.local_root_dim + cfg_.body_dim;
    if (n != stats_dim) fail("motion stats length != global+local+body dim");
    const int gr = cfg_.motion_root_dim;   // 5, indices [0:5]
    const int lr = cfg_.local_root_dim;    // 4, indices [5:9]
    gr_mean_.assign(mean, mean + gr);
    gr_stdeps_.resize(gr);
    for (int i = 0; i < gr; ++i) gr_stdeps_[i] = std::sqrt(std[i] * std[i] + eps);
    lr_mean_.assign(mean + gr, mean + gr + lr);
    lr_stdeps_.resize(lr);
    for (int i = 0; i < lr; ++i)
        lr_stdeps_[i] = std::sqrt(std[gr + i] * std[gr + i] + eps);
}

// Angle difference dtheta/dt via the robust cos/sin formulation, matching
// ardy motion_rep/tools.py diff_angles: fps * atan2(sin(b)cos(a)-cos(b)sin(a),
// cos(b)cos(a)+sin(b)sin(a)) for consecutive angles a=theta[t], b=theta[t+1].
void ArdyDenoiser::global_root_to_local_root(const float* groot_norm,
                                             int T, float* lroot_norm) const {
    if (gr_mean_.empty()) fail("global_root_to_local_root: motion stats not set");
    const int GR = cfg_.motion_root_dim;   // 5
    const float fps = cfg_.fps;

    // unnormalize global root; extract root xyz + heading angle per frame.
    std::vector<float> px(T), py(T), pz(T), ang(T);
    for (int t = 0; t < T; ++t) {
        const float* g = groot_norm + static_cast<size_t>(t) * GR;
        const float x = g[0] * gr_stdeps_[0] + gr_mean_[0];
        const float y = g[1] * gr_stdeps_[1] + gr_mean_[1];
        const float z = g[2] * gr_stdeps_[2] + gr_mean_[2];
        const float c = g[3] * gr_stdeps_[3] + gr_mean_[3];  // cos(heading)
        const float s = g[4] * gr_stdeps_[4] + gr_mean_[4];  // sin(heading)
        px[t] = x; py[t] = y; pz[t] = z;
        ang[t] = std::atan2(s, c);
    }

    // rot-vel (heading rate) and planar velocity, both fps-scaled forward diffs
    // with the last valid frame duplicated (lengths == T here, no padding).
    std::vector<float> rot_vel(T, 0.0f), vel_x(T, 0.0f), vel_z(T, 0.0f);
    for (int t = 0; t + 1 < T; ++t) {
        const float ca = std::cos(ang[t]),   sa = std::sin(ang[t]);
        const float cb = std::cos(ang[t + 1]), sb = std::sin(ang[t + 1]);
        rot_vel[t] = fps * std::atan2(sb * ca - cb * sa, cb * ca + sb * sa);
        vel_x[t]   = fps * (px[t + 1] - px[t]);
        vel_z[t]   = fps * (pz[t + 1] - pz[t]);
    }
    if (T >= 2) {
        rot_vel[T - 1] = rot_vel[T - 2];
        vel_x[T - 1]   = vel_x[T - 2];
        vel_z[T - 1]   = vel_z[T - 2];
    }

    // pack [rot_vel, vel_x, vel_z, global_root_y] and normalize with local stats.
    for (int t = 0; t < T; ++t) {
        float v[4] = {rot_vel[t], vel_x[t], vel_z[t], py[t]};
        float* o = lroot_norm + static_cast<size_t>(t) * 4;
        for (int i = 0; i < 4; ++i) o[i] = (v[i] - lr_mean_[i]) / lr_stdeps_[i];
    }
}

void ArdyDenoiser::forward(const float* hybrid, const float* text_feat,
                           int timestep, float first_heading_angle, int T_tok,
                           bt::Tensor& out) {
    const int fpt = cfg_.num_frames_per_token;
    const int NR  = cfg_.nframe_root_dim;         // 20
    const int LB  = cfg_.latent_embedding_dim;    // 128
    const int hyb = hybrid_dim();                 // 148
    const int T_frames = T_tok * fpt;             // 52
    const int gr_ext = hyb + (cfg_.motion_rep_dim + cfg_.body_dim) * fpt;  // 3440
    const int nframe_local_root = cfg_.local_root_dim * fpt;               // 16
    const int lr_hyb = nframe_local_root + LB;                             // 144
    const int lr_ext = lr_hyb + (cfg_.motion_rep_dim + cfg_.body_dim) * fpt;  // 3436

    // Generation-window token index: origin at frame 0 -> arange(T_tok).
    std::vector<int> tok_idx(T_tok);
    for (int i = 0; i < T_tok; ++i) tok_idx[i] = i;

    // ── stage 1: root ──
    // x_infilled_extended = [latent_body(128), global_root(20), zeros(observed
    // body + motion mask)]; zero-initialized so the constraint slots stay 0.
    std::vector<float> gr_in(static_cast<size_t>(T_tok) * gr_ext, 0.0f);
    for (int t = 0; t < T_tok; ++t) {
        const float* xt = hybrid + static_cast<size_t>(t) * hyb;  // [root20, body128]
        float* row = gr_in.data() + static_cast<size_t>(t) * gr_ext;
        for (int i = 0; i < LB; ++i) row[i] = xt[NR + i];         // latent_body
        for (int i = 0; i < NR; ++i) row[LB + i] = xt[i];         // global_root
    }
    bt::Tensor gr_inT = detail::upload_host(gr_in.data(), T_tok, gr_ext);
    bt::Tensor root_stage_input;
    detail::linear_batched(global_root_hybrid_constraints_proj_.W,
                           &global_root_hybrid_constraints_proj_.b,
                           gr_inT, root_stage_input);

    bt::Tensor root_pred_T;  // (T_tok, 20)
    root_model_.forward(root_stage_input, text_feat, timestep,
                        first_heading_angle, tok_idx.data(), T_tok,
                        /*key_mask=*/nullptr, root_pred_T);
    bt::sync_all();
    std::vector<float> root_pred = to_host_f32(root_pred_T);

    // global_root_motion_pred as (T_frames, 5) [row-major identical to (T_tok,20)].
    // convert to local root, then reshape (T_frames,4) -> (T_tok, 16).
    std::vector<float> local_root(static_cast<size_t>(T_frames) * cfg_.local_root_dim);
    global_root_to_local_root(root_pred.data(), T_frames, local_root.data());

    // ── stage 2: body ──
    // x_new = [local_root(16), latent_body(128)]; x_new_extended pads zeros.
    std::vector<float> lr_in(static_cast<size_t>(T_tok) * lr_ext, 0.0f);
    for (int t = 0; t < T_tok; ++t) {
        const float* xt = hybrid + static_cast<size_t>(t) * hyb;
        const float* lr = local_root.data() + static_cast<size_t>(t) * nframe_local_root;
        float* row = lr_in.data() + static_cast<size_t>(t) * lr_ext;
        for (int i = 0; i < nframe_local_root; ++i) row[i] = lr[i];   // local root
        for (int i = 0; i < LB; ++i) row[nframe_local_root + i] = xt[NR + i];  // body
    }
    bt::Tensor lr_inT = detail::upload_host(lr_in.data(), T_tok, lr_ext);
    bt::Tensor body_stage_input;
    detail::linear_batched(local_root_hybrid_constraints_proj_.W,
                           &local_root_hybrid_constraints_proj_.b,
                           lr_inT, body_stage_input);

    bt::Tensor body_pred_T;  // (T_tok, 128)
    body_model_.forward(body_stage_input, text_feat, timestep,
                        first_heading_angle, tok_idx.data(), T_tok,
                        /*key_mask=*/nullptr, body_pred_T);
    bt::sync_all();
    std::vector<float> body_pred = to_host_f32(body_pred_T);

    // ── output: [global_root_pred(20), body_pred(128)] per token ──
    std::vector<float> outh(static_cast<size_t>(T_tok) * hyb);
    for (int t = 0; t < T_tok; ++t) {
        float* o = outh.data() + static_cast<size_t>(t) * hyb;
        for (int i = 0; i < NR; ++i) o[i] = root_pred[static_cast<size_t>(t) * NR + i];
        for (int i = 0; i < LB; ++i) o[NR + i] = body_pred[static_cast<size_t>(t) * LB + i];
    }
    out = detail::upload_host(outh.data(), T_tok, hyb);
}

}  // namespace brodiffusion::ardy
