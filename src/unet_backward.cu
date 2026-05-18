// dX-only backward through the FROZEN teacher UNet's mid_block + up_blocks +
// conv_out path. See include/brodiffusion/unet_backward.h.
//
// Implementation notes
// ────────────────────
// • Re-runs the teacher forward inside this function using the UNFUSED
//   brotensor primitives (rather than the fused kernels used at inference)
//   so every backward op has a forward input it can recompute from. The
//   numerical result of this replayed forward is bit-equivalent (modulo
//   accumulation order) to the inference forward.
// • Every weight grad is dumped into thread_local "sink" buffers (teacher
//   is frozen — values discarded). Each sink is zero'd before use to
//   satisfy the accumulated/caller-zeros contract of every brotensor
//   backward op.
// • Transformer LayerNorm: brotensor's layernorm_backward_gpu is single-row;
//   we provide an inline batched FP16 dX-only kernel below.
// • GEGLU forward: brotensor has geglu_exact_backward_gpu but no public
//   geglu_exact_forward — we provide an inline kernel below.
// • For residual adds (y = x + z): dx = dz = dY. We split by cloning dY
//   into the residual path before recursing into the sub-branch.
//
// Cost: roughly 2× a teacher up-path forward; activation cache peaks around
// 50–80 MB (dominated by (L=64×64, C=320) transformer sequence caches and
// the conv_in (1, 320, 64, 64) tensors near conv_out).

#include "brodiffusion/unet_backward.h"
#include "brodiffusion/unet.h"

#include "brotensor/ops.h"
#include "brotensor/tensor.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdio>
#include <stdexcept>
#include <utility>
#include <vector>

namespace brodiffusion::unet {

namespace bt = ::brotensor;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("unet_up_path_backward: " + msg);
}

// ─── Inline batched LayerNorm backward (FP16, dX only) ───────────────────────
__global__ void batched_ln_bwd_dX_k(const __half* __restrict__ X,
                                     const __half* __restrict__ gamma,
                                     const __half* __restrict__ dY,
                                     __half* __restrict__ dX,
                                     int D, float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int bs  = blockDim.x;

    extern __shared__ float smem[];
    float* s1 = smem;
    float* s2 = smem + bs;

    const __half* xr  = X  + row * D;
    const __half* dyr = dY + row * D;
    __half*       dxr = dX + row * D;

    float sx = 0.f, sxx = 0.f;
    for (int i = tid; i < D; i += bs) {
        float v = __half2float(xr[i]);
        sx  += v;
        sxx += v * v;
    }
    s1[tid] = sx;  s2[tid] = sxx;
    __syncthreads();
    for (int off = bs / 2; off > 0; off >>= 1) {
        if (tid < off) { s1[tid] += s1[tid + off]; s2[tid] += s2[tid + off]; }
        __syncthreads();
    }
    float mean = s1[0] / (float)D;
    float var  = s2[0] / (float)D - mean * mean;
    float rstd = rsqrtf(var + eps);
    __syncthreads();

    float a = 0.f, b = 0.f;
    for (int i = tid; i < D; i += bs) {
        float xhat  = (__half2float(xr[i]) - mean) * rstd;
        float gv    = __half2float(gamma[i]);
        float dxhat = __half2float(dyr[i]) * gv;
        a += dxhat;
        b += dxhat * xhat;
    }
    s1[tid] = a;  s2[tid] = b;
    __syncthreads();
    for (int off = bs / 2; off > 0; off >>= 1) {
        if (tid < off) { s1[tid] += s1[tid + off]; s2[tid] += s2[tid + off]; }
        __syncthreads();
    }
    float sum_dxhat   = s1[0];
    float sum_dxhat_x = s2[0];

    float invD = 1.0f / (float)D;
    for (int i = tid; i < D; i += bs) {
        float xhat  = (__half2float(xr[i]) - mean) * rstd;
        float gv    = __half2float(gamma[i]);
        float dxhat = __half2float(dyr[i]) * gv;
        float dx    = invD * rstd * ((float)D * dxhat - sum_dxhat - xhat * sum_dxhat_x);
        dxr[i] = __float2half(dx);
    }
}

void batched_ln_bwd_dX(const bt::GpuTensor& X,
                       const bt::GpuTensor& gamma,
                       const bt::GpuTensor& dY,
                       bt::GpuTensor& dX,
                       float eps) {
    const int L = dY.rows;
    const int D = dY.cols;
    if (dX.rows != L || dX.cols != D || dX.dtype != bt::Dtype::FP16) {
        dX.resize(L, D, bt::Dtype::FP16);
    }
    if (L == 0 || D == 0) return;
    int bs = 128;
    while (bs > D) bs >>= 1;
    if (bs < 32) bs = 32;
    const size_t smem = 2 * bs * sizeof(float);
    batched_ln_bwd_dX_k<<<L, bs, smem>>>(
        reinterpret_cast<const __half*>(X.data_fp16()),
        reinterpret_cast<const __half*>(gamma.data_fp16()),
        reinterpret_cast<const __half*>(dY.data_fp16()),
        reinterpret_cast<__half*>(dX.data_fp16()),
        D, eps);
}

// ─── Sink-buffer manager: thread_local scratch for discarded teacher grads ───
struct SinkPool {
    bt::GpuTensor t[24];
    int next = 0;
    bt::GpuTensor& get(int r, int c) {
        bt::GpuTensor& s = t[next];
        next = (next + 1) % 24;
        if (s.rows != r || s.cols != c || s.dtype != bt::Dtype::FP16) {
            s.resize(r, c, bt::Dtype::FP16);
        }
        s.zero();
        return s;
    }
};

}  // namespace

// ─── Main: forward replay + dX-only backward ─────────────────────────────────

void unet_up_path_backward(const UNet& net,
                           const bt::GpuTensor& bottleneck_in,
                           const std::array<const bt::GpuTensor*, 12>& skips_in,
                           const bt::GpuTensor& ctx,
                           const bt::GpuTensor& t_emb_raw,
                           const bt::GpuTensor& d_eps_pred,
                           std::array<bt::GpuTensor, 12>& d_skips_out,
                           bt::GpuTensor& d_bottleneck_out) {
    using Resnet        = UNet::Resnet;
    using Transformer2D = UNet::Transformer2D;

    const UNetConfig& cfg = net.cfg_;
    const int nb       = static_cast<int>(cfg.block_out_channels.size());
    const int first_C  = cfg.block_out_channels.front();
    const int mid_C    = cfg.block_out_channels.back();
    (void)mid_C;

    if (net.conv_in_W_.size() == 0) fail("teacher weights not loaded");
    if (skips_in.size() != 12) fail("expected 12 skip tensors");

    thread_local SinkPool g_sink;
    g_sink.next = 0;

    // ── 1. SiLU(t_emb_raw) once ─────────────────────────────────────────────
    thread_local bt::GpuTensor temb_silu;
    bt::silu_forward_gpu(t_emb_raw, temb_silu);

    // ── 2. Per-op forward replay caches ─────────────────────────────────────
    struct ResnetCache {
        bt::GpuTensor x_in;
        bt::GpuTensor temb_proj;  // (1, C_out) projected, fed to t_emb_shift
        int H, W;
        const Resnet* r;
    };
    struct TransformerCache {
        bt::GpuTensor x_in_nchw;      // pre-GN
        bt::GpuTensor gn_out_nchw;    // post-GN (= seq input pre-permute)
        bt::GpuTensor seq_pre_proj;   // (L, C) — input to proj_in
        bt::GpuTensor tseq_pre_ln1;   // (L, C) — after proj_in, pre self-attn
        bt::GpuTensor tseq_pre_ln2;   // after self-attn residual
        bt::GpuTensor tseq_pre_ln3;   // after cross-attn residual
        bt::GpuTensor tseq_post_ff;   // after ff residual — input to proj_out
        bt::GpuTensor ln1_out;        // LN1 output (= self-attn input)
        bt::GpuTensor ln2_out;        // LN2 output (= cross-attn input)
        bt::GpuTensor ln3_out;        // LN3 output (= ff1 input)
        bt::GpuTensor ff1_out;        // (L, 8C) — input to GEGLU
        bt::GpuTensor ff_geglu;       // (L, 4C) — output of GEGLU, input to ff2
        bt::GpuTensor xattn_K, xattn_V;  // (Lk, D) recomputed cross-attn K/V
        int H, W;
        const Transformer2D* t;
    };
    struct UpsamplerCache {
        bt::GpuTensor x_in;        // (1, C, H, W) pre-upsample
        bt::GpuTensor upsamp_out;  // (1, C, 2H, 2W) post-upsample, pre-conv
        int C, H, W;
        const bt::GpuTensor* Wconv;
        const bt::GpuTensor* bconv;
    };
    struct ConcatCache {
        int C_x, C_skip, H, W;
        int skip_index;
    };

    thread_local std::vector<ResnetCache>      rc_stack;
    thread_local std::vector<TransformerCache> xc_stack;
    thread_local std::vector<UpsamplerCache>   uc_stack;
    thread_local std::vector<ConcatCache>      cc_stack;
    rc_stack.clear();
    xc_stack.clear();
    uc_stack.clear();
    cc_stack.clear();

    // The "skip stack" indexes we still have to consume during the up path
    // (LIFO; we pop from the top each iteration). skips_in[0..11] are pushed
    // in order; up path pops from 11 downward.
    int next_skip_pop = 11;

    // ── 3. Forward replay helpers ───────────────────────────────────────────
    thread_local bt::GpuTensor tmp_a, tmp_b;

    auto fwd_resnet = [&](const Resnet& r, int H, int W, bt::GpuTensor& x) {
        ResnetCache rc;
        rc.x_in = x.clone();
        rc.H = H; rc.W = W; rc.r = &r;
        bt::linear_forward_batched_fp16_gpu(r.temb_W, &r.temb_b, temb_silu, rc.temb_proj);
        const bt::GpuTensor* Wskip = r.has_shortcut ? &r.Ws : nullptr;
        const bt::GpuTensor* bskip = r.has_shortcut ? &r.bs : nullptr;
        bt::resblock_forward_gpu(rc.x_in,
                                 r.n1g, r.n1b, r.W1, &r.b1,
                                 &rc.temb_proj,
                                 r.n2g, r.n2b, r.W2, &r.b2,
                                 Wskip, bskip,
                                 1, r.C_in, r.C_out, H, W,
                                 cfg.norm_num_groups, cfg.eps,
                                 tmp_a);
        std::swap(x, tmp_a);
        rc_stack.push_back(std::move(rc));
    };

    auto fwd_xform = [&](const Transformer2D& t, int H, int W, bt::GpuTensor& x) {
        if (t.blocks.size() != 1) fail("Transformer2D must have exactly 1 inner block");
        const auto& blk = t.blocks[0];
        TransformerCache tc;
        tc.H = H; tc.W = W; tc.t = &t;

        tc.x_in_nchw = x.clone();
        bt::group_norm_forward_gpu(tc.x_in_nchw, t.gn_g, t.gn_b,
                                   1, t.C, H, W, cfg.norm_num_groups, 1e-6f,
                                   tc.gn_out_nchw);
        bt::nchw_to_sequence_gpu(tc.gn_out_nchw, 1, t.C, H, W, tc.seq_pre_proj);
        bt::GpuTensor proj_in_seq;
        bt::linear_forward_batched_fp16_gpu(t.pi_W, &t.pi_b, tc.seq_pre_proj, proj_in_seq);
        tc.tseq_pre_ln1 = proj_in_seq.clone();

        // self-attn
        bt::layernorm_forward_inference_batched_fp16_gpu(
            tc.tseq_pre_ln1, blk.n1g, blk.n1b, tc.ln1_out, cfg.eps);
        bt::GpuTensor sa_out;
        bt::flash_attention_qkvo_forward_gpu(
            tc.ln1_out, /*Ctx=*/nullptr,
            blk.Wq1, nullptr, blk.Wk1, nullptr, blk.Wv1, nullptr,
            blk.Wo1, &blk.bo1, nullptr, t.num_heads, false, sa_out);
        tc.tseq_pre_ln2 = tc.tseq_pre_ln1.clone();
        bt::add_inplace_gpu(tc.tseq_pre_ln2, sa_out);

        // cross-attn
        bt::flash_attention_project_kv_gpu(ctx, blk.Wk2, nullptr,
                                           blk.Wv2, nullptr,
                                           tc.xattn_K, tc.xattn_V);
        bt::layernorm_forward_inference_batched_fp16_gpu(
            tc.tseq_pre_ln2, blk.n2g, blk.n2b, tc.ln2_out, cfg.eps);
        bt::GpuTensor xa_out;
        bt::flash_attention_qkvo_forward_gpu(
            tc.ln2_out, &ctx,
            blk.Wq2, nullptr, blk.Wk2, nullptr, blk.Wv2, nullptr,
            blk.Wo2, &blk.bo2, nullptr, t.num_heads, false, xa_out);
        tc.tseq_pre_ln3 = tc.tseq_pre_ln2.clone();
        bt::add_inplace_gpu(tc.tseq_pre_ln3, xa_out);

        // FF: ln3 → ff1 → geglu → ff2 → add.
        bt::layernorm_forward_inference_batched_fp16_gpu(
            tc.tseq_pre_ln3, blk.n3g, blk.n3b, tc.ln3_out, cfg.eps);
        bt::linear_forward_batched_fp16_gpu(blk.ff1_W, &blk.ff1_b, tc.ln3_out, tc.ff1_out);
        bt::geglu_exact_forward_gpu(tc.ff1_out, tc.ff_geglu);
        bt::GpuTensor ff2_out;
        bt::linear_forward_batched_fp16_gpu(blk.ff2_W, &blk.ff2_b, tc.ff_geglu, ff2_out);
        tc.tseq_post_ff = tc.tseq_pre_ln3.clone();
        bt::add_inplace_gpu(tc.tseq_post_ff, ff2_out);

        // proj_out, seq→NCHW, residual add into x.
        bt::GpuTensor proj_out_seq;
        bt::linear_forward_batched_fp16_gpu(t.po_W, &t.po_b, tc.tseq_post_ff, proj_out_seq);
        bt::GpuTensor proj_out_nchw;
        bt::sequence_to_nchw_gpu(proj_out_seq, 1, t.C, H, W, proj_out_nchw);
        bt::add_inplace_gpu(x, proj_out_nchw);

        xc_stack.push_back(std::move(tc));
    };

    // ── 4. Forward: mid + up + conv_out ─────────────────────────────────────
    bt::GpuTensor x_cur = bottleneck_in.clone();
    int Hc = 8, Wc = 8;

    // mid: r0 → t → r1
    fwd_resnet(net.mid_.r0, Hc, Wc, x_cur);
    fwd_xform(net.mid_.t, Hc, Wc, x_cur);
    fwd_resnet(net.mid_.r1, Hc, Wc, x_cur);

    // up_blocks
    for (int i = 0; i < nb; ++i) {
        const auto& u = net.up_blocks_[static_cast<std::size_t>(i)];
        const int layers = cfg.layers_per_block + 1;
        for (int j = 0; j < layers; ++j) {
            // concat with skip popped from inlet's stack.
            if (next_skip_pop < 0) fail("skip stack underflow");
            const bt::GpuTensor* skip = skips_in[static_cast<std::size_t>(next_skip_pop)];
            const int C_x_now    = x_cur.cols / (Hc * Wc);
            const int C_skip_now = skip->cols / (Hc * Wc);
            ConcatCache cc;
            cc.C_x = C_x_now; cc.C_skip = C_skip_now;
            cc.H = Hc; cc.W = Wc;
            cc.skip_index = next_skip_pop;
            cc_stack.push_back(cc);
            next_skip_pop--;

            std::vector<const bt::GpuTensor*> parts = {&x_cur, skip};
            std::vector<int> C_parts = {C_x_now, C_skip_now};
            bt::concat_nchw_channels_gpu(parts, 1, Hc, Wc, C_parts, tmp_b);
            std::swap(x_cur, tmp_b);

            fwd_resnet(u.resnets[static_cast<std::size_t>(j)], Hc, Wc, x_cur);
            if (u.has_attention) {
                fwd_xform(u.transformers[static_cast<std::size_t>(j)], Hc, Wc, x_cur);
            }
        }
        if (u.has_upsampler) {
            UpsamplerCache uc;
            uc.x_in = x_cur.clone();
            uc.C = u.C_out; uc.H = Hc; uc.W = Wc;
            uc.Wconv = &u.upsampler.W; uc.bconv = &u.upsampler.b;
            bt::upsample_nearest_2x_gpu(x_cur, 1, u.C_out, Hc, Wc, uc.upsamp_out);
            // 3x3 stride 1 same-channels.
            bt::conv2d_forward_gpu(uc.upsamp_out, *uc.Wconv, uc.bconv,
                                   1, u.C_out, 2 * Hc, 2 * Wc,
                                   u.C_out, 3, 3, 1, 1, 1, 1, 1, 1,
                                   tmp_a);
            std::swap(x_cur, tmp_a);
            Hc *= 2; Wc *= 2;
            uc_stack.push_back(std::move(uc));
        }
    }

    if (next_skip_pop != -1) fail("skip stack not drained");
    if (Hc != 64 || Wc != 64) fail("post-up spatial dims must be 64x64 for SD1.5");

    // conv_norm_out → silu → conv_out
    // Cache: x_cur (= norm_out input), silu input (= GN output).
    thread_local bt::GpuTensor norm_out_in_cache;
    thread_local bt::GpuTensor silu_in_cache;     // input to silu, = GN out
    norm_out_in_cache = x_cur.clone();
    bt::group_norm_forward_gpu(norm_out_in_cache,
                               net.norm_out_g_, net.norm_out_b_,
                               1, first_C, Hc, Wc, cfg.norm_num_groups, cfg.eps,
                               silu_in_cache);
    thread_local bt::GpuTensor conv_out_in_cache;  // = silu output
    bt::silu_forward_gpu(silu_in_cache, conv_out_in_cache);
    // conv_out forward (just to verify shapes; we don't actually need its
    // output here — d_eps_pred is the upstream grad).
    // (skip recomputation; brodiffusion already produced the eps; we trust
    // the caller's d_eps_pred.)

    // ── 5. BACKWARD walk ────────────────────────────────────────────────────
    // dY at conv_out output = d_eps_pred. Move backward through:
    //   conv_out → silu → norm_out → up_blocks(reverse) → mid(reverse) → bottleneck.
    thread_local bt::GpuTensor dY;          // running gradient (op output side)
    thread_local bt::GpuTensor dY_next;     // scratch

    // 5a. conv_out backward (dX only).
    // Forward: y = conv3x3_same(conv_out_in_cache, conv_out_W_, conv_out_b_)
    bt::conv2d_backward_input_gpu(net.conv_out_W_, d_eps_pred,
                                  1, first_C, Hc, Wc,
                                  cfg.out_channels, 3, 3,
                                  1, 1, 1, 1, 1, 1,
                                  /*groups=*/1, dY);

    // 5b. silu backward. Input was silu_in_cache (= GN output).
    bt::silu_backward_gpu(silu_in_cache, dY, dY_next);
    std::swap(dY, dY_next);

    // 5c. group_norm backward (norm_out). Input was norm_out_in_cache.
    {
        bt::GpuTensor& dG = g_sink.get(first_C, 1);
        bt::GpuTensor& dB = g_sink.get(first_C, 1);
        bt::group_norm_backward_gpu(norm_out_in_cache, net.norm_out_g_, dY,
                                    1, first_C, Hc, Wc, cfg.norm_num_groups,
                                    cfg.eps, dY_next, dG, dB);
        std::swap(dY, dY_next);
    }

    // ── 5d. up_blocks in REVERSE ────────────────────────────────────────────
    // Mirror the forward up loop in reverse: upsampler → (layers reversed) →
    // for each layer: transformer(if attn) → resnet → concat-split.
    int rc_idx = static_cast<int>(rc_stack.size()) - 1;
    int xc_idx = static_cast<int>(xc_stack.size()) - 1;
    int uc_idx = static_cast<int>(uc_stack.size()) - 1;
    int cc_idx = static_cast<int>(cc_stack.size()) - 1;

    auto bwd_resnet = [&](const ResnetCache& rc) {
        // dY at resblock output → dX at resblock input.
        const Resnet& r = *rc.r;
        const bt::GpuTensor* Wskip = r.has_shortcut ? &r.Ws : nullptr;
        const bt::GpuTensor* bskip = r.has_shortcut ? &r.bs : nullptr;
        bt::GpuTensor& dG1 = g_sink.get(r.C_in, 1);
        bt::GpuTensor& dB1 = g_sink.get(r.C_in, 1);
        bt::GpuTensor& dG2 = g_sink.get(r.C_out, 1);
        bt::GpuTensor& dB2 = g_sink.get(r.C_out, 1);
        bt::GpuTensor& dW1 = g_sink.get(r.C_out, r.C_in * 9);
        bt::GpuTensor& dW2 = g_sink.get(r.C_out, r.C_out * 9);
        bt::GpuTensor& db1 = g_sink.get(r.C_out, 1);
        bt::GpuTensor& db2 = g_sink.get(r.C_out, 1);
        bt::GpuTensor& dtemb = g_sink.get(rc.temb_proj.rows, rc.temb_proj.cols);
        bt::GpuTensor* dWskip_p = nullptr;
        bt::GpuTensor* dbskip_p = nullptr;
        bt::GpuTensor dWskip_buf, dbskip_buf;
        if (r.has_shortcut) {
            dWskip_buf.resize(r.C_out, r.C_in, bt::Dtype::FP16); dWskip_buf.zero();
            dbskip_buf.resize(r.C_out, 1, bt::Dtype::FP16); dbskip_buf.zero();
            dWskip_p = &dWskip_buf;
            dbskip_p = &dbskip_buf;
        }
        bt::resblock_backward_gpu(rc.x_in,
                                  r.n1g, r.n1b, r.W1, &r.b1,
                                  &rc.temb_proj,
                                  r.n2g, r.n2b, r.W2, &r.b2,
                                  Wskip, bskip,
                                  1, r.C_in, r.C_out, rc.H, rc.W,
                                  cfg.norm_num_groups, cfg.eps,
                                  dY, dY_next,
                                  dG1, dB1, dW1, &db1, &dtemb,
                                  dG2, dB2, dW2, &db2,
                                  dWskip_p, dbskip_p);
        std::swap(dY, dY_next);
    };

    auto bwd_xform = [&](const TransformerCache& tc) {
        // dY currently holds dY at the *output of the residual add* in NCHW.
        // Residual: x_out = x_in + proj_out_nchw  →  dx_in_branch = dY,
        // dproj_out_nchw_branch = dY. We process the sub-branch first, then
        // the dY accumulated on the residual branch is the running gradient
        // at the start of the transformer block input.
        const Transformer2D& t = *tc.t;
        const auto& blk = t.blocks[0];

        // Save residual branch (dY itself becomes dx_in's contribution from
        // the residual).
        bt::GpuTensor d_residual = dY.clone();

        // Backward sub-branch: NCHW dY → seq → proj_out backward → ff backward
        // → cross-attn backward → self-attn backward → proj_in backward →
        // seq→NCHW → GN backward → dx_in.

        // 1. seq←NCHW for the dY going to proj_out output.
        bt::GpuTensor d_proj_out_seq;
        bt::nchw_to_sequence_gpu(dY, 1, t.C, tc.H, tc.W, d_proj_out_seq);

        // 2. proj_out backward (Linear). Input was tseq_post_ff (L, C).
        bt::GpuTensor d_tseq_post_ff;
        bt::GpuTensor& dW_po = g_sink.get(t.C, t.C);
        bt::GpuTensor& db_po = g_sink.get(t.C, 1);
        bt::linear_backward_batched_gpu(t.po_W, tc.tseq_post_ff, d_proj_out_seq,
                                d_tseq_post_ff, dW_po, db_po);

        // 3. FF residual: tseq_post_ff = tseq_pre_ln3 + ff2_out.
        //    d_tseq_pre_ln3 ← d_tseq_post_ff (residual)
        //    d_ff2_out      ← d_tseq_post_ff
        bt::GpuTensor d_tseq_pre_ln3 = d_tseq_post_ff.clone();
        bt::GpuTensor& d_ff2_out = d_tseq_post_ff;  // alias, consumed below

        // 4. ff2 backward. Input was tc.ff_geglu (L, 4C). Wff2 (C, 4C).
        bt::GpuTensor d_ff_geglu;
        bt::GpuTensor& dW_ff2 = g_sink.get(blk.ff2_W.rows, blk.ff2_W.cols);
        bt::GpuTensor& db_ff2 = g_sink.get(t.C, 1);
        bt::linear_backward_batched_gpu(blk.ff2_W, tc.ff_geglu, d_ff2_out,
                                d_ff_geglu, dW_ff2, db_ff2);

        // 5. GEGLU backward. Input X = ff1_out (L, 8C).
        bt::GpuTensor d_ff1_out;
        bt::geglu_exact_backward_gpu(tc.ff1_out, d_ff_geglu, d_ff1_out);

        // 6. ff1 backward. Input was tc.ln3_out (L, C). Wff1 (8C, C).
        bt::GpuTensor d_ln3_out;
        bt::GpuTensor& dW_ff1 = g_sink.get(blk.ff1_W.rows, blk.ff1_W.cols);
        bt::GpuTensor& db_ff1 = g_sink.get(blk.ff1_b.rows, 1);
        bt::linear_backward_batched_gpu(blk.ff1_W, tc.ln3_out, d_ff1_out,
                                d_ln3_out, dW_ff1, db_ff1);

        // 7. LN3 backward (dX only). Input was tseq_pre_ln3.
        bt::GpuTensor d_ln3_in;
        batched_ln_bwd_dX(tc.tseq_pre_ln3, blk.n3g, d_ln3_out, d_ln3_in, cfg.eps);
        // Accumulate into d_tseq_pre_ln3 (residual branch).
        bt::add_inplace_gpu(d_tseq_pre_ln3, d_ln3_in);

        // 8. Cross-attn residual: tseq_pre_ln3 = tseq_pre_ln2 + cross_attn_out.
        bt::GpuTensor d_tseq_pre_ln2 = d_tseq_pre_ln3.clone();
        bt::GpuTensor& d_xa_out = d_tseq_pre_ln3;  // alias, consumed below

        // 9. Cross-attn backward. dX = grad on ln2_out (Q side); dCtx = grad on ctx.
        // We don't propagate dCtx back to the caller (ctx is external) but we
        // must still pass a real buffer because Ctx != nullptr.
        bt::GpuTensor& d_ctx_sink = g_sink.get(ctx.rows, ctx.cols);
        bt::GpuTensor d_ln2_out;
        bt::GpuTensor& dWq = g_sink.get(blk.Wq2.rows, blk.Wq2.cols);
        bt::GpuTensor& dWk = g_sink.get(blk.Wk2.rows, blk.Wk2.cols);
        bt::GpuTensor& dWv = g_sink.get(blk.Wv2.rows, blk.Wv2.cols);
        bt::GpuTensor& dWo = g_sink.get(blk.Wo2.rows, blk.Wo2.cols);
        bt::GpuTensor& dbo = g_sink.get(t.C, 1);
        bt::flash_attention_qkvo_backward_gpu(
            tc.ln2_out, &ctx,
            blk.Wq2, nullptr, blk.Wk2, nullptr, blk.Wv2, nullptr,
            blk.Wo2, &blk.bo2,
            nullptr, t.num_heads, false,
            d_xa_out,
            d_ln2_out, &d_ctx_sink,
            dWq, nullptr, dWk, nullptr, dWv, nullptr,
            dWo, &dbo);

        // 10. LN2 backward.
        bt::GpuTensor d_ln2_in;
        batched_ln_bwd_dX(tc.tseq_pre_ln2, blk.n2g, d_ln2_out, d_ln2_in, cfg.eps);
        bt::add_inplace_gpu(d_tseq_pre_ln2, d_ln2_in);

        // 11. Self-attn residual: tseq_pre_ln2 = tseq_pre_ln1 + self_attn_out.
        bt::GpuTensor d_tseq_pre_ln1 = d_tseq_pre_ln2.clone();
        bt::GpuTensor& d_sa_out = d_tseq_pre_ln2;  // alias, consumed below

        // 12. Self-attn backward (Ctx = nullptr; K/V grads absorbed into dX).
        bt::GpuTensor d_ln1_out;
        bt::GpuTensor& dWq1 = g_sink.get(blk.Wq1.rows, blk.Wq1.cols);
        bt::GpuTensor& dWk1 = g_sink.get(blk.Wk1.rows, blk.Wk1.cols);
        bt::GpuTensor& dWv1 = g_sink.get(blk.Wv1.rows, blk.Wv1.cols);
        bt::GpuTensor& dWo1 = g_sink.get(blk.Wo1.rows, blk.Wo1.cols);
        bt::GpuTensor& dbo1 = g_sink.get(t.C, 1);
        bt::flash_attention_qkvo_backward_gpu(
            tc.ln1_out, nullptr,
            blk.Wq1, nullptr, blk.Wk1, nullptr, blk.Wv1, nullptr,
            blk.Wo1, &blk.bo1,
            nullptr, t.num_heads, false,
            d_sa_out,
            d_ln1_out, nullptr,
            dWq1, nullptr, dWk1, nullptr, dWv1, nullptr,
            dWo1, &dbo1);

        // 13. LN1 backward.
        bt::GpuTensor d_ln1_in;
        batched_ln_bwd_dX(tc.tseq_pre_ln1, blk.n1g, d_ln1_out, d_ln1_in, cfg.eps);
        bt::add_inplace_gpu(d_tseq_pre_ln1, d_ln1_in);

        // 14. proj_in backward. Input was tc.seq_pre_proj (L, C). Wpi (C, C).
        bt::GpuTensor d_seq_pre_proj;
        bt::GpuTensor& dW_pi = g_sink.get(t.C, t.C);
        bt::GpuTensor& db_pi = g_sink.get(t.C, 1);
        bt::linear_backward_batched_gpu(t.pi_W, tc.seq_pre_proj, d_tseq_pre_ln1,
                                d_seq_pre_proj, dW_pi, db_pi);

        // 15. seq→NCHW (inverse of nchw_to_sequence). Forward Y[seqpos, c] =
        // X[1, c, h, w]; inverse goes back the same way as sequence_to_nchw.
        bt::GpuTensor d_gn_out_nchw;
        bt::sequence_to_nchw_gpu(d_seq_pre_proj, 1, t.C, tc.H, tc.W, d_gn_out_nchw);

        // 16. group_norm backward (transformer-outer GN, eps=1e-6).
        bt::GpuTensor d_xform_in;
        bt::GpuTensor& dG_gn = g_sink.get(t.C, 1);
        bt::GpuTensor& dB_gn = g_sink.get(t.C, 1);
        bt::group_norm_backward_gpu(tc.x_in_nchw, t.gn_g, d_gn_out_nchw,
                                    1, t.C, tc.H, tc.W,
                                    cfg.norm_num_groups, 1e-6f,
                                    d_xform_in, dG_gn, dB_gn);

        // 17. Combine with residual branch: dY = d_residual + d_xform_in.
        bt::add_inplace_gpu(d_residual, d_xform_in);
        dY = std::move(d_residual);
    };

    // 5d. Up loop in reverse.
    for (int i = nb - 1; i >= 0; --i) {
        const auto& u = net.up_blocks_[static_cast<std::size_t>(i)];
        const int layers = cfg.layers_per_block + 1;
        if (u.has_upsampler) {
            // Forward was: upsample_nearest_2x → conv3x3_same.
            const UpsamplerCache& uc = uc_stack[static_cast<std::size_t>(uc_idx--)];
            // conv backward (input). dY is post-conv; input was uc.upsamp_out (2H, 2W).
            bt::GpuTensor d_upsamp_out;
            bt::conv2d_backward_input_gpu(*uc.Wconv, dY,
                                          1, uc.C, 2 * uc.H, 2 * uc.W,
                                          uc.C, 3, 3, 1, 1, 1, 1, 1, 1, 1,
                                          d_upsamp_out);
            // upsample backward (input H, W = pre-upsample dims).
            bt::GpuTensor d_pre_up;
            bt::upsample_nearest_2x_backward_gpu(d_upsamp_out, 1, uc.C, uc.H, uc.W,
                                                  d_pre_up);
            dY = std::move(d_pre_up);
        }
        for (int j = layers - 1; j >= 0; --j) {
            if (u.has_attention) {
                const TransformerCache& tc = xc_stack[static_cast<std::size_t>(xc_idx--)];
                bwd_xform(tc);
            }
            // resnet
            const ResnetCache& rc = rc_stack[static_cast<std::size_t>(rc_idx--)];
            bwd_resnet(rc);
            // concat-split: dY here is on the resblock input (which was the
            // concat output). Split channels back into dY_xpart (C_x) and
            // dY_skippart (C_skip), routing skippart into d_skips_out.
            const ConcatCache& cc = cc_stack[static_cast<std::size_t>(cc_idx--)];
            // Prepare skip-grad output and an x-grad target.
            bt::GpuTensor& dskip_out = d_skips_out[static_cast<std::size_t>(cc.skip_index)];
            if (dskip_out.rows != 1 ||
                dskip_out.cols != cc.C_skip * cc.H * cc.W ||
                dskip_out.dtype != bt::Dtype::FP16) {
                dskip_out.resize(1, cc.C_skip * cc.H * cc.W, bt::Dtype::FP16);
            }
            bt::GpuTensor d_xpart;
            d_xpart.resize(1, cc.C_x * cc.H * cc.W, bt::Dtype::FP16);
            std::vector<int> C_parts = {cc.C_x, cc.C_skip};
            std::vector<bt::GpuTensor*> outs = {&d_xpart, &dskip_out};
            bt::concat_nchw_channels_backward_gpu(dY, 1, cc.H, cc.W, C_parts, outs);
            dY = std::move(d_xpart);
        }
    }

    // 5e. mid: r1 → t → r0 (reverse).
    {
        const ResnetCache& rc1 = rc_stack[static_cast<std::size_t>(rc_idx--)];
        bwd_resnet(rc1);
        const TransformerCache& tcm = xc_stack[static_cast<std::size_t>(xc_idx--)];
        bwd_xform(tcm);
        const ResnetCache& rc0 = rc_stack[static_cast<std::size_t>(rc_idx--)];
        bwd_resnet(rc0);
    }

    if (rc_idx != -1 || xc_idx != -1 || uc_idx != -1 || cc_idx != -1) {
        fail("internal: cache stacks not fully consumed");
    }

    // 5f. dY is now the gradient on the bottleneck (mid r0 input).
    if (d_bottleneck_out.rows != bottleneck_in.rows ||
        d_bottleneck_out.cols != bottleneck_in.cols ||
        d_bottleneck_out.dtype != bt::Dtype::FP16) {
        d_bottleneck_out.resize(bottleneck_in.rows, bottleneck_in.cols, bt::Dtype::FP16);
    }
    // Copy dY into d_bottleneck_out.
    {
        const int n = dY.size();
        bt::copy_d2d_gpu(dY, 0, d_bottleneck_out, 0, n);
    }
}

}  // namespace brodiffusion::unet
