#include "brodiffusion/unet.h"
#include "brotensor/safetensors.h"
#include "brodiffusion/fused_resblock.h"
#include "brodiffusion/fused_transformer.h"
#include "brodiffusion/detail/device.h"
#include "brodiffusion/detail/compute.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace brodiffusion::unet {

namespace bt = ::brotensor;
namespace st = ::brotensor::safetensors;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("unet::UNet: " + msg);
}

// Concrete payload behind PreparedConditioning for the SD1.5 UNet: the
// cloned text context (cond + optional uncond) plus the pre-projected
// cross-attention K/V caches. Built by UNet::prepare, consumed by the
// Denoiser-interface UNet::forward.
struct UNetPrepared : public PreparedConditioning::Impl {
    brotensor::Tensor      ctx_cond, ctx_uncond;
    UNet::CrossAttnKVCache cache_cond, cache_uncond;
    float                  lcm_guidance = 0.0f;
};

}  // namespace

// ─── ctor / dtor ───────────────────────────────────────────────────────────

UNet::UNet(const UNetConfig& cfg) : cfg_(cfg) {
    const int nb = static_cast<int>(cfg_.block_out_channels.size());
    if (nb < 2) fail("block_out_channels must have at least 2 entries");
    if (cfg_.layers_per_block <= 0) fail("layers_per_block must be positive");
    if (cfg_.attention_head_dim <= 0) fail("attention_head_dim must be positive");
    if (cfg_.cross_attention_dim <= 0) fail("cross_attention_dim must be positive");
    if (cfg_.time_embed_dim_mult <= 0) fail("time_embed_dim_mult must be positive");
    if (cfg_.time_cond_proj_dim < 0) fail("time_cond_proj_dim must be >= 0");
    if (cfg_.time_cond_proj_dim > 0 && (cfg_.time_cond_proj_dim % 2) != 0) {
        fail("time_cond_proj_dim must be even (sinusoidal embedding)");
    }
    for (int c : cfg_.block_out_channels) {
        if (c <= 0 || c % cfg_.norm_num_groups != 0) {
            fail("each block_out_channels entry must be a positive multiple of norm_num_groups");
        }
        if (c % cfg_.attention_head_dim != 0) {
            fail("each block_out_channels entry must be divisible by attention_head_dim");
        }
    }

    freq_dim_       = cfg_.block_out_channels.front();
    if (freq_dim_ % 2 != 0) fail("block_out_channels[0] must be even (sinusoidal embedding)");
    time_embed_dim_ = cfg_.block_out_channels.front() * cfg_.time_embed_dim_mult;

    down_blocks_.resize(static_cast<std::size_t>(nb));
    up_blocks_.resize(static_cast<std::size_t>(nb));
    for (int i = 0; i < nb; ++i) {
        // Down: all except the last are cross-attention + downsample.
        DownBlock& d = down_blocks_[static_cast<std::size_t>(i)];
        d.has_attention   = (i < nb - 1);
        d.has_downsampler = (i < nb - 1);
        d.C_out = cfg_.block_out_channels[static_cast<std::size_t>(i)];

        // Up (reversed): all except the first are cross-attention; all except
        // the last have an upsampler.
        UpBlock& u = up_blocks_[static_cast<std::size_t>(i)];
        u.has_attention = (i > 0);
        u.has_upsampler = (i < nb - 1);
        u.C_out = cfg_.block_out_channels[static_cast<std::size_t>(nb - 1 - i)];
    }
}

UNet::~UNet() = default;

// ─── per-block forward helpers ─────────────────────────────────────────────
//
// Phase D1 lifted the per-block helpers (apply_resnet, apply_transformer,
// apply_conv3x3, apply_conv3x3_q) into brodiffusion::unet::detail (see
// src/unet_blocks.cpp). UNet now forwards to those free functions with its
// `block_scratch_` member supplying the scratch tensors.

// ─── forward ───────────────────────────────────────────────────────────────

namespace {

// Sinusoidal embedding matching diffusers' `get_guidance_scale_embedding`
// helper used by LCM. Two important differences from the timestep helper:
//   1. The frequency divisor is `(half - 1)`, not `half`.
//   2. The output layout is [sin..., cos...] (no flip_sin_to_cos).
// Callers should pre-multiply w by 1000 (diffusers does this before calling
// the helper).
//   Output layout: [sin(w*freq_0), ..., sin(w*freq_{H-1}), cos(w*freq_0), ...].
void compute_guidance_scale_emb(float w, int dim,
                                std::vector<float>& out) {
    const int half = dim / 2;
    out.resize(static_cast<std::size_t>(dim));
    const float log_period = std::log(10000.0f);
    const float denom = (half > 1) ? static_cast<float>(half - 1) : 1.0f;
    for (int i = 0; i < half; ++i) {
        const float freq = std::exp(-log_period * static_cast<float>(i) / denom);
        const float angle = w * freq;
        out[static_cast<std::size_t>(i)]        = std::sin(angle);
        out[static_cast<std::size_t>(i + half)] = std::cos(angle);
    }
}

// Sinusoidal timestep embedding matching diffusers `get_timestep_embedding`
// with flip_sin_to_cos=True, downscale_freq_shift=0, scale=1, max_period=10000.
// Output layout: [cos(t*freq_0), ..., cos(t*freq_{H-1}), sin(t*freq_0), ...].
void compute_sinusoidal_emb(float t, int dim,
                            std::vector<float>& out) {
    const int half = dim / 2;
    out.resize(static_cast<std::size_t>(dim));
    const float log_period = std::log(10000.0f);
    for (int i = 0; i < half; ++i) {
        const float freq = std::exp(-log_period * static_cast<float>(i) /
                                     static_cast<float>(half));
        const float angle = t * freq;
        out[static_cast<std::size_t>(i)]        = std::cos(angle);
        out[static_cast<std::size_t>(i + half)] = std::sin(angle);
    }
}

}  // namespace

// ── Denoiser interface ─────────────────────────────────────────────────────

brotensor::Dtype UNet::compute_dtype() const {
    return brodiffusion::compute_dtype();
}

PreparedConditioning UNet::prepare(const Conditioning& cond) {
    auto p = std::make_unique<UNetPrepared>();
    p->ctx_cond = cond.text_embeddings.clone();
    prime_xattn_cache(p->ctx_cond, p->cache_cond);
    if (cond.has_uncond) {
        p->ctx_uncond = cond.uncond_embeddings.clone();
        prime_xattn_cache(p->ctx_uncond, p->cache_uncond);
    }
    p->lcm_guidance = cond.guidance;
    return PreparedConditioning(std::move(p));
}

const UNet::CrossAttnKVCache&
UNet::kv_cache_for(const PreparedConditioning& prepared, Branch branch) const {
    auto* p = static_cast<UNetPrepared*>(prepared.get());
    if (p == nullptr) fail("kv_cache_for: prepare() was not called");
    const bool uncond = (branch == Branch::Uncond);
    if (uncond && p->cache_uncond.empty()) {
        fail("kv_cache_for: Branch::Uncond requested but no uncond "
             "conditioning was prepared");
    }
    return uncond ? p->cache_uncond : p->cache_cond;
}

void UNet::forward(const bt::Tensor& latent, int H, int W, float timestep,
                   const PreparedConditioning& prepared, Branch branch,
                   bt::Tensor& out) {
    auto* p = static_cast<UNetPrepared*>(prepared.get());
    if (p == nullptr) fail("forward: prepare() was not called");
    const bool uncond = (branch == Branch::Uncond);
    if (uncond && p->cache_uncond.empty()) {
        fail("forward: Branch::Uncond requested but no uncond conditioning "
             "was prepared");
    }
    const bt::Tensor& ctx = uncond ? p->ctx_uncond : p->ctx_cond;
    const CrossAttnKVCache& cache = uncond ? p->cache_uncond : p->cache_cond;
    if (cfg_.time_cond_proj_dim > 0) {
        // LCM: route through the guidance-scale-embedding overload.
        forward(latent, H, W, timestep, p->lcm_guidance, ctx, cache, out);
    } else {
        forward(latent, H, W, timestep, ctx, cache, out);
    }
}

void UNet::prepare_step(float timestep, const PreparedConditioning& prepared) {
    auto* p = static_cast<UNetPrepared*>(prepared.get());
    if (p == nullptr) fail("prepare_step: prepare() was not called");
    const float* gs = (cfg_.time_cond_proj_dim > 0) ? &p->lcm_guidance
                                                    : nullptr;
    compute_step_inputs_(timestep, gs);
}

void UNet::forward_body(const bt::Tensor& latent, int H, int W,
                        const PreparedConditioning& prepared, Branch branch,
                        bt::Tensor& out) {
    auto* p = static_cast<UNetPrepared*>(prepared.get());
    if (p == nullptr) fail("forward_body: prepare() was not called");
    const bool uncond = (branch == Branch::Uncond);
    if (uncond && p->cache_uncond.empty()) {
        fail("forward_body: Branch::Uncond requested but no uncond "
             "conditioning was prepared");
    }
    const bt::Tensor& ctx = uncond ? p->ctx_uncond : p->ctx_cond;
    const CrossAttnKVCache& cache = uncond ? p->cache_uncond : p->cache_cond;
    forward_body_impl_(latent, H, W, ctx, &cache,
                       /*attn_logit_biases=*/nullptr, /*trace_out=*/nullptr,
                       /*down_residuals=*/nullptr, /*mid_residual=*/nullptr,
                       out);
}

void UNet::forward_traced(
        const bt::Tensor& latent, int H, int W, float timestep,
        const PreparedConditioning& prepared, Branch branch,
        const std::vector<const bt::Tensor*>* attn_logit_biases,
        AttentionTrace* trace_out, bt::Tensor& out) {
    auto* p = static_cast<UNetPrepared*>(prepared.get());
    if (p == nullptr) fail("forward_traced: prepare() was not called");
    const bool uncond = (branch == Branch::Uncond);
    if (uncond && p->ctx_uncond.size() == 0) {
        fail("forward_traced: Branch::Uncond requested but no uncond "
             "conditioning was prepared");
    }
    // forward_trace reprojects K/V from the raw context (it bypasses the K/V
    // cache), so the trace path needs ctx, not p->cache_*. An LCM-distilled
    // U-Net routes the guidance scale through cond_proj — pass it through.
    const bt::Tensor& ctx = uncond ? p->ctx_uncond : p->ctx_cond;
    const float* gs = (cfg_.time_cond_proj_dim > 0) ? &p->lcm_guidance
                                                    : nullptr;
    forward_trace(latent, H, W, timestep, gs, ctx, attn_logit_biases,
                  trace_out, out);
}

void UNet::forward(const bt::Tensor& sample,
                   int H, int W,
                   float timestep,
                   const bt::Tensor& encoder_hidden_states,
                   bt::Tensor& out) {
    if (cfg_.time_cond_proj_dim > 0) {
        fail("forward: UNet built with time_cond_proj_dim>0 requires the "
             "guidance_scale_embedding overload");
    }
    forward_impl_(sample, H, W, timestep, /*gs_emb=*/nullptr,
                  encoder_hidden_states, /*xattn_cache=*/nullptr,
                  /*attn_logit_biases=*/nullptr, /*trace_out=*/nullptr,
                  /*down_residuals=*/nullptr, /*mid_residual=*/nullptr, out);
}

void UNet::forward(const bt::Tensor& sample,
                   int H, int W,
                   float timestep,
                   const bt::Tensor& encoder_hidden_states,
                   const CrossAttnKVCache& xattn_cache,
                   bt::Tensor& out) {
    if (cfg_.time_cond_proj_dim > 0) {
        fail("forward: UNet built with time_cond_proj_dim>0 requires the "
             "guidance_scale_embedding overload");
    }
    if (static_cast<int>(xattn_cache.size()) != num_xattn_blocks()) {
        fail("forward: cross-attn KV cache has " +
             std::to_string(xattn_cache.size()) + " entries, expected " +
             std::to_string(num_xattn_blocks()));
    }
    forward_impl_(sample, H, W, timestep, /*gs_emb=*/nullptr,
                  encoder_hidden_states, &xattn_cache,
                  /*attn_logit_biases=*/nullptr, /*trace_out=*/nullptr,
                  /*down_residuals=*/nullptr, /*mid_residual=*/nullptr, out);
}

void UNet::forward(const bt::Tensor& sample,
                   int H, int W,
                   float timestep,
                   const bt::Tensor& encoder_hidden_states,
                   const CrossAttnKVCache& xattn_cache,
                   const std::vector<const bt::Tensor*>& down_residuals,
                   const bt::Tensor* mid_residual,
                   bt::Tensor& out) {
    // ControlNet-augmented forward. Vanilla SD1.5 only (no LCM cond_proj —
    // ControlNet has no LCM-distilled variant). Same shape contract as the
    // cached forward; residual lengths are validated inside forward_impl_.
    if (cfg_.time_cond_proj_dim > 0) {
        fail("forward(controlnet): residual-aware forward is not wired for "
             "LCM-distilled U-Nets (time_cond_proj_dim>0)");
    }
    if (static_cast<int>(xattn_cache.size()) != num_xattn_blocks()) {
        fail("forward(controlnet): cross-attn KV cache has " +
             std::to_string(xattn_cache.size()) + " entries, expected " +
             std::to_string(num_xattn_blocks()));
    }
    const std::vector<const bt::Tensor*>* dr =
        down_residuals.empty() ? nullptr : &down_residuals;
    forward_impl_(sample, H, W, timestep, /*gs_emb=*/nullptr,
                  encoder_hidden_states, &xattn_cache,
                  /*attn_logit_biases=*/nullptr, /*trace_out=*/nullptr,
                  dr, mid_residual, out);
}

void UNet::forward(const bt::Tensor& sample,
                   int H, int W,
                   float timestep,
                   float guidance_scale_embedding,
                   const bt::Tensor& encoder_hidden_states,
                   const CrossAttnKVCache& xattn_cache,
                   bt::Tensor& out) {
    if (cfg_.time_cond_proj_dim <= 0) {
        fail("forward(guidance_scale_embedding): UNet built with "
             "time_cond_proj_dim=0; LCM cond_proj path unavailable");
    }
    if (static_cast<int>(xattn_cache.size()) != num_xattn_blocks()) {
        fail("forward: cross-attn KV cache has " +
             std::to_string(xattn_cache.size()) + " entries, expected " +
             std::to_string(num_xattn_blocks()));
    }
    forward_impl_(sample, H, W, timestep, &guidance_scale_embedding,
                  encoder_hidden_states, &xattn_cache,
                  /*attn_logit_biases=*/nullptr, /*trace_out=*/nullptr,
                  /*down_residuals=*/nullptr, /*mid_residual=*/nullptr, out);
}

void UNet::forward(const bt::Tensor& sample,
                   int H, int W,
                   float timestep,
                   float guidance_scale_embedding,
                   const bt::Tensor& encoder_hidden_states,
                   const CrossAttnKVCache& xattn_cache,
                   const std::vector<const bt::Tensor*>& down_residuals,
                   const bt::Tensor* mid_residual,
                   bt::Tensor& out) {
    if (cfg_.time_cond_proj_dim <= 0) {
        fail("forward(controlnet+lcm): UNet built with time_cond_proj_dim=0; "
             "use the non-guidance overload for vanilla SD1.5");
    }
    if (static_cast<int>(xattn_cache.size()) != num_xattn_blocks()) {
        fail("forward(controlnet+lcm): cross-attn KV cache has " +
             std::to_string(xattn_cache.size()) + " entries, expected " +
             std::to_string(num_xattn_blocks()));
    }
    const std::vector<const bt::Tensor*>* dr =
        down_residuals.empty() ? nullptr : &down_residuals;
    forward_impl_(sample, H, W, timestep, &guidance_scale_embedding,
                  encoder_hidden_states, &xattn_cache,
                  /*attn_logit_biases=*/nullptr, /*trace_out=*/nullptr,
                  dr, mid_residual, out);
}

void UNet::forward_trace(const bt::Tensor& sample,
                         int H, int W,
                         float timestep,
                         const float* guidance_scale_embedding,
                         const bt::Tensor& encoder_hidden_states,
                         const std::vector<const bt::Tensor*>* attn_logit_biases,
                         CrossAttnTrace* trace_out,
                         bt::Tensor& out) {
    if (cfg_.quantize_weights) {
        fail("forward_trace: INT8 quantize_weights not yet supported in trace "
             "mode (brotensor::cross_attention_forward_with_attn is FP16 "
             "only)");
    }
    // The LCM cond_proj path and the trace plumbing are independent inside
    // forward_impl_, so trace mode supports LCM-distilled U-Nets — the caller
    // just supplies the guidance scale, exactly as the LCM forward() overload
    // requires. Enforce the same non-null-iff-LCM contract.
    if (guidance_scale_embedding != nullptr && cfg_.time_cond_proj_dim <= 0) {
        fail("forward_trace: guidance_scale_embedding supplied but UNet was "
             "built with time_cond_proj_dim=0");
    }
    if (guidance_scale_embedding == nullptr && cfg_.time_cond_proj_dim > 0) {
        fail("forward_trace: UNet built with time_cond_proj_dim>0 requires a "
             "guidance_scale_embedding (LCM cond_proj path)");
    }
    const int n = num_xattn_blocks();
    if (attn_logit_biases &&
        static_cast<int>(attn_logit_biases->size()) != n) {
        fail("forward_trace: attn_logit_biases has " +
             std::to_string(attn_logit_biases->size()) + " entries, expected " +
             std::to_string(n));
    }
    if (trace_out) {
        trace_out->resize(static_cast<std::size_t>(n));
    }
    forward_impl_(sample, H, W, timestep, guidance_scale_embedding,
                  encoder_hidden_states, /*xattn_cache=*/nullptr,
                  attn_logit_biases, trace_out,
                  /*down_residuals=*/nullptr, /*mid_residual=*/nullptr, out);
}

void UNet::forward_trace(const bt::Tensor& sample,
                         int H, int W,
                         float timestep,
                         const float* guidance_scale_embedding,
                         const bt::Tensor& encoder_hidden_states,
                         const std::vector<const bt::Tensor*>& down_residuals,
                         const bt::Tensor* mid_residual,
                         const std::vector<const bt::Tensor*>* attn_logit_biases,
                         CrossAttnTrace* trace_out,
                         bt::Tensor& out) {
    if (cfg_.quantize_weights) {
        fail("forward_trace(controlnet): INT8 quantize_weights not yet "
             "supported in trace mode");
    }
    if (guidance_scale_embedding != nullptr && cfg_.time_cond_proj_dim <= 0) {
        fail("forward_trace(controlnet): guidance_scale_embedding supplied "
             "but UNet was built with time_cond_proj_dim=0");
    }
    if (guidance_scale_embedding == nullptr && cfg_.time_cond_proj_dim > 0) {
        fail("forward_trace(controlnet): UNet built with "
             "time_cond_proj_dim>0 requires a guidance_scale_embedding "
             "(LCM cond_proj path)");
    }
    const int n = num_xattn_blocks();
    if (attn_logit_biases &&
        static_cast<int>(attn_logit_biases->size()) != n) {
        fail("forward_trace(controlnet): attn_logit_biases has " +
             std::to_string(attn_logit_biases->size()) +
             " entries, expected " + std::to_string(n));
    }
    if (trace_out) {
        trace_out->resize(static_cast<std::size_t>(n));
    }
    const std::vector<const bt::Tensor*>* dr =
        down_residuals.empty() ? nullptr : &down_residuals;
    forward_impl_(sample, H, W, timestep, guidance_scale_embedding,
                  encoder_hidden_states, /*xattn_cache=*/nullptr,
                  attn_logit_biases, trace_out,
                  dr, mid_residual, out);
}

int UNet::num_xattn_blocks() const {
    int n = 0;
    for (const DownBlock& d : down_blocks_) {
        if (d.has_attention) n += static_cast<int>(d.transformers.size());
    }
    n += 1;  // mid block always has one Transformer2D
    for (const UpBlock& u : up_blocks_) {
        if (u.has_attention) n += static_cast<int>(u.transformers.size());
    }
    return n;
}

std::vector<int> UNet::layer_strides() const {
    // SD1.5 schedule: down 0/1/2 (2 transformers each), mid (1), up 1/2/3
    // (3 transformers each). Strides relative to the top latent resolution.
    // Hard-coded for the stock SD1.5 UNet — reject any other block
    // configuration rather than return a silently mismatched vector.
    std::vector<int> strides = {1,1, 2,2, 4,4, 8, 4,4,4, 2,2,2, 1,1,1};
    if (static_cast<int>(strides.size()) != num_xattn_blocks()) {
        fail("layer_strides: hard-coded for the stock SD1.5 UNet; this UNet "
             "has a non-standard block configuration");
    }
    return strides;
}

void UNet::prime_xattn_cache(const bt::Tensor& ctx,
                             CrossAttnKVCache& cache) {
    if (conv_in_W_.size() == 0) fail("prime_xattn_cache: weights not loaded");
    const int n = num_xattn_blocks();
    cache.resize(static_cast<std::size_t>(n));

    int idx = 0;
    auto project = [&](const Transformer2D& t) {
        if (t.blocks.empty()) fail("internal: Transformer2D has no inner blocks");
        const AttnFFN& blk = t.blocks[0];
        CrossAttnKVCacheEntry& e = cache[static_cast<std::size_t>(idx++)];
        if (blk.Wk2_q.active()) {
            bt::flash_attention_project_kv_int8w_fp16(
                ctx,
                blk.Wk2_q.W_int8, blk.Wk2_q.scales, /*bk=*/nullptr,
                blk.Wv2_q.W_int8, blk.Wv2_q.scales, /*bv=*/nullptr,
                e.K, e.V);
        } else {
            bt::flash_attention_project_kv(
                ctx,
                blk.Wk2, /*bk=*/nullptr,
                blk.Wv2, /*bv=*/nullptr,
                e.K, e.V);
        }
    };

    for (const DownBlock& d : down_blocks_) {
        if (!d.has_attention) continue;
        for (const Transformer2D& t : d.transformers) project(t);
    }
    project(mid_.t);
    for (const UpBlock& u : up_blocks_) {
        if (!u.has_attention) continue;
        for (const Transformer2D& t : u.transformers) project(t);
    }

    if (idx != n) fail("internal: prime_xattn_cache traversal mismatch");
}

void UNet::compute_step_inputs_(float timestep, const float* gs_emb) {
    if (conv_in_W_.size() == 0) fail("forward: weights not loaded");

    std::vector<float> sin_vals;
    compute_sinusoidal_emb(timestep, freq_dim_, sin_vals);
    freq_emb_ = brodiffusion::detail::upload_host(sin_vals.data(), 1, freq_dim_);

    // LCM cond_proj: add projected guidance-scale embedding to the freq_emb
    // *before* linear_1. Matches diffusers' TimestepEmbedding.forward:
    //   if condition is not None: sample = sample + cond_proj(condition)
    //   sample = linear_1(sample) -> SiLU -> linear_2
    // cond_proj projects (cond_proj_dim) -> (freq_dim), not -> (time_embed_dim).
    if (gs_emb != nullptr) {
        if (cfg_.time_cond_proj_dim <= 0) fail("forward: internal: gs_emb without cond_proj");
        // diffusers' get_guidance_scale_embedding scales w by 1000 before the
        // sinusoidal embedding.
        std::vector<float> w_vals;
        compute_guidance_scale_emb((*gs_emb) * 1000.0f, cfg_.time_cond_proj_dim, w_vals);
        w_emb_ = brodiffusion::detail::upload_host(w_vals.data(), 1, cfg_.time_cond_proj_dim);
        // cond_proj is bias-free: pass nullptr.
        brodiffusion::detail::linear_batched(te_cond_W_, /*bias=*/nullptr, w_emb_, temb_cond_);
        bt::add_inplace(freq_emb_, temb_cond_);
    }

    brodiffusion::detail::linear_batched(te_l1_W_, &te_l1_b_, freq_emb_, temb_a_);
    bt::silu_forward(temb_a_, temb_a_);
    brodiffusion::detail::linear_batched(te_l2_W_, &te_l2_b_, temb_a_, temb_b_);
    // Master temb in temb_b_. Pre-compute SiLU once for reuse across resblocks.
    bt::silu_forward(temb_b_, block_scratch_.temb_silu);
}

void UNet::forward_impl_(const bt::Tensor& sample,
                         int H, int W,
                         float timestep,
                         const float* gs_emb,
                         const bt::Tensor& encoder_hidden_states,
                         const CrossAttnKVCache* xattn_cache,
                         const std::vector<const bt::Tensor*>* attn_logit_biases,
                         CrossAttnTrace* trace_out,
                         const std::vector<const bt::Tensor*>* down_residuals,
                         const bt::Tensor* mid_residual,
                         bt::Tensor& out) {
    compute_step_inputs_(timestep, gs_emb);
    forward_body_impl_(sample, H, W, encoder_hidden_states, xattn_cache,
                       attn_logit_biases, trace_out, down_residuals,
                       mid_residual, out);
}

void UNet::forward_body_impl_(const bt::Tensor& sample,
                              int H, int W,
                              const bt::Tensor& encoder_hidden_states,
                              const CrossAttnKVCache* xattn_cache,
                              const std::vector<const bt::Tensor*>* attn_logit_biases,
                              CrossAttnTrace* trace_out,
                              const std::vector<const bt::Tensor*>* down_residuals,
                              const bt::Tensor* mid_residual,
                              bt::Tensor& out) {
    if (conv_in_W_.size() == 0) fail("forward: weights not loaded");

    const int nb      = static_cast<int>(cfg_.block_out_channels.size());
    const int first_C = cfg_.block_out_channels.front();

    // Validate caller-supplied shapes up front — a mismatch otherwise fails
    // deep inside a brotensor op with an opaque error.
    if (H <= 0 || W <= 0) fail("forward: H and W must be positive");
    const int hw_div = 1 << (nb - 1);
    if (H % hw_div != 0 || W % hw_div != 0) {
        fail("forward: H and W must each be divisible by " +
             std::to_string(hw_div) + " (2^(num_blocks-1))");
    }
    if (sample.rows != 1 || sample.cols != cfg_.in_channels * H * W) {
        fail("forward: sample must be (1, in_channels*H*W)");
    }
    if (encoder_hidden_states.cols != cfg_.cross_attention_dim) {
        fail("forward: encoder_hidden_states width must equal "
             "cross_attention_dim");
    }

    // Total number of skip pushes the down pass will make: 1 from conv_in,
    // plus layers_per_block per stage, plus 1 per downsampler. For the stock
    // SD1.5 config (nb=4, layers_per_block=2) this is 1 + 4*2 + 3 = 12.
    int expected_skips = 1;
    for (int i = 0; i < nb; ++i) {
        expected_skips += cfg_.layers_per_block;
        if (down_blocks_[static_cast<std::size_t>(i)].has_downsampler) {
            ++expected_skips;
        }
    }
    if (down_residuals != nullptr &&
        static_cast<int>(down_residuals->size()) != expected_skips) {
        fail("forward: down_residuals has " +
             std::to_string(down_residuals->size()) +
             " entries, expected " + std::to_string(expected_skips));
    }

    // Skip stack: each entry is a deep copy of the residual stream at push
    // time, stored in the persistent skip_pool_ (entry shapes are constant
    // across steps at fixed (H, W), so after one warm-up forward the pool is
    // pointer-stable — a per-call clone() would allocate mid-capture).
    // Down-path xattn caches advance xattn_idx (kept in scope for mid + up).
    int n_skips = 0;
    int Hc = H, Wc = W;
    int xattn_idx = 0;
    int skip_idx  = 0;
    auto cache_at = [&](int i) -> const CrossAttnKVCacheEntry* {
        return xattn_cache ? &(*xattn_cache)[static_cast<std::size_t>(i)] : nullptr;
    };
    auto trace_at = [&](int i) -> bt::Tensor* {
        return trace_out ? &(*trace_out)[static_cast<std::size_t>(i)] : nullptr;
    };
    auto bias_at = [&](int i) -> const bt::Tensor* {
        return attn_logit_biases ? (*attn_logit_biases)[static_cast<std::size_t>(i)]
                                 : nullptr;
    };
    // Push the current `x_` onto the skip stack (copy into the persistent
    // pool slot), first applying the ControlNet residual for this skip index
    // (if any).
    auto push_skip = [&]() {
        if (static_cast<int>(skip_pool_.size()) <= n_skips) {
            skip_pool_.emplace_back();
        }
        bt::Tensor& skip = skip_pool_[static_cast<std::size_t>(n_skips)];
        brodiffusion::detail::resize_like(skip, x_.rows, x_.cols, x_.dtype,
                                          x_.device);
        bt::copy_d2d(x_, 0, skip, 0, x_.size());
        if (down_residuals != nullptr) {
            const bt::Tensor* res =
                (*down_residuals)[static_cast<std::size_t>(skip_idx)];
            if (res != nullptr) {
                bt::add_inplace(skip, *res);
            }
        }
        ++skip_idx;
        ++n_skips;
    };

    // ── 2. conv_in: in_channels -> first_C ─────────────────────────────────
    brodiffusion::detail::resize_like(x_, sample.rows, sample.cols,
                                      sample.dtype, sample.device);
    bt::copy_d2d(sample, 0, x_, 0, sample.size());
    detail::apply_conv3x3(conv_in_W_, conv_in_b_, cfg_.in_channels, first_C, H, W,
                          /*stride=*/1, /*pad=*/1, x_, y_);
    std::swap(x_, y_);

    push_skip();

    // ── 3. down_blocks ─────────────────────────────────────────────────────
    for (int i = 0; i < nb; ++i) {
        DownBlock& d = down_blocks_[static_cast<std::size_t>(i)];
        for (int j = 0; j < cfg_.layers_per_block; ++j) {
            detail::apply_resnet(d.resnets[static_cast<std::size_t>(j)], Hc, Wc, x_, y_,
                                 block_scratch_, cfg_.norm_num_groups, cfg_.eps);
            if (d.has_attention) {
                const int idx = xattn_idx++;
                detail::apply_transformer(d.transformers[static_cast<std::size_t>(j)],
                                          encoder_hidden_states,
                                          cache_at(idx),
                                          Hc, Wc, x_,
                                          block_scratch_,
                                          cfg_.norm_num_groups, cfg_.eps,
                                          trace_at(idx), bias_at(idx));
            }
            push_skip();
        }
        if (d.has_downsampler) {
            // 3x3 stride-2 conv same-channels.
            if (d.downsampler.W_q.active()) {
                detail::apply_conv3x3_q(d.downsampler.W_q, d.downsampler.b,
                                        d.C_out, d.C_out, Hc, Wc,
                                        /*stride=*/2, /*pad=*/1, x_, y_);
            } else {
                detail::apply_conv3x3(d.downsampler.W, d.downsampler.b,
                                      d.C_out, d.C_out, Hc, Wc,
                                      /*stride=*/2, /*pad=*/1, x_, y_);
            }
            std::swap(x_, y_);
            Hc /= 2;
            Wc /= 2;
            push_skip();
        }
    }

    // ── 4. mid_block: resnet -> transformer -> resnet ──────────────────────
    detail::apply_resnet(mid_.r0, Hc, Wc, x_, y_,
                         block_scratch_, cfg_.norm_num_groups, cfg_.eps);
    {
        const int idx = xattn_idx++;
        detail::apply_transformer(mid_.t, encoder_hidden_states,
                                  cache_at(idx),
                                  Hc, Wc, x_,
                                  block_scratch_,
                                  cfg_.norm_num_groups, cfg_.eps,
                                  trace_at(idx), bias_at(idx));
    }
    detail::apply_resnet(mid_.r1, Hc, Wc, x_, y_,
                         block_scratch_, cfg_.norm_num_groups, cfg_.eps);

    // ControlNet mid residual: added once after the mid block, before the
    // up pass reads x_.
    if (mid_residual != nullptr) {
        bt::add_inplace(x_, *mid_residual);
    }

    // ── 5. up_blocks ───────────────────────────────────────────────────────
    for (int i = 0; i < nb; ++i) {
        UpBlock& u = up_blocks_[static_cast<std::size_t>(i)];
        const int layers = cfg_.layers_per_block + 1;
        for (int j = 0; j < layers; ++j) {
            // Channel-axis concat with popped skip.
            bt::Tensor& skip = skip_pool_[static_cast<std::size_t>(--n_skips)];
            const int C_x_now    = x_.cols / (Hc * Wc);
            const int C_skip_now = skip.cols / (Hc * Wc);
            const std::vector<int> C_parts    = {C_x_now, C_skip_now};
            const std::vector<const bt::Tensor*> parts = {&x_, &skip};
            bt::concat_nchw_channels(parts, /*N=*/1, Hc, Wc, C_parts, cat_buf_);
            std::swap(x_, cat_buf_);

            detail::apply_resnet(u.resnets[static_cast<std::size_t>(j)], Hc, Wc, x_, y_,
                                 block_scratch_, cfg_.norm_num_groups, cfg_.eps);
            if (u.has_attention) {
                const int idx = xattn_idx++;
                detail::apply_transformer(u.transformers[static_cast<std::size_t>(j)],
                                          encoder_hidden_states,
                                          cache_at(idx),
                                          Hc, Wc, x_,
                                          block_scratch_,
                                          cfg_.norm_num_groups, cfg_.eps,
                                          trace_at(idx), bias_at(idx));
            }
        }
        if (u.has_upsampler) {
            bt::upsample_nearest_2x(x_, 1, u.C_out, Hc, Wc, y_);
            if (u.upsampler.W_q.active()) {
                detail::apply_conv3x3_q(u.upsampler.W_q, u.upsampler.b,
                                        u.C_out, u.C_out, 2 * Hc, 2 * Wc,
                                        /*stride=*/1, /*pad=*/1, y_, x_);
            } else {
                detail::apply_conv3x3(u.upsampler.W, u.upsampler.b,
                                      u.C_out, u.C_out, 2 * Hc, 2 * Wc,
                                      /*stride=*/1, /*pad=*/1, y_, x_);
            }
            Hc *= 2;
            Wc *= 2;
        }
    }

    // ── 6. conv_norm_out -> SiLU -> conv_out ───────────────────────────────
    bt::group_norm_forward(x_, norm_out_g_, norm_out_b_,
                               1, first_C, Hc, Wc, cfg_.norm_num_groups, cfg_.eps,
                               y_);
    bt::silu_forward(y_, y_);
    detail::apply_conv3x3(conv_out_W_, conv_out_b_,
                          first_C, cfg_.out_channels, Hc, Wc,
                          /*stride=*/1, /*pad=*/1, y_, out);
}

}  // namespace brodiffusion::unet
