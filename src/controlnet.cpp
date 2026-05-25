#include "brodiffusion/controlnet.h"

#include "brodiffusion/detail/compute.h"
#include "brodiffusion/detail/unet_blocks.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace brodiffusion::controlnet {

namespace bt = ::brotensor;
namespace st = ::brotensor::safetensors;
namespace ud = ::brodiffusion::unet::detail;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("controlnet::ControlNet: " + msg);
}

const st::TensorView& need(const st::File& f, const std::string& key) {
    const auto* v = f.find(key);
    if (!v) fail("missing tensor '" + key + "'");
    return *v;
}

// 3x3 conv2d with bias (matches detail::apply_conv3x3 but exposed at a
// per-stride/per-pad point so the conditioning-embedding stride-2 convs
// share the same path).
void conv3x3(const bt::Tensor& W, const bt::Tensor& b,
             int C_in, int C_out, int H, int W_,
             int stride, int pad,
             const bt::Tensor& in, bt::Tensor& out) {
    bt::conv2d_forward(in, W, &b,
                       /*N=*/1, C_in, H, W_,
                       C_out, /*kH=*/3, /*kW=*/3,
                       stride, stride, pad, pad, /*dil=*/1, 1,
                       out);
}

// 1x1 conv2d with bias — used for the zero-convs.
void conv1x1(const bt::Tensor& W, const bt::Tensor& b,
             int C_in, int C_out, int H, int W_,
             const bt::Tensor& in, bt::Tensor& out) {
    bt::conv2d_forward(in, W, &b,
                       /*N=*/1, C_in, H, W_,
                       C_out, /*kH=*/1, /*kW=*/1,
                       /*sH=*/1, /*sW=*/1, /*pH=*/0, /*pW=*/0,
                       /*dH=*/1, /*dW=*/1, out);
}

// Sinusoidal timestep embedding matching diffusers `get_timestep_embedding`
// with flip_sin_to_cos=True. Identical to UNet's.
void compute_sinusoidal_emb(float t, int dim, std::vector<float>& out) {
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

// ─── ctor / dtor ───────────────────────────────────────────────────────────

ControlNet::ControlNet(const ControlNetConfig& cfg) : cfg_(cfg) {
    const int nb = static_cast<int>(cfg_.block_out_channels.size());
    if (nb < 2) fail("block_out_channels must have at least 2 entries");
    if (cfg_.layers_per_block <= 0) fail("layers_per_block must be positive");
    if (cfg_.transformer_num_heads <= 0) fail("transformer_num_heads must be positive");
    if (cfg_.cross_attention_dim <= 0) fail("cross_attention_dim must be positive");
    if (cfg_.freq_dim <= 0 || (cfg_.freq_dim % 2) != 0) {
        fail("freq_dim must be a positive even integer");
    }
    if (cfg_.time_embed_dim <= 0) fail("time_embed_dim must be positive");
    if (cfg_.conditioning_embedding_channels.empty()) {
        fail("conditioning_embedding_channels must be non-empty");
    }
    for (int c : cfg_.block_out_channels) {
        if (c <= 0 || c % cfg_.norm_num_groups != 0) {
            fail("each block_out_channels entry must be a positive multiple "
                 "of norm_num_groups");
        }
        if (c % cfg_.transformer_num_heads != 0) {
            fail("each block_out_channels entry must be divisible by "
                 "transformer_num_heads");
        }
    }

    // Match UNet's SD1.5 block flag rule: all stages have attention +
    // downsampler EXCEPT the last (final stage has neither).
    down_blocks_.resize(static_cast<std::size_t>(nb));
    for (int i = 0; i < nb; ++i) {
        DownBlock& d = down_blocks_[static_cast<std::size_t>(i)];
        d.has_attention   = (i < nb - 1);
        d.has_downsampler = (i < nb - 1);
        d.C_out = cfg_.block_out_channels[static_cast<std::size_t>(i)];
    }

    // Compute expected skip-push count: 1 (conv_in) + layers_per_block per
    // stage + 1 per downsampler.
    int n_skips = 1;
    for (int i = 0; i < nb; ++i) {
        n_skips += cfg_.layers_per_block;
        if (down_blocks_[static_cast<std::size_t>(i)].has_downsampler) ++n_skips;
    }
    num_down_residuals_ = n_skips;

    // Conditioning embedding: paired (same-channel, stride-2) convs per
    // adjacent-pair step. For HF SD1.5 ControlNet,
    // conditioning_embedding_channels = {16, 32, 96, 256} -> 3 pairs -> 6
    // inner convs (matches diffusers' ControlNetConditioningEmbedding).
    const int n_pairs = static_cast<int>(cfg_.conditioning_embedding_channels.size()) - 1;
    if (n_pairs < 1) fail("conditioning_embedding_channels must have at least 2 entries");
    cond_blocks_.resize(static_cast<std::size_t>(2 * n_pairs));
}

ControlNet::~ControlNet() = default;

// ─── load_weights ──────────────────────────────────────────────────────────

void ControlNet::load_weights(const st::File& f, const std::string& prefix) {
    const int nb       = static_cast<int>(cfg_.block_out_channels.size());
    const int first_C  = cfg_.block_out_channels.front();
    const int mid_C    = cfg_.block_out_channels.back();

    // ── conv_in: in_channels -> first_C ───────────────────────────────────
    st::upload_compute_checked(need(f, prefix + "conv_in.weight"),
                               first_C, cfg_.in_channels * 3 * 3,
                               conv_in_W_, "conv_in.weight");
    st::upload_compute_checked(need(f, prefix + "conv_in.bias"),
                               first_C, 1, conv_in_b_, "conv_in.bias");

    // ── time_embedding: Linear -> SiLU -> Linear ──────────────────────────
    st::upload_compute_checked(need(f, prefix + "time_embedding.linear_1.weight"),
                               cfg_.time_embed_dim, cfg_.freq_dim,
                               te_l1_W_, "te.linear_1.weight");
    st::upload_compute_checked(need(f, prefix + "time_embedding.linear_1.bias"),
                               cfg_.time_embed_dim, 1,
                               te_l1_b_, "te.linear_1.bias");
    st::upload_compute_checked(need(f, prefix + "time_embedding.linear_2.weight"),
                               cfg_.time_embed_dim, cfg_.time_embed_dim,
                               te_l2_W_, "te.linear_2.weight");
    st::upload_compute_checked(need(f, prefix + "time_embedding.linear_2.bias"),
                               cfg_.time_embed_dim, 1,
                               te_l2_b_, "te.linear_2.bias");

    // ── conditioning embedding CNN ────────────────────────────────────────
    //
    // HF layout (lllyasviel/sd-controlnet-*):
    //   controlnet_cond_embedding.conv_in.{weight,bias} : 3 -> 16  (stride 1)
    //   controlnet_cond_embedding.blocks.{2k+0}.*       : Ck -> Ck       (stride 1)
    //   controlnet_cond_embedding.blocks.{2k+1}.*       : Ck -> Ck+1     (stride 2)
    // ... where Ck iterates over conditioning_embedding_channels.
    //   controlnet_cond_embedding.conv_out.{weight,bias}: last -> first_C (stride 1)
    {
        const std::string cp = prefix + "controlnet_cond_embedding.";
        const int first_ce = cfg_.conditioning_embedding_channels.front();
        st::upload_compute_checked(need(f, cp + "conv_in.weight"),
                                   first_ce, cfg_.control_channels * 3 * 3,
                                   cond_conv_in_.W, "cond.conv_in.weight");
        st::upload_compute_checked(need(f, cp + "conv_in.bias"),
                                   first_ce, 1, cond_conv_in_.b,
                                   "cond.conv_in.bias");
        cond_conv_in_.C_in   = cfg_.control_channels;
        cond_conv_in_.C_out  = first_ce;
        cond_conv_in_.stride = 1;

        const int n_pairs = static_cast<int>(cfg_.conditioning_embedding_channels.size()) - 1;
        for (int k = 0; k < n_pairs; ++k) {
            const int Ck   = cfg_.conditioning_embedding_channels[static_cast<std::size_t>(k)];
            const int Cnxt = cfg_.conditioning_embedding_channels[static_cast<std::size_t>(k + 1)];
            // Same-channel stride-1 conv.
            CondConv& a = cond_blocks_[static_cast<std::size_t>(2 * k + 0)];
            const std::string ap = cp + "blocks." + std::to_string(2 * k) + ".";
            st::upload_compute_checked(need(f, ap + "weight"),
                                       Ck, Ck * 3 * 3, a.W,
                                       "cond.block.weight");
            st::upload_compute_checked(need(f, ap + "bias"),
                                       Ck, 1, a.b, "cond.block.bias");
            a.C_in = Ck; a.C_out = Ck; a.stride = 1;

            // Stride-2 channel-increasing conv: Ck -> Cnxt.
            CondConv& s2 = cond_blocks_[static_cast<std::size_t>(2 * k + 1)];
            const std::string sp = cp + "blocks." + std::to_string(2 * k + 1) + ".";
            st::upload_compute_checked(need(f, sp + "weight"),
                                       Cnxt, Ck * 3 * 3, s2.W,
                                       "cond.block.s2.weight");
            st::upload_compute_checked(need(f, sp + "bias"),
                                       Cnxt, 1, s2.b, "cond.block.s2.bias");
            s2.C_in = Ck; s2.C_out = Cnxt; s2.stride = 2;
        }

        const int last_ce = cfg_.conditioning_embedding_channels.back();
        st::upload_compute_checked(need(f, cp + "conv_out.weight"),
                                   first_C, last_ce * 3 * 3,
                                   cond_conv_out_.W, "cond.conv_out.weight");
        st::upload_compute_checked(need(f, cp + "conv_out.bias"),
                                   first_C, 1, cond_conv_out_.b,
                                   "cond.conv_out.bias");
        cond_conv_out_.C_in  = last_ce;
        cond_conv_out_.C_out = first_C;
        cond_conv_out_.stride = 1;
    }

    // ── down_blocks (same prefix layout as UNet) ──────────────────────────
    int C_prev = first_C;
    for (int i = 0; i < nb; ++i) {
        DownBlock& d = down_blocks_[static_cast<std::size_t>(i)];
        const int C_out = d.C_out;
        d.resnets.clear();
        d.resnets.resize(static_cast<std::size_t>(cfg_.layers_per_block));
        d.transformers.clear();
        if (d.has_attention) {
            d.transformers.resize(static_cast<std::size_t>(cfg_.layers_per_block));
        }
        for (int j = 0; j < cfg_.layers_per_block; ++j) {
            const int Ci = (j == 0) ? C_prev : C_out;
            const std::string rp = prefix + "down_blocks." + std::to_string(i) +
                                   ".resnets." + std::to_string(j) + ".";
            ud::load_resnet(f, rp, Ci, C_out, cfg_.time_embed_dim,
                            d.resnets[static_cast<std::size_t>(j)]);
            if (d.has_attention) {
                const std::string tp = prefix + "down_blocks." + std::to_string(i) +
                                       ".attentions." + std::to_string(j) + ".";
                ud::load_transformer(f, tp, C_out, cfg_.transformer_num_heads,
                                     cfg_.cross_attention_dim,
                                     d.transformers[static_cast<std::size_t>(j)]);
            }
        }
        if (d.has_downsampler) {
            const std::string sp = prefix + "down_blocks." + std::to_string(i) +
                                   ".downsamplers.0.conv.";
            st::upload_compute_checked(need(f, sp + "weight"),
                                       C_out, C_out * 3 * 3,
                                       d.downsampler.W,
                                       "downsampler.conv.weight");
            st::upload_compute_checked(need(f, sp + "bias"),
                                       C_out, 1, d.downsampler.b,
                                       "downsampler.conv.bias");
        }
        C_prev = C_out;
    }

    // ── mid_block ─────────────────────────────────────────────────────────
    {
        const std::string mp = prefix + "mid_block.";
        ud::load_resnet(f, mp + "resnets.0.", mid_C, mid_C,
                        cfg_.time_embed_dim, mid_.r0);
        ud::load_transformer(f, mp + "attentions.0.", mid_C,
                             cfg_.transformer_num_heads,
                             cfg_.cross_attention_dim, mid_.t);
        ud::load_resnet(f, mp + "resnets.1.", mid_C, mid_C,
                        cfg_.time_embed_dim, mid_.r1);
    }

    // ── zero-convs ────────────────────────────────────────────────────────
    //
    // Build the per-skip channel sequence by replaying the forward push
    // order — this is what guarantees ControlNet's residuals line up with
    // the UNet's skip stack.
    std::vector<int> skip_channels;
    skip_channels.reserve(static_cast<std::size_t>(num_down_residuals_));
    skip_channels.push_back(first_C);   // conv_in push
    for (int i = 0; i < nb; ++i) {
        const int Cb = cfg_.block_out_channels[static_cast<std::size_t>(i)];
        for (int j = 0; j < cfg_.layers_per_block; ++j) skip_channels.push_back(Cb);
        if (down_blocks_[static_cast<std::size_t>(i)].has_downsampler) {
            skip_channels.push_back(Cb);
        }
    }
    if (static_cast<int>(skip_channels.size()) != num_down_residuals_) {
        fail("internal: skip channel count mismatch");
    }

    down_zero_convs_.resize(static_cast<std::size_t>(num_down_residuals_));
    for (int i = 0; i < num_down_residuals_; ++i) {
        const int C = skip_channels[static_cast<std::size_t>(i)];
        ZeroConv& z = down_zero_convs_[static_cast<std::size_t>(i)];
        const std::string zp = prefix + "controlnet_down_blocks." +
                               std::to_string(i) + ".";
        st::upload_compute_checked(need(f, zp + "weight"), C, C, z.W,
                                   "controlnet_down_blocks.weight");
        st::upload_compute_checked(need(f, zp + "bias"), C, 1, z.b,
                                   "controlnet_down_blocks.bias");
        z.C = C;
    }
    {
        const std::string zp = prefix + "controlnet_mid_block.";
        st::upload_compute_checked(need(f, zp + "weight"), mid_C, mid_C,
                                   mid_zero_conv_.W,
                                   "controlnet_mid_block.weight");
        st::upload_compute_checked(need(f, zp + "bias"), mid_C, 1,
                                   mid_zero_conv_.b,
                                   "controlnet_mid_block.bias");
        mid_zero_conv_.C = mid_C;
    }
}

// ─── forward ───────────────────────────────────────────────────────────────

void ControlNet::forward(const bt::Tensor& sample,
                         int H_lat, int W_lat,
                         float timestep,
                         const bt::Tensor& ctx,
                         const bt::Tensor& control_image,
                         float conditioning_scale,
                         std::vector<bt::Tensor>& down_residuals_out,
                         bt::Tensor& mid_residual_out) {
    if (conv_in_W_.size() == 0) fail("forward: weights not loaded");
    const int nb      = static_cast<int>(cfg_.block_out_channels.size());
    const int first_C = cfg_.block_out_channels.front();

    if (H_lat <= 0 || W_lat <= 0) fail("forward: H_lat and W_lat must be positive");
    const int hw_div = 1 << (nb - 1);
    if (H_lat % hw_div != 0 || W_lat % hw_div != 0) {
        fail("forward: H_lat and W_lat must each be divisible by " +
             std::to_string(hw_div));
    }
    if (sample.rows != 1 || sample.cols != cfg_.in_channels * H_lat * W_lat) {
        fail("forward: sample must be (1, in_channels*H_lat*W_lat)");
    }
    if (ctx.cols != cfg_.cross_attention_dim) {
        fail("forward: ctx width must equal cross_attention_dim");
    }
    const int H_img = H_lat * 8;
    const int W_img = W_lat * 8;
    if (control_image.rows != 1 ||
        control_image.cols != cfg_.control_channels * H_img * W_img) {
        fail("forward: control_image must be (1, control_channels*H_lat*8*W_lat*8)");
    }

    // ── 1. Time embedding (mirrors UNet) ──────────────────────────────────
    std::vector<float> sin_vals;
    compute_sinusoidal_emb(timestep, cfg_.freq_dim, sin_vals);
    freq_emb_ = brodiffusion::detail::upload_host(sin_vals.data(), 1, cfg_.freq_dim);
    brodiffusion::detail::linear_batched(te_l1_W_, &te_l1_b_, freq_emb_, temb_a_);
    bt::silu_forward(temb_a_, temb_a_);
    brodiffusion::detail::linear_batched(te_l2_W_, &te_l2_b_, temb_a_, temb_);
    // Pre-compute SiLU(temb) once for reuse across resblocks.
    bt::silu_forward(temb_, scratch_.temb_silu);

    // ── 2. Conditioning embedding CNN ─────────────────────────────────────
    cond_x_ = control_image.clone();
    int Hc_img = H_img, Wc_img = W_img;
    // conv_in: 3 -> first_ce, stride 1.
    conv3x3(cond_conv_in_.W, cond_conv_in_.b,
            cond_conv_in_.C_in, cond_conv_in_.C_out,
            Hc_img, Wc_img, /*stride=*/1, /*pad=*/1,
            cond_x_, cond_y_);
    std::swap(cond_x_, cond_y_);
    bt::silu_forward(cond_x_, cond_x_);

    // Inner ladder of paired (same-channel, stride-2) convs. With
    // conditioning_embedding_channels of length N, there are 2*(N-1) inner
    // convs. The final stride-2 downsample brings H_img/W_img from H_lat*8
    // down to H_lat*8 / 2^(N-1).
    for (CondConv& cc : cond_blocks_) {
        conv3x3(cc.W, cc.b, cc.C_in, cc.C_out, Hc_img, Wc_img,
                cc.stride, /*pad=*/1, cond_x_, cond_y_);
        std::swap(cond_x_, cond_y_);
        bt::silu_forward(cond_x_, cond_x_);
        if (cc.stride == 2) { Hc_img /= 2; Wc_img /= 2; }
    }

    // conv_out: last_ce -> first_C, stride 1. No activation after.
    conv3x3(cond_conv_out_.W, cond_conv_out_.b,
            cond_conv_out_.C_in, cond_conv_out_.C_out,
            Hc_img, Wc_img, /*stride=*/1, /*pad=*/1,
            cond_x_, cond_y_);
    std::swap(cond_x_, cond_y_);
    // Now cond_x_ should be (1, first_C, H_lat, W_lat).
    if (Hc_img != H_lat || Wc_img != W_lat) {
        fail("forward: conditioning embedding did not land at latent resolution");
    }

    // ── 3. Latent input: conv_in(sample) + cond_emb ───────────────────────
    x_ = sample.clone();
    ud::apply_conv3x3(conv_in_W_, conv_in_b_,
                      cfg_.in_channels, first_C, H_lat, W_lat,
                      /*stride=*/1, /*pad=*/1, x_, y_);
    std::swap(x_, y_);
    bt::add_inplace(x_, cond_x_);

    // ── 4. Encoder pass (mirrors UNet::forward_impl_'s down pass) ─────────
    std::vector<bt::Tensor> skips;
    skips.reserve(static_cast<std::size_t>(num_down_residuals_));
    skips.push_back(x_.clone());  // conv_in push

    int Hc = H_lat, Wc = W_lat;
    for (int i = 0; i < nb; ++i) {
        DownBlock& d = down_blocks_[static_cast<std::size_t>(i)];
        for (int j = 0; j < cfg_.layers_per_block; ++j) {
            ud::apply_resnet(d.resnets[static_cast<std::size_t>(j)],
                             Hc, Wc, x_, y_,
                             scratch_, cfg_.norm_num_groups, cfg_.eps);
            if (d.has_attention) {
                ud::apply_transformer(d.transformers[static_cast<std::size_t>(j)],
                                      ctx, /*cache_entry=*/nullptr,
                                      Hc, Wc, x_,
                                      scratch_, cfg_.norm_num_groups, cfg_.eps);
            }
            skips.push_back(x_.clone());
        }
        if (d.has_downsampler) {
            ud::apply_conv3x3(d.downsampler.W, d.downsampler.b,
                              d.C_out, d.C_out, Hc, Wc,
                              /*stride=*/2, /*pad=*/1, x_, y_);
            std::swap(x_, y_);
            Hc /= 2; Wc /= 2;
            skips.push_back(x_.clone());
        }
    }

    if (static_cast<int>(skips.size()) != num_down_residuals_) {
        fail("internal: skip stack size mismatch in forward");
    }

    // ── 5. Mid pass ───────────────────────────────────────────────────────
    ud::apply_resnet(mid_.r0, Hc, Wc, x_, y_,
                     scratch_, cfg_.norm_num_groups, cfg_.eps);
    ud::apply_transformer(mid_.t, ctx, /*cache_entry=*/nullptr,
                          Hc, Wc, x_,
                          scratch_, cfg_.norm_num_groups, cfg_.eps);
    ud::apply_resnet(mid_.r1, Hc, Wc, x_, y_,
                     scratch_, cfg_.norm_num_groups, cfg_.eps);

    // ── 6. Zero-conv outputs ──────────────────────────────────────────────
    //
    // Replay the per-skip spatial dims to feed conv1x1.
    down_residuals_out.clear();
    down_residuals_out.resize(static_cast<std::size_t>(num_down_residuals_));
    int idx = 0;
    auto emit = [&](int C, int H, int W) {
        ZeroConv& z = down_zero_convs_[static_cast<std::size_t>(idx)];
        bt::Tensor& out = down_residuals_out[static_cast<std::size_t>(idx)];
        conv1x1(z.W, z.b, C, C, H, W, skips[static_cast<std::size_t>(idx)], out);
        if (conditioning_scale != 1.0f) bt::scale_inplace(out, conditioning_scale);
        ++idx;
    };
    int Hs = H_lat, Ws = W_lat;
    emit(first_C, Hs, Ws);  // conv_in push
    for (int i = 0; i < nb; ++i) {
        DownBlock& d = down_blocks_[static_cast<std::size_t>(i)];
        for (int j = 0; j < cfg_.layers_per_block; ++j) {
            emit(d.C_out, Hs, Ws);
        }
        if (d.has_downsampler) {
            Hs /= 2; Ws /= 2;
            emit(d.C_out, Hs, Ws);
        }
    }
    if (idx != num_down_residuals_) fail("internal: emit traversal mismatch");

    conv1x1(mid_zero_conv_.W, mid_zero_conv_.b,
            mid_zero_conv_.C, mid_zero_conv_.C, Hc, Wc,
            x_, mid_residual_out);
    if (conditioning_scale != 1.0f) {
        bt::scale_inplace(mid_residual_out, conditioning_scale);
    }
}

}  // namespace brodiffusion::controlnet
