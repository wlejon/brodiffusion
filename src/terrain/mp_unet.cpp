#include "brodiffusion/terrain/mp_unet.h"

#include "brodiffusion/detail/compute.h"
#include "brodiffusion/detail/json.h"
#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace brodiffusion::terrain {

namespace bt = ::brotensor;
namespace st = ::brotensor::safetensors;

namespace {

constexpr float kMPSiluScale = 1.0f / 0.596f;

// The activation `normalize(x, dim)` of mp_layers.py is
//   x / (1e-4 + sqrt(mean(x^2) over dim))
// i.e. the epsilon sits OUTSIDE the sqrt. brotensor's pixel_norm_forward
// computes x * rsqrt(mean(x^2) + eps) — epsilon INSIDE. Using brotensor's op
// with eps = 1e-4 is a deliberate, measured deviation: this network is
// magnitude-preserving, so the activations it is applied to have RMS ~1, where
// the two forms differ by at most ~5e-5 relative — two orders of magnitude
// inside the 2e-3 parity bar, and worth the fused kernel.
constexpr float kNormEps = 1e-4f;

[[noreturn]] void fail(const std::string& m) {
    throw std::runtime_error("terrain::MPUNet: " + m);
}

const st::TensorView& need(const st::File& f, const std::string& key) {
    if (const auto* v = f.find(key)) return *v;
    fail("missing tensor '" + key + "'");
}

// Read a checkpoint buffer (F32/F16/BF16) into host FP32.
std::vector<float> view_to_host(const st::TensorView& v, const std::string& key) {
    const std::size_t n = static_cast<std::size_t>(v.numel());
    std::vector<float> out(n);
    const auto* raw = v.data;
    switch (v.dtype) {
    case st::Dtype::F32: {
        const auto* src = reinterpret_cast<const float*>(raw);
        for (std::size_t i = 0; i < n; ++i) out[i] = src[i];
        break;
    }
    case st::Dtype::F16: {
        const auto* src = reinterpret_cast<const std::uint16_t*>(raw);
        for (std::size_t i = 0; i < n; ++i) out[i] = bt::fp16_bits_to_fp32(src[i]);
        break;
    }
    case st::Dtype::BF16: {
        const auto* src = reinterpret_cast<const std::uint16_t*>(raw);
        for (std::size_t i = 0; i < n; ++i) out[i] = bt::bf16_bits_to_fp32(src[i]);
        break;
    }
    default:
        fail("tensor '" + key + "' has a non-float dtype");
    }
    return out;
}

bt::Tensor clone(const bt::Tensor& t) {
    bt::Tensor o = bt::Tensor::zeros_on(t.device, t.rows, t.cols, t.dtype);
    bt::copy_d2d(t, 0, o, 0, t.rows * t.cols);
    return o;
}

void mp_silu(bt::Tensor& t) {
    bt::silu_forward(t, t);
    bt::scale_inplace(t, kMPSiluScale);
}

// mp_sum([a, b], w=s): (a*(1-s) + b*s) / sqrt((1-s)^2 + s^2).
void mp_sum2(const bt::Tensor& a, const bt::Tensor& b, float s, bt::Tensor& out) {
    const float w0 = 1.0f - s;
    const float den = std::sqrt(w0 * w0 + s * s);
    out = clone(a);
    bt::axpby_inplace(out, b, w0 / den, s / den);
}

// mp_concat([a, b], dim=channels, w=s).
void mp_concat2(const bt::Tensor& a, int Ca, const bt::Tensor& b, int Cb,
                float s, int N, int H, int W, bt::Tensor& out) {
    const float w0 = 1.0f - s;
    const float C = std::sqrt(static_cast<float>(Ca + Cb) / (w0 * w0 + s * s));
    bt::Tensor ta = clone(a);
    bt::Tensor tb = clone(b);
    bt::scale_inplace(ta, C / std::sqrt(static_cast<float>(Ca)) * w0);
    bt::scale_inplace(tb, C / std::sqrt(static_cast<float>(Cb)) * s);
    const std::vector<const bt::Tensor*> parts{&ta, &tb};
    bt::concat_nchw_channels(parts, N, H, W, {Ca, Cb}, out);
}

// normalize(x, dim=1) on an NCHW activation — see kNormEps.
void pixel_norm_channels(bt::Tensor& x, int N, int C, int H, int W) {
    bt::Tensor seq, nrm;
    bt::nchw_to_sequence(x, N, C, H, W, seq);
    bt::pixel_norm_forward(seq, kNormEps, nrm);
    bt::sequence_to_nchw(nrm, N, C, H, W, x);
}

// y[n,c,h,w] *= c[n,c].
void mul_per_channel(bt::Tensor& y, const bt::Tensor& c, int N, int C, int H, int W) {
    const int L = H * W;
    bt::Tensor seq, prod;
    bt::nchw_to_sequence(y, N, C, H, W, seq);
    if (N == 1) {
        bt::broadcast_mul(seq, c, prod);
    } else {
        prod = bt::Tensor::zeros_on(seq.device, N * L, C, seq.dtype);
        bt::Tensor row = bt::Tensor::zeros_on(c.device, 1, C, c.dtype);
        bt::Tensor blk = bt::Tensor::zeros_on(seq.device, L, C, seq.dtype);
        bt::Tensor sub;
        for (int n = 0; n < N; ++n) {
            bt::copy_d2d(c, n * C, row, 0, C);
            bt::copy_d2d(seq, n * L * C, blk, 0, L * C);
            bt::broadcast_mul(blk, row, sub);
            bt::copy_d2d(sub, 0, prod, n * L * C, L * C);
        }
    }
    bt::sequence_to_nchw(prod, N, C, H, W, y);
}

void conv_same(const bt::Tensor& X, const bt::Tensor& W, int N, int C_in,
               int H, int Wd, int C_out, int k, bt::Tensor& Y) {
    const int pad = k / 2;
    bt::conv2d_forward(X, W, /*bias=*/nullptr, N, C_in, H, Wd, C_out, k, k,
                       /*stride=*/1, 1, pad, pad, /*dil=*/1, 1, /*groups=*/1, Y);
}

}  // namespace

// ─── config ────────────────────────────────────────────────────────────────

MPUNetConfig MPUNetConfig::from_config_json(const std::string& config_path,
                                            const std::string& stage) {
    std::ifstream in(config_path, std::ios::binary);
    if (!in) fail("cannot open " + config_path);
    std::ostringstream ss;
    ss << in.rdbuf();
    const detail::json::Value root = detail::json::parse(ss.str());

    const detail::json::Value* models = root.find("models");
    if (models == nullptr) fail("config.json has no 'models' object");
    const detail::json::Value* m = models->find(stage);
    if (m == nullptr) fail("config.json has no models." + stage);

    MPUNetConfig c;
    c.image_size     = m->get_int("image_size", 0);
    c.in_channels    = m->get_int("in_channels", 0);
    c.out_channels   = m->get_int("out_channels", c.in_channels);
    c.model_channels = m->get_int("model_channels", 0);
    c.model_channel_mults = m->get_int_array("model_channel_mults", {});
    c.layers_per_block    = m->get_int_array("layers_per_block", {});
    c.emb_channels   = m->get_int("emb_channels", 0);
    c.noise_emb_dims = m->get_int("noise_emb_dims", 0);
    c.attn_resolutions = m->get_int_array("attn_resolutions", {});
    c.midblock_attention = m->get_bool("midblock_attention", true);
    c.concat_balance = m->get_float("concat_balance", 0.5f);
    c.res_balance    = m->get_float("res_balance", 0.3f);
    c.attn_balance   = m->get_float("attn_balance", 0.3f);
    c.channels_per_head = m->get_int("channels_per_head", 64);

    if (const detail::json::Value* ca = m->find("clip_act")) {
        c.has_clip_act = !ca->is_null();
        c.clip_act = c.has_clip_act ? static_cast<float>(ca->as_number()) : 0.0f;
    }
    if (const detail::json::Value* fs = m->find("fourier_scale")) {
        c.fourier_scale = fs->is_string() ? fs->as_string()
                                          : std::to_string(fs->as_number());
    }
    if (const detail::json::Value* ci = m->find("conditional_inputs")) {
        for (const auto& e : ci->as_array()) {
            const auto& tup = e.as_array();
            if (tup.size() != 3) fail("conditional_inputs entry is not a 3-tuple");
            CondInput in_;
            in_.kind   = tup[0].as_string();
            in_.dim    = static_cast<int>(tup[1].as_number());
            in_.weight = static_cast<float>(tup[2].as_number());
            c.conditional_inputs.push_back(std::move(in_));
        }
    }
    if (const detail::json::Value* cw = m->find("conditional_weights")) {
        for (const auto& e : cw->as_array()) {
            c.conditional_weights.push_back(static_cast<float>(e.as_number()));
        }
    }
    if (c.layers_per_block.size() != c.model_channel_mults.size()) {
        fail("layers_per_block and model_channel_mults have different lengths");
    }
    return c;
}

// ─── construction ──────────────────────────────────────────────────────────

MPUNet::MPUNet(const MPUNetConfig& cfg) : cfg_(cfg) { build_modules_(); }
MPUNet::~MPUNet() = default;

void MPUNet::build_modules_() {
    const int L = static_cast<int>(cfg_.model_channel_mults.size());
    const bool has_emb = cfg_.emb_channels > 0;

    auto has_attn_at = [&](int res) {
        for (int r : cfg_.attn_resolutions) if (r == res) return true;
        return false;
    };
    auto heads_for = [&](int out_ch, bool attention) {
        return attention ? out_ch / cfg_.channels_per_head : 0;
    };
    auto res_name = [](int res) {
        return std::to_string(res) + "x" + std::to_string(res);
    };

    // Encoder. cout starts at in_channels+1 for the appended ones channel.
    int cout = cfg_.in_channels + 1;
    for (int level = 0; level < L; ++level) {
        const int ch  = cfg_.model_channels * cfg_.model_channel_mults[level];
        const int nb  = cfg_.layers_per_block[level];
        const int res = cfg_.image_size >> level;
        if (level == 0) {
            Block b;
            b.key = "enc." + res_name(res) + "_conv";
            b.in_ch = cout;
            b.out_ch = ch;
            b.plain_conv = true;
            enc_.push_back(std::move(b));
            cout = ch;
        } else {
            Block b;
            b.key = "enc." + res_name(res) + "_down";
            b.in_ch = cout;
            b.out_ch = cout;
            b.resample = Resample::Down;
            b.has_emb = has_emb;
            enc_.push_back(std::move(b));
        }
        for (int idx = 0; idx < nb; ++idx) {
            Block b;
            b.key = "enc." + res_name(res) + "_block" + std::to_string(idx);
            b.in_ch = cout;
            b.out_ch = ch;
            b.has_skip = (cout != ch);
            b.has_emb = has_emb;
            b.num_heads = heads_for(ch, has_attn_at(res));
            enc_.push_back(std::move(b));
            cout = ch;
        }
    }

    // Decoder. `skips` is the encoder block output widths, popped from the back.
    std::vector<int> skips;
    for (const Block& b : enc_) skips.push_back(b.out_ch);

    for (int level = L - 1; level >= 0; --level) {
        const int ch  = cfg_.model_channels * cfg_.model_channel_mults[level];
        const int nb  = cfg_.layers_per_block[level];
        const int res = cfg_.image_size >> level;
        if (level == L - 1) {
            Block b0;
            b0.key = "dec." + res_name(res) + "_in0";
            b0.in_ch = cout; b0.out_ch = cout;
            b0.dec_mode = true;
            b0.has_emb = has_emb;
            b0.num_heads = heads_for(cout, cfg_.midblock_attention);
            dec_.push_back(std::move(b0));

            Block b1;
            b1.key = "dec." + res_name(res) + "_in1";
            b1.in_ch = cout; b1.out_ch = cout;
            b1.dec_mode = true;
            b1.has_emb = has_emb;
            dec_.push_back(std::move(b1));
        } else {
            Block b;
            b.key = "dec." + res_name(res) + "_up";
            b.in_ch = cout; b.out_ch = cout;
            b.dec_mode = true;
            b.resample = Resample::Up;
            b.has_emb = has_emb;
            dec_.push_back(std::move(b));
        }
        for (int idx = 0; idx <= nb; ++idx) {   // nb + 1 blocks
            if (skips.empty()) fail("decoder ran out of encoder skips");
            const int cin = cout + skips.back();
            skips.pop_back();
            Block b;
            b.key = "dec." + res_name(res) + "_block" + std::to_string(idx);
            b.in_ch = cin;
            b.out_ch = ch;
            b.dec_mode = true;
            b.has_skip = (cin != ch);
            b.has_emb = has_emb;
            b.num_heads = heads_for(ch, has_attn_at(res));
            dec_.push_back(std::move(b));
            cout = ch;
        }
    }
    if (!skips.empty()) fail("encoder skips left unconsumed by the decoder");
}

// ─── loading ───────────────────────────────────────────────────────────────

void MPUNet::load_weights(const st::File& f) {
    auto load_conv = [&](const std::string& key, int C_out, int C_in, int k,
                         bt::Tensor& dst) {
        st::upload_compute_checked(need(f, key), C_out, C_in * k * k, dst, key);
    };

    for (Block& b : enc_) {
        if (b.plain_conv) {
            load_conv(b.key + ".weight", b.out_ch, b.in_ch, 3, b.conv_res0);
            continue;
        }
        // mode 'enc': conv_res0 sees out_channels (conv_skip runs first).
        load_conv(b.key + ".conv_res0.weight", b.out_ch, b.out_ch, 3, b.conv_res0);
        load_conv(b.key + ".conv_res1.weight", b.out_ch, b.out_ch, 3, b.conv_res1);
        if (b.has_skip) load_conv(b.key + ".conv_skip.weight", b.out_ch, b.in_ch, 1, b.conv_skip);
        if (b.has_emb) {
            const std::string k = b.key + ".emb_linear.weight";
            st::upload_compute_checked(need(f, k), b.out_ch, cfg_.emb_channels,
                                       b.emb_linear, k);
        }
        if (b.num_heads != 0) fail("encoder attention is unused by the shipped configs");
    }

    for (Block& b : dec_) {
        // mode 'dec': conv_res0 sees in_channels.
        load_conv(b.key + ".conv_res0.weight", b.out_ch, b.in_ch, 3, b.conv_res0);
        load_conv(b.key + ".conv_res1.weight", b.out_ch, b.out_ch, 3, b.conv_res1);
        if (b.has_skip) load_conv(b.key + ".conv_skip.weight", b.out_ch, b.in_ch, 1, b.conv_skip);
        if (b.has_emb) {
            const std::string k = b.key + ".emb_linear.weight";
            st::upload_compute_checked(need(f, k), b.out_ch, cfg_.emb_channels,
                                       b.emb_linear, k);
        }
        if (b.num_heads != 0) {
            const int C = b.out_ch, d = cfg_.channels_per_head, H = b.num_heads;
            // attn_qkv emits 3C channels laid out as [head][c_head][qkv] — q/k/v
            // are INTERLEAVED with stride 3 in the innermost position, not three
            // contiguous blocks. Permute the 1x1 conv's output rows once here so
            // the runtime sees the far friendlier [qkv][head][c_head] order; for
            // a 1x1 conv a row permutation is exactly an output-channel permutation.
            const std::string k = b.key + ".attn_qkv.weight";
            const st::TensorView& v = need(f, k);
            if (v.numel() != static_cast<std::int64_t>(3) * C * C) {
                fail("'" + k + "' has an unexpected element count");
            }
            std::vector<float> src = view_to_host(v, k);
            std::vector<float> dst(src.size());
            for (int h = 0; h < H; ++h) {
                for (int c = 0; c < d; ++c) {
                    for (int j = 0; j < 3; ++j) {
                        const int from = h * d * 3 + c * 3 + j;
                        const int to   = j * C + h * d + c;
                        std::copy(src.begin() + static_cast<std::ptrdiff_t>(from) * C,
                                  src.begin() + static_cast<std::ptrdiff_t>(from + 1) * C,
                                  dst.begin() + static_cast<std::ptrdiff_t>(to) * C);
                    }
                }
            }
            b.attn_qkv = detail::upload_host(dst.data(), 3 * C, C);
            load_conv(b.key + ".attn_proj.weight", C, C, 1, b.attn_proj);
        }
    }

    {
        const std::string k = "out_conv.weight";
        // out_gain is already folded into these weights by the converter.
        st::upload_compute_checked(need(f, k), cfg_.out_channels,
                                   dec_.back().out_ch * 9, out_conv_, k);
    }

    // Noise embedding.
    if (cfg_.noise_emb_dims > 0) {
        has_noise_ = true;
        const bool positional = (cfg_.fourier_scale == "pos");
        const std::string fk = "noise_fourier.freqs";
        noise_freqs_ = view_to_host(need(f, fk), fk);
        const std::size_t want = positional
            ? static_cast<std::size_t>(cfg_.noise_emb_dims / 2)
            : static_cast<std::size_t>(cfg_.noise_emb_dims);
        if (noise_freqs_.size() != want) fail("'" + fk + "' has an unexpected length");
        if (!positional) {
            const std::string pk = "noise_fourier.phases";
            noise_phases_ = view_to_host(need(f, pk), pk);
        }
        const std::string lk = "noise_linear.weight";
        st::upload_compute_checked(need(f, lk), cfg_.emb_channels,
                                   cfg_.noise_emb_dims, noise_linear_, lk);
    }

    // Conditional branches.
    for (std::size_t i = 0; i < cfg_.conditional_inputs.size(); ++i) {
        const MPUNetConfig::CondInput& ci = cfg_.conditional_inputs[i];
        const std::string base = "conditional_layers." + std::to_string(i);
        CondLayer cl;
        if (ci.kind == "tensor") {
            cl.is_tensor = true;
            const std::string k = base + ".weight";
            st::upload_compute_checked(need(f, k), cfg_.emb_channels, ci.dim, cl.W, k);
        } else if (ci.kind == "float") {
            // nn.Sequential(MPFourier(dim), MPConv(dim, emb, kernel=[])).
            const std::string fk = base + ".0.freqs";
            const std::string pk = base + ".0.phases";
            const std::string wk = base + ".1.weight";
            cl.freqs  = view_to_host(need(f, fk), fk);
            cl.phases = view_to_host(need(f, pk), pk);
            st::upload_compute_checked(need(f, wk), cfg_.emb_channels, ci.dim, cl.W, wk);
        } else {
            fail("unsupported conditional input kind '" + ci.kind + "'");
        }
        cond_layers_.push_back(std::move(cl));
    }
}

const bt::Tensor& MPUNet::ones_filter_(int C) {
    auto it = ones_filters_.find(C);
    if (it != ones_filters_.end()) return it->second;
    std::vector<float> ones(static_cast<std::size_t>(C), 1.0f);
    bt::Tensor t = detail::upload_host(ones.data(), C, 1);
    return ones_filters_.emplace(C, std::move(t)).first->second;
}

// ─── forward ───────────────────────────────────────────────────────────────

void MPUNet::attention_(const Block& b, bt::Tensor& x, int N, int C, int S) {
    const int L = S * S;
    const int heads = b.num_heads;
    const int d = C / heads;

    bt::Tensor qkv;
    conv_same(x, b.attn_qkv, N, C, S, S, 3 * C, 1, qkv);       // (N, 3C*L)

    // (N*L, 3C) with columns qkv*C + head*d + c.
    bt::Tensor seq;
    bt::nchw_to_sequence(qkv, N, 3 * C, S, S, seq);

    // normalize(y, dim=2) — over the c_head axis. Row-major, that axis is the
    // innermost run of `d` elements, so the whole (N*L, 3C) buffer reinterpreted
    // as (N*L*3*heads, d) is exactly the set of vectors to normalise.
    bt::Tensor flat = bt::Tensor::zeros_on(seq.device, N * L * 3 * heads, d, seq.dtype);
    bt::copy_d2d(seq, 0, flat, 0, N * L * 3 * C);
    bt::Tensor nrm;
    bt::pixel_norm_forward(flat, kNormEps, nrm);

    // The same normalised buffer in sequence and NCHW layouts: q/v are read as
    // (L, d) slices of the former, k^T as a contiguous (d, L) slice of the latter.
    bt::Tensor seqn = bt::Tensor::zeros_on(nrm.device, N * L, 3 * C, nrm.dtype);
    bt::copy_d2d(nrm, 0, seqn, 0, N * L * 3 * C);
    bt::Tensor nchw;
    bt::sequence_to_nchw(seqn, N, 3 * C, S, S, nchw);

    bt::Tensor yout = bt::Tensor::zeros_on(x.device, N, C * L, x.dtype);
    bt::Tensor q  = bt::Tensor::zeros_on(x.device, L, d, x.dtype);
    bt::Tensor v  = bt::Tensor::zeros_on(x.device, L, d, x.dtype);
    bt::Tensor kT = bt::Tensor::zeros_on(x.device, d, L, x.dtype);
    bt::Tensor w, yh, yh_nchw;
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));

    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < heads; ++h) {
            bt::copy_d2d_strided(seqn, n * L * 3 * C + h * d, 3 * C,
                                 q, 0, d, /*width=*/d, /*height=*/L);
            bt::copy_d2d_strided(seqn, n * L * 3 * C + 2 * C + h * d, 3 * C,
                                 v, 0, d, d, L);
            bt::copy_d2d(nchw, n * 3 * C * L + (C + h * d) * L, kT, 0, d * L);
            bt::scale_inplace(kT, scale);

            bt::matmul(q, kT, w);                       // (L, L) scores
            bt::softmax_rows_forward(w, w, L, L);
            bt::matmul(w, v, yh);                       // (L, d)
            bt::sequence_to_nchw(yh, 1, d, S, S, yh_nchw);
            bt::copy_d2d(yh_nchw, 0, yout, n * C * L + h * d * L, d * L);
        }
    }

    conv_same(yout, b.attn_proj, N, C, S, S, C, 1, x);
}

void MPUNet::run_block_(const Block& b, bt::Tensor& x, const bt::Tensor& emb,
                        bool have_emb, int N, int& C, int& S) {
    if (b.plain_conv) {
        bt::Tensor y;
        conv_same(x, b.conv_res0, N, b.in_ch, S, S, b.out_ch, 3, y);
        x = std::move(y);
        C = b.out_ch;
        return;
    }

    // resample
    if (b.resample == Resample::Down) {
        // Upstream: conv2d with an all-ones depthwise 1x1 kernel at stride 2 —
        // a pure x[:, :, ::2, ::2] subsample.
        bt::Tensor y;
        bt::conv2d_forward(x, ones_filter_(C), nullptr, N, C, S, S, C, 1, 1,
                           /*stride=*/2, 2, /*pad=*/0, 0, /*dil=*/1, 1,
                           /*groups=*/C, y);
        x = std::move(y);
        S /= 2;
    } else if (b.resample == Resample::Up) {
        // Upstream: conv_transpose2d with an all-ones depthwise 2x2 kernel at
        // stride 2 — identical to a nearest-neighbour 2x repeat.
        bt::Tensor y;
        bt::upsample_nearest_2x(x, N, C, S, S, y);
        x = std::move(y);
        S *= 2;
    }

    if (!b.dec_mode) {
        if (b.has_skip) {
            bt::Tensor y;
            conv_same(x, b.conv_skip, N, C, S, S, b.out_ch, 1, y);
            x = std::move(y);
            C = b.out_ch;
        }
        pixel_norm_channels(x, N, C, S, S);
    }

    // Residual branch.
    bt::Tensor act = clone(x);
    mp_silu(act);
    bt::Tensor y;
    conv_same(act, b.conv_res0, N, C, S, S, b.out_ch, 3, y);

    if (b.has_emb && have_emb) {
        bt::Tensor c;
        detail::linear_batched(b.emb_linear, /*bias=*/nullptr, emb, c);  // (N, out_ch)
        bt::add_scalar_inplace(c, 1.0f);
        // c / sqrt(mean(c^2, dim=1) + 1e-8) — epsilon inside the sqrt, which is
        // exactly brotensor's pixel_norm.
        bt::Tensor cn;
        bt::pixel_norm_forward(c, 1e-8f, cn);
        mul_per_channel(y, cn, N, b.out_ch, S, S);
    }
    mp_silu(y);
    bt::Tensor y1;
    conv_same(y, b.conv_res1, N, b.out_ch, S, S, b.out_ch, 3, y1);

    if (b.dec_mode && b.has_skip) {
        bt::Tensor xs;
        conv_same(x, b.conv_skip, N, C, S, S, b.out_ch, 1, xs);
        x = std::move(xs);
        C = b.out_ch;
    }

    bt::Tensor summed;
    mp_sum2(x, y1, cfg_.res_balance, summed);
    x = std::move(summed);
    C = b.out_ch;

    if (b.num_heads != 0) {
        bt::Tensor a = clone(x);
        attention_(b, a, N, C, S);
        bt::Tensor blended;
        mp_sum2(x, a, cfg_.attn_balance, blended);
        x = std::move(blended);
    }

    if (cfg_.has_clip_act) bt::clamp(x, -cfg_.clip_act, cfg_.clip_act);
}

void MPUNet::forward(const bt::Tensor& x_in, int N, int S,
                     const float* noise_labels,
                     const std::vector<std::vector<float>>& cond,
                     bt::Tensor& out) {
    if (enc_.empty()) fail("forward: weights not loaded");
    if (cond.size() != cond_layers_.size()) {
        fail("forward: expected " + std::to_string(cond_layers_.size()) +
             " conditional inputs, got " + std::to_string(cond.size()));
    }

    // ── embeddings ──
    std::vector<bt::Tensor> embeds;
    std::vector<float> weights;
    const float sqrt2 = std::sqrt(2.0f);

    if (has_noise_) {
        const bool positional = (cfg_.fourier_scale == "pos");
        const int D = cfg_.noise_emb_dims;
        std::vector<float> host(static_cast<std::size_t>(N) * D);
        if (positional) {
            // MPPositionalEmbedding: cat([sin(x*f), cos(x*f)]) * sqrt(2).
            const int half = D / 2;
            for (int n = 0; n < N; ++n) {
                for (int k = 0; k < half; ++k) {
                    const float a = noise_labels[n] * noise_freqs_[k];
                    host[static_cast<std::size_t>(n) * D + k]        = std::sin(a) * sqrt2;
                    host[static_cast<std::size_t>(n) * D + half + k] = std::cos(a) * sqrt2;
                }
            }
        } else {
            for (int n = 0; n < N; ++n) {
                for (int k = 0; k < D; ++k) {
                    host[static_cast<std::size_t>(n) * D + k] =
                        std::cos(noise_labels[n] * noise_freqs_[k] + noise_phases_[k]) * sqrt2;
                }
            }
        }
        bt::Tensor fin = detail::upload_host(host.data(), N, D);
        bt::Tensor e;
        detail::linear_batched(noise_linear_, nullptr, fin, e);
        embeds.push_back(std::move(e));
        weights.push_back(cfg_.conditional_weights.empty() ? 1.0f
                                                           : cfg_.conditional_weights[0]);
    }

    for (std::size_t i = 0; i < cond_layers_.size(); ++i) {
        const CondLayer& cl = cond_layers_[i];
        const int dim = cfg_.conditional_inputs[i].dim;
        bt::Tensor e;
        if (cl.is_tensor) {
            if (cond[i].size() != static_cast<std::size_t>(N) * dim) {
                fail("forward: conditional input " + std::to_string(i) +
                     " should carry N*" + std::to_string(dim) + " values");
            }
            bt::Tensor in = detail::upload_host(cond[i].data(), N, dim);
            detail::linear_batched(cl.W, nullptr, in, e);
            mp_silu(e);   // 'tensor' inputs get mp_silu here; 'float' ones do not.
        } else {
            if (cond[i].size() != static_cast<std::size_t>(N)) {
                fail("forward: conditional input " + std::to_string(i) +
                     " should carry N scalar values");
            }
            // MPFourier: cos(x*freqs + phases) * sqrt(2).
            std::vector<float> host(static_cast<std::size_t>(N) * dim);
            for (int n = 0; n < N; ++n) {
                for (int k = 0; k < dim; ++k) {
                    host[static_cast<std::size_t>(n) * dim + k] =
                        std::cos(cond[i][n] * cl.freqs[k] + cl.phases[k]) * sqrt2;
                }
            }
            bt::Tensor in = detail::upload_host(host.data(), N, dim);
            detail::linear_batched(cl.W, nullptr, in, e);
        }
        embeds.push_back(std::move(e));
        const std::size_t widx = (has_noise_ ? 1u : 0u) + i;
        weights.push_back(widx < cfg_.conditional_weights.size()
                              ? cfg_.conditional_weights[widx]
                              : cfg_.conditional_inputs[i].weight);
    }

    bt::Tensor emb;
    const bool have_emb = !embeds.empty();
    if (have_emb) {
        // mp_sum(embeds, w) = sum_i e_i*w_i / ||w||.
        float nrm = 0.0f;
        for (float w : weights) nrm += w * w;
        nrm = std::sqrt(nrm);
        emb = bt::Tensor::zeros_on(embeds[0].device, embeds[0].rows,
                                   embeds[0].cols, embeds[0].dtype);
        for (std::size_t i = 0; i < embeds.size(); ++i) {
            bt::axpby_inplace(emb, embeds[i], 1.0f, weights[i] / nrm);
        }
        mp_silu(emb);
    }

    // ── encoder ──
    // Append one all-ones channel to stand in for the bias.
    bt::Tensor x;
    {
        std::vector<float> ones(static_cast<std::size_t>(N) * S * S, 1.0f);
        bt::Tensor ones_t = detail::upload_host(ones.data(), N, S * S);
        const std::vector<const bt::Tensor*> parts{&x_in, &ones_t};
        bt::concat_nchw_channels(parts, N, S, S, {cfg_.in_channels, 1}, x);
    }
    int C = cfg_.in_channels + 1;
    int Sz = S;

    struct Skip { bt::Tensor t; int C; int S; };
    std::vector<Skip> skips;
    skips.reserve(enc_.size());
    for (const Block& b : enc_) {
        run_block_(b, x, emb, have_emb, N, C, Sz);
        skips.push_back(Skip{clone(x), C, Sz});
    }

    // ── decoder ──
    for (const Block& b : dec_) {
        if (b.key.find("_block") != std::string::npos) {
            if (skips.empty()) fail("forward: skip stack underflow");
            const Skip& s = skips.back();
            bt::Tensor merged;
            mp_concat2(x, C, s.t, s.C, cfg_.concat_balance, N, Sz, Sz, merged);
            C += s.C;
            x = std::move(merged);
            skips.pop_back();
        }
        run_block_(b, x, emb, have_emb, N, C, Sz);
    }

    conv_same(x, out_conv_, N, C, Sz, Sz, cfg_.out_channels, 3, out);
}

}  // namespace brodiffusion::terrain
