// Flux DiT denoiser smoke test.
//
// Builds a tiny FluxTransformer2DModel (in_channels=16, num_layers=2,
// num_single_layers=2, attention_head_dim=8, num_attention_heads=2,
// joint_attention_dim=10, pooled_projection_dim=6, axes_dims_rope={2,2,4}),
// synthesizes a safetensors fixture with the HF tensor names, loads it via
// FluxDenoiser::load_weights, prepares conditioning, and runs forward.
// Verifies output shape, that all outputs are finite, and that two
// consecutive forwards produce identical results. Also exercises the
// guidance_embeds=true (flux-dev) variant.
//
// Numerical accuracy against a reference is intentionally not checked here —
// that needs real Flux weights and lives in image-generation integration.

#include "brodiffusion/dit/flux.h"
#include "brodiffusion/denoiser.h"
#include "brodiffusion/detail/compute.h"
#include "brotensor/safetensors.h"

#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include "test_compute.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace dit = brodiffusion::dit;
namespace st  = brotensor::safetensors;
namespace bt  = brotensor;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

// ─── safetensors fixture builder ───────────────────────────────────────────

namespace {

struct Builder {
    std::string entries;
    std::vector<uint8_t> payload;
    bool first = true;

    void add(const std::string& name, std::vector<int> shape,
             const std::vector<uint16_t>& fp16_bits) {
        std::size_t expected = 1;
        for (int d : shape) expected *= static_cast<std::size_t>(d);
        if (expected != fp16_bits.size()) {
            std::fprintf(stderr, "fixture: shape/data mismatch for %s\n",
                         name.c_str());
            std::abort();
        }
        std::uint64_t start = payload.size();
        const std::uint8_t* bytes =
            reinterpret_cast<const std::uint8_t*>(fp16_bits.data());
        payload.insert(payload.end(), bytes, bytes + fp16_bits.size() * 2);
        std::uint64_t end = payload.size();

        if (!first) entries += ",";
        first = false;
        entries += "\"" + name + "\":{\"dtype\":\"F16\",\"shape\":[";
        for (std::size_t i = 0; i < shape.size(); ++i) {
            if (i) entries += ",";
            entries += std::to_string(shape[i]);
        }
        entries += "],\"data_offsets\":[" + std::to_string(start) + "," +
                   std::to_string(end) + "]}";
    }

    void write(const std::filesystem::path& path) const {
        std::string header = "{" + entries + "}";
        std::uint64_t hdr_size = header.size();
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f) std::abort();
        f.write(reinterpret_cast<const char*>(&hdr_size), 8);
        f.write(header.data(), header.size());
        f.write(reinterpret_cast<const char*>(payload.data()),
                static_cast<std::streamsize>(payload.size()));
    }
};

std::vector<uint16_t> fp16_ones(std::size_t n) {
    return std::vector<uint16_t>(n, bt::fp32_to_fp16_bits(1.0f));
}
// Small deterministic values bounded around zero so the synthetic forward
// stays numerically tame: a wrapped sawtooth scaled small.
std::vector<uint16_t> fp16_seq(std::size_t n, float scale) {
    std::vector<uint16_t> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        float v = (static_cast<float>(i % 7) - 3.0f) * scale;
        out[i] = bt::fp32_to_fp16_bits(v);
    }
    return out;
}

// Emit a biased Linear (weight (out,in) + bias (out,)).
void add_linear(Builder& b, const std::string& key, int out, int in,
                float wscale, float bscale) {
    b.add(key + ".weight", {out, in},
          fp16_seq(static_cast<std::size_t>(out) * in, wscale));
    b.add(key + ".bias", {out}, fp16_seq(static_cast<std::size_t>(out), bscale));
}

// Build the full tiny-Flux fixture for `cfg`.
void build_fixture(Builder& b, const dit::FluxConfig& cfg) {
    const int D  = cfg.inner_dim();
    const int IC = cfg.in_channels;
    const int JD = cfg.joint_attention_dim;
    const int PD = cfg.pooled_projection_dim;
    const int HD = cfg.attention_head_dim;

    add_linear(b, "x_embedder", D, IC, 0.05f, 0.01f);
    add_linear(b, "context_embedder", D, JD, 0.05f, 0.01f);

    const std::string tt = "time_text_embed.";
    add_linear(b, tt + "timestep_embedder.linear_1", D, 256, 0.02f, 0.01f);
    add_linear(b, tt + "timestep_embedder.linear_2", D, D, 0.05f, 0.01f);
    add_linear(b, tt + "text_embedder.linear_1", D, PD, 0.05f, 0.01f);
    add_linear(b, tt + "text_embedder.linear_2", D, D, 0.05f, 0.01f);
    if (cfg.guidance_embeds) {
        add_linear(b, tt + "guidance_embedder.linear_1", D, 256, 0.02f, 0.01f);
        add_linear(b, tt + "guidance_embedder.linear_2", D, D, 0.05f, 0.01f);
    }

    for (int i = 0; i < cfg.num_layers; ++i) {
        const std::string p =
            "transformer_blocks." + std::to_string(i) + ".";
        add_linear(b, p + "norm1.linear", 6 * D, D, 0.03f, 0.01f);
        add_linear(b, p + "norm1_context.linear", 6 * D, D, 0.03f, 0.01f);
        add_linear(b, p + "attn.to_q", D, D, 0.05f, 0.01f);
        add_linear(b, p + "attn.to_k", D, D, 0.05f, 0.01f);
        add_linear(b, p + "attn.to_v", D, D, 0.05f, 0.01f);
        add_linear(b, p + "attn.add_q_proj", D, D, 0.05f, 0.01f);
        add_linear(b, p + "attn.add_k_proj", D, D, 0.05f, 0.01f);
        add_linear(b, p + "attn.add_v_proj", D, D, 0.05f, 0.01f);
        add_linear(b, p + "attn.to_out.0", D, D, 0.05f, 0.01f);
        add_linear(b, p + "attn.to_add_out", D, D, 0.05f, 0.01f);
        b.add(p + "attn.norm_q.weight", {HD}, fp16_ones(HD));
        b.add(p + "attn.norm_k.weight", {HD}, fp16_ones(HD));
        b.add(p + "attn.norm_added_q.weight", {HD}, fp16_ones(HD));
        b.add(p + "attn.norm_added_k.weight", {HD}, fp16_ones(HD));
        add_linear(b, p + "ff.net.0.proj", 4 * D, D, 0.03f, 0.01f);
        add_linear(b, p + "ff.net.2", D, 4 * D, 0.03f, 0.01f);
        add_linear(b, p + "ff_context.net.0.proj", 4 * D, D, 0.03f, 0.01f);
        add_linear(b, p + "ff_context.net.2", D, 4 * D, 0.03f, 0.01f);
    }

    for (int i = 0; i < cfg.num_single_layers; ++i) {
        const std::string p =
            "single_transformer_blocks." + std::to_string(i) + ".";
        add_linear(b, p + "norm.linear", 3 * D, D, 0.03f, 0.01f);
        add_linear(b, p + "attn.to_q", D, D, 0.05f, 0.01f);
        add_linear(b, p + "attn.to_k", D, D, 0.05f, 0.01f);
        add_linear(b, p + "attn.to_v", D, D, 0.05f, 0.01f);
        b.add(p + "attn.norm_q.weight", {HD}, fp16_ones(HD));
        b.add(p + "attn.norm_k.weight", {HD}, fp16_ones(HD));
        add_linear(b, p + "proj_mlp", 4 * D, D, 0.03f, 0.01f);
        add_linear(b, p + "proj_out", D, 5 * D, 0.03f, 0.01f);
    }

    add_linear(b, "norm_out.linear", 2 * D, D, 0.03f, 0.01f);
    add_linear(b, "proj_out", IC, D, 0.05f, 0.01f);
}

// Run a tiny-Flux config end to end; returns the forward output (host FP32).
std::vector<float> run_flux(const dit::FluxConfig& cfg,
                            const std::filesystem::path& path) {
    Builder b;
    build_fixture(b, cfg);
    b.write(path);

    auto file = st::File::open(path.string());
    dit::FluxDenoiser flux(cfg);
    flux.load_weights(file, "");
    flux.finalize_weights();
    flux.finalize_weights();  // idempotent

    CHECK(flux.prediction_type() == brodiffusion::PredictionType::Velocity);
    CHECK(flux.uses_cfg() == false);
    CHECK(flux.latent_channels() == cfg.in_channels / 4);
    CHECK(flux.as_unet() == nullptr);
    CHECK(flux.compute_dtype() == brodiffusion::compute_dtype());

    // Conditioning: synthetic T5 context (txt_len=3, joint_attention_dim),
    // pooled CLIP (1, pooled_projection_dim).
    const int txt_len = 3;
    std::vector<float> ctx(static_cast<std::size_t>(txt_len) *
                           cfg.joint_attention_dim);
    for (std::size_t i = 0; i < ctx.size(); ++i) {
        ctx[i] = (static_cast<float>(i % 5) - 2.0f) * 0.1f;
    }
    std::vector<float> pooled(static_cast<std::size_t>(cfg.pooled_projection_dim));
    for (std::size_t i = 0; i < pooled.size(); ++i) {
        pooled[i] = (static_cast<float>(i % 3) - 1.0f) * 0.2f;
    }

    brodiffusion::Conditioning cond;
    cond.text_embeddings =
        bdtest::bd_upload(ctx, txt_len, cfg.joint_attention_dim);
    cond.pooled = bdtest::bd_upload(pooled, 1, cfg.pooled_projection_dim);
    cond.guidance = cfg.guidance_embeds ? 3.5f : 0.0f;
    cond.has_uncond = false;

    auto prepared = flux.prepare(cond);

    // Latent: (1, latent_channels * H_lat * W_lat), H_lat=W_lat=4.
    const int H_lat = 4, W_lat = 4;
    const int LC = cfg.in_channels / 4;
    std::vector<float> lat(
        static_cast<std::size_t>(LC) * H_lat * W_lat);
    for (std::size_t i = 0; i < lat.size(); ++i) {
        lat[i] = (static_cast<float>(i % 9) - 4.0f) * 0.15f;
    }
    bt::Tensor latent = bdtest::bd_upload(lat, 1, static_cast<int>(lat.size()));

    bt::Tensor out;
    flux.forward(latent, H_lat, W_lat, /*timestep=*/500.0f, prepared,
                 brodiffusion::Branch::Cond, out);
    bt::sync_all();

    CHECK(out.rows == 1);
    CHECK(out.cols == LC * H_lat * W_lat);
    CHECK(out.dtype == brodiffusion::compute_dtype());

    std::vector<float> v1 = bdtest::bd_download(out);
    int nonfinite = 0;
    for (float v : v1) if (!bdtest::bd_finite(v)) ++nonfinite;
    CHECK(nonfinite == 0);

    // Second forward — must be identical for a deterministic graph.
    bt::Tensor out2;
    flux.forward(latent, H_lat, W_lat, 500.0f, prepared,
                 brodiffusion::Branch::Cond, out2);
    bt::sync_all();
    std::vector<float> v2 = bdtest::bd_download(out2);
    CHECK(v1 == v2);

    // ── attention trace ───────────────────────────────────────────────────
    // num_xattn_blocks == num_layers + num_single_layers (one image→text map
    // per joint-attention block).
    const int n_blocks = cfg.num_layers + cfg.num_single_layers;
    CHECK(flux.num_xattn_blocks() == n_blocks);

    // forward_traced must fill exactly n_blocks maps, each (img_len, txt_len),
    // and the velocity output must match a plain forward() to a tight
    // tolerance (the traced path must not change the denoising result).
    const int hp = H_lat / 2, wp = W_lat / 2;
    const int img_len = hp * wp;
    brodiffusion::AttentionTrace trace;
    bt::Tensor out_traced;
    flux.forward_traced(latent, H_lat, W_lat, 500.0f, prepared,
                        brodiffusion::Branch::Cond,
                        /*attn_logit_biases=*/nullptr, &trace, out_traced);
    bt::sync_all();

    CHECK(static_cast<int>(trace.size()) == n_blocks);
    int bad_shape = 0, bad_weight = 0;
    for (const bt::Tensor& m : trace) {
        if (m.rows != img_len || m.cols != txt_len) { ++bad_shape; continue; }
        std::vector<float> mv = bdtest::bd_download(m);
        for (float w : mv) {
            // Head-averaged softmax weights: non-negative and finite. The
            // sliced image→text portion sums to <= 1 per row (text is only
            // part of the joint keys), so we assert non-negativity, not
            // row-sum == 1.
            if (!bdtest::bd_finite(w) || w < -1e-4f) ++bad_weight;
        }
    }
    CHECK(bad_shape == 0);
    CHECK(bad_weight == 0);

    // Traced velocity output matches the plain forward output tightly.
    std::vector<float> vt = bdtest::bd_download(out_traced);
    CHECK(vt.size() == v1.size());
    float max_abs_diff = 0.0f;
    if (vt.size() == v1.size()) {
        for (std::size_t i = 0; i < vt.size(); ++i) {
            float d = std::fabs(vt[i] - v1[i]);
            if (d > max_abs_diff) max_abs_diff = d;
        }
    }
    CHECK(max_abs_diff < 1e-3f);

    // ── joint-attention steering ──────────────────────────────────────────
    // Helper: column sum of an (img_len, txt_len) trace map over all image
    // rows for text token `col` — the total image→text attention mass that
    // text token receives in this block.
    auto col_sum = [&](const bt::Tensor& m, int col) -> double {
        std::vector<float> mv = bdtest::bd_download(m);
        double s = 0.0;
        for (int r = 0; r < m.rows; ++r) {
            s += static_cast<double>(mv[static_cast<std::size_t>(r) * m.cols +
                                        col]);
        }
        return s;
    };

    // A biases vector of all-null entries must produce a trace identical to
    // passing attn_logit_biases = nullptr (within tight tolerance).
    {
        std::vector<const bt::Tensor*> null_biases(
            static_cast<std::size_t>(n_blocks), nullptr);
        brodiffusion::AttentionTrace trace_nb;
        bt::Tensor out_nb;
        flux.forward_traced(latent, H_lat, W_lat, 500.0f, prepared,
                            brodiffusion::Branch::Cond, &null_biases,
                            &trace_nb, out_nb);
        bt::sync_all();
        CHECK(static_cast<int>(trace_nb.size()) == n_blocks);
        float max_trace_diff = 0.0f;
        for (int bi = 0; bi < n_blocks; ++bi) {
            std::vector<float> a = bdtest::bd_download(trace[bi]);
            std::vector<float> c = bdtest::bd_download(trace_nb[bi]);
            CHECK(a.size() == c.size());
            if (a.size() == c.size()) {
                for (std::size_t i = 0; i < a.size(); ++i) {
                    float d = std::fabs(a[i] - c[i]);
                    if (d > max_trace_diff) max_trace_diff = d;
                }
            }
        }
        CHECK(max_trace_diff < 1e-5f);
    }

    // Positive bias on one text token's column must increase that token's
    // image→text attention mass; a negative bias must decrease it. Steer the
    // first double block, inspect block 0's trace map. Bias targets text
    // token `tgt`.
    {
        const int tgt = 1;
        std::vector<float> pos_b(
            static_cast<std::size_t>(img_len) * txt_len, 0.0f);
        std::vector<float> neg_b(
            static_cast<std::size_t>(img_len) * txt_len, 0.0f);
        for (int r = 0; r < img_len; ++r) {
            pos_b[static_cast<std::size_t>(r) * txt_len + tgt] = 4.0f;
            neg_b[static_cast<std::size_t>(r) * txt_len + tgt] = -4.0f;
        }
        bt::Tensor pos_t = bt::Tensor::from_host(pos_b.data(), img_len, txt_len)
                               .to(bt::default_device());
        bt::Tensor neg_t = bt::Tensor::from_host(neg_b.data(), img_len, txt_len)
                               .to(bt::default_device());

        std::vector<const bt::Tensor*> pos_biases(
            static_cast<std::size_t>(n_blocks), nullptr);
        pos_biases[0] = &pos_t;
        std::vector<const bt::Tensor*> neg_biases(
            static_cast<std::size_t>(n_blocks), nullptr);
        neg_biases[0] = &neg_t;

        brodiffusion::AttentionTrace trace_pos, trace_neg;
        bt::Tensor out_pos, out_neg;
        flux.forward_traced(latent, H_lat, W_lat, 500.0f, prepared,
                            brodiffusion::Branch::Cond, &pos_biases,
                            &trace_pos, out_pos);
        flux.forward_traced(latent, H_lat, W_lat, 500.0f, prepared,
                            brodiffusion::Branch::Cond, &neg_biases,
                            &trace_neg, out_neg);
        bt::sync_all();

        const double base = col_sum(trace[0], tgt);
        const double pos  = col_sum(trace_pos[0], tgt);
        const double neg  = col_sum(trace_neg[0], tgt);
        std::printf("flux steer: block0 text-token %d image->text mass  "
                    "base=%.5f  +4bias=%.5f  -4bias=%.5f\n",
                    tgt, base, pos, neg);
        CHECK(pos > base);
        CHECK(neg < base);

        // Biases must work with trace_out null (steer + discard the trace):
        // the steered velocity output is the same whether or not the trace is
        // captured. (Compute-dtype tolerance — on an FP16 backend the extra
        // trace-slice ops perturb GPU reduction order by ~1 ULP, the same band
        // the traced-vs-plain check above uses.)
        bt::Tensor out_pos2;
        flux.forward_traced(latent, H_lat, W_lat, 500.0f, prepared,
                            brodiffusion::Branch::Cond, &pos_biases,
                            /*trace_out=*/nullptr, out_pos2);
        bt::sync_all();
        std::vector<float> op1 = bdtest::bd_download(out_pos);
        std::vector<float> op2 = bdtest::bd_download(out_pos2);
        CHECK(op1.size() == op2.size());
        float op_diff = 0.0f;
        if (op1.size() == op2.size()) {
            for (std::size_t i = 0; i < op1.size(); ++i) {
                float d = std::fabs(op1[i] - op2[i]);
                if (d > op_diff) op_diff = d;
            }
        }
        std::printf("flux steer: trace vs no-trace out max-abs-diff %.3e\n",
                    op_diff);
        CHECK(op_diff < 1e-3f);
    }

    // attn_logit_biases with the wrong length must throw.
    {
        bool len_threw = false;
        std::vector<const bt::Tensor*> bad_len(
            static_cast<std::size_t>(n_blocks + 1), nullptr);
        try {
            bt::Tensor dummy;
            flux.forward_traced(latent, H_lat, W_lat, 500.0f, prepared,
                                brodiffusion::Branch::Cond, &bad_len,
                                /*trace_out=*/nullptr, dummy);
        } catch (const std::exception&) {
            len_threw = true;
        }
        CHECK(len_threw);
    }

    // A bias entry with the wrong shape must throw.
    {
        bool shape_threw = false;
        std::vector<float> wrong(
            static_cast<std::size_t>(img_len) * (txt_len + 1), 0.0f);
        bt::Tensor wrong_t =
            bt::Tensor::from_host(wrong.data(), img_len, txt_len + 1)
                .to(bt::default_device());
        std::vector<const bt::Tensor*> bad_shape_biases(
            static_cast<std::size_t>(n_blocks), nullptr);
        bad_shape_biases[0] = &wrong_t;
        try {
            bt::Tensor dummy;
            flux.forward_traced(latent, H_lat, W_lat, 500.0f, prepared,
                                brodiffusion::Branch::Cond, &bad_shape_biases,
                                /*trace_out=*/nullptr, dummy);
        } catch (const std::exception&) {
            shape_threw = true;
        }
        CHECK(shape_threw);
    }

    // Uncond branch must be rejected.
    bool threw = false;
    try {
        bt::Tensor dummy;
        flux.forward(latent, H_lat, W_lat, 500.0f, prepared,
                     brodiffusion::Branch::Uncond, dummy);
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);

    return v1;
}

}  // namespace

// ─── test ──────────────────────────────────────────────────────────────────

int main() {
    try {
        bt::init();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "init failed: %s\n", e.what());
        return 1;
    }

    auto path = std::filesystem::temp_directory_path() /
                "brodiffusion_flux_test.safetensors";

    // ── flux-schnell variant (guidance_embeds = false) ────────────────────
    {
        dit::FluxConfig cfg;
        cfg.in_channels         = 16;
        cfg.num_layers          = 2;
        cfg.num_single_layers   = 2;
        cfg.attention_head_dim  = 8;
        cfg.num_attention_heads = 2;
        cfg.joint_attention_dim = 10;
        cfg.pooled_projection_dim = 6;
        cfg.guidance_embeds     = false;
        cfg.axes_dims_rope      = {2, 2, 4};
        run_flux(cfg, path);
    }

    // ── flux-dev variant (guidance_embeds = true) ─────────────────────────
    {
        dit::FluxConfig cfg;
        cfg.in_channels         = 16;
        cfg.num_layers          = 2;
        cfg.num_single_layers   = 2;
        cfg.attention_head_dim  = 8;
        cfg.num_attention_heads = 2;
        cfg.joint_attention_dim = 10;
        cfg.pooled_projection_dim = 6;
        cfg.guidance_embeds     = true;
        cfg.axes_dims_rope      = {2, 2, 4};
        run_flux(cfg, path);
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);

    if (g_failures == 0) std::printf("flux_blocks: OK\n");
    else std::fprintf(stderr, "flux_blocks: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
