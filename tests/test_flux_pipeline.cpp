// Flux end-to-end Pipeline assembly test.
//
// Builds a *tiny* but fully consistent Flux model directory in diffusers
// layout — model_index.json + per-component config.json files, synthetic
// safetensors weights for CLIP / T5 / the Flux transformer / the VAE decoder,
// and the two tokenizer files — then loads it through
// Pipeline::from_model_dir() and runs generate().
//
// Dimensions are kept consistent across modules exactly as the real model
// requires: CLIP hidden_dim == FluxConfig.pooled_projection_dim,
// T5 d_model == FluxConfig.joint_attention_dim, and the VAE latent_channels
// == FluxDenoiser::latent_channels() (= flux.in_channels / 4).
//
// This exercises the whole Phase 6 assembly: model_config ingestion, the
// open_component_files() sharded-loader helper (single-file path), the Flux
// Pipeline constructor, encode_prompt_ branching (T5 sequence + CLIP pooled),
// FlowMatch scheduling, the FluxDenoiser forward, and the VAE decode. It
// asserts the output image buffer has the expected size and is all-finite.
//
// Numerical accuracy against a reference is intentionally not checked — that
// needs real Flux weights and lives in image-generation integration.

#include "brodiffusion/pipeline.h"

#include "brotensor/runtime.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <system_error>
#include <vector>

namespace pl = brodiffusion::pipeline;
namespace bt = brotensor;
namespace fs = std::filesystem;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

// ─── safetensors fixture builder (FP16-only) ───────────────────────────────

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

    void write(const fs::path& path) const {
        fs::create_directories(path.parent_path());
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

std::vector<uint16_t> fp16_zeros(std::size_t n) {
    return std::vector<uint16_t>(n, 0);
}
std::vector<uint16_t> fp16_ones(std::size_t n) {
    return std::vector<uint16_t>(n, bt::fp32_to_fp16_bits(1.0f));
}
// Small values clustered around zero so the cascade of layers stays tame.
std::vector<uint16_t> fp16_seq(std::size_t n, float scale, std::size_t salt = 0) {
    std::vector<uint16_t> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        float v = (static_cast<float>((i + salt) % 7) - 3.0f) * scale;
        out[i] = bt::fp32_to_fp16_bits(v);
    }
    return out;
}

void write_text(const fs::path& path, const std::string& content) {
    fs::create_directories(path.parent_path());
    std::ofstream(path, std::ios::binary | std::ios::trunc) << content;
}

// Emit a biased Linear (weight (out,in) + bias (out,)).
void add_linear(Builder& b, const std::string& key, int out, int in,
                float wscale, float bscale) {
    b.add(key + ".weight", {out, in},
          fp16_seq(static_cast<std::size_t>(out) * in, wscale));
    b.add(key + ".bias", {out}, fp16_seq(static_cast<std::size_t>(out), bscale));
}

// ─── component config dimensions ───────────────────────────────────────────

// Flux transformer.
constexpr int kFluxInChannels   = 16;   // -> latent_channels = 4
constexpr int kFluxLayers       = 2;
constexpr int kFluxSingleLayers = 2;
constexpr int kHeadDim          = 8;
constexpr int kNumHeads         = 2;    // inner_dim = 16
constexpr int kJointDim         = 8;    // == T5 d_model
constexpr int kPooledDim        = 4;    // == CLIP hidden_dim
constexpr int kInnerDim         = kNumHeads * kHeadDim;  // 16
constexpr int kLatentChannels   = kFluxInChannels / 4;   // 4

// CLIP text encoder. The CLIP tokenizer always emits a fixed 77-token
// sequence framed with bos=49406 / eos=49407, so the encoder's max_position
// must be 77 and its vocab must span the full CLIP id range.
constexpr int kClipVocab   = 49408;
constexpr int kClipMaxPos  = 77;
constexpr int kClipFFN     = 8;

// T5 text encoder.
constexpr int kT5Vocab     = 20;
constexpr int kT5DKV       = 4;
constexpr int kT5Heads     = 2;     // num_heads * d_kv == d_model (8)
constexpr int kT5FF        = 16;
constexpr int kT5Layers    = 2;
constexpr int kT5Buckets   = 8;
constexpr int kT5MaxDist   = 16;

// VAE decoder. Four block_out_channels -> three 2x upsamples -> the 8x latent
// upscale the pipeline's decode() assumes.
const std::vector<int> kVaeBlocks = {4, 4, 8, 8};
constexpr int kVaeGroups   = 2;
constexpr int kVaeLayers   = 1;   // layers_per_block

// ─── per-component fixture builders ────────────────────────────────────────

void build_flux(const fs::path& dir) {
    Builder b;
    const int D  = kInnerDim;
    const int IC = kFluxInChannels;
    const int JD = kJointDim;
    const int PD = kPooledDim;
    const int HD = kHeadDim;

    add_linear(b, "x_embedder", D, IC, 0.05f, 0.01f);
    add_linear(b, "context_embedder", D, JD, 0.05f, 0.01f);

    const std::string tt = "time_text_embed.";
    add_linear(b, tt + "timestep_embedder.linear_1", D, 256, 0.02f, 0.01f);
    add_linear(b, tt + "timestep_embedder.linear_2", D, D, 0.05f, 0.01f);
    add_linear(b, tt + "text_embedder.linear_1", D, PD, 0.05f, 0.01f);
    add_linear(b, tt + "text_embedder.linear_2", D, D, 0.05f, 0.01f);

    for (int i = 0; i < kFluxLayers; ++i) {
        const std::string p = "transformer_blocks." + std::to_string(i) + ".";
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
    for (int i = 0; i < kFluxSingleLayers; ++i) {
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

    b.write(dir / "transformer" / "diffusion_pytorch_model.safetensors");
    write_text(dir / "transformer" / "config.json",
        "{\"in_channels\":" + std::to_string(kFluxInChannels) +
        ",\"num_layers\":" + std::to_string(kFluxLayers) +
        ",\"num_single_layers\":" + std::to_string(kFluxSingleLayers) +
        ",\"attention_head_dim\":" + std::to_string(kHeadDim) +
        ",\"num_attention_heads\":" + std::to_string(kNumHeads) +
        ",\"joint_attention_dim\":" + std::to_string(kJointDim) +
        ",\"pooled_projection_dim\":" + std::to_string(kPooledDim) +
        ",\"guidance_embeds\":false,\"axes_dims_rope\":[2,2,4]}");
}

void build_clip(const fs::path& dir) {
    Builder b;
    const int V = kClipVocab;
    const int P = kClipMaxPos;
    const int D = kPooledDim;        // CLIP hidden_dim
    const int F = kClipFFN;
    const std::string p = "text_model.";

    b.add(p + "embeddings.token_embedding.weight", {V, D},
          fp16_seq(static_cast<std::size_t>(V) * D, 0.05f));
    b.add(p + "embeddings.position_embedding.weight", {P, D},
          fp16_seq(static_cast<std::size_t>(P) * D, 0.05f));

    const std::string lp = p + "encoder.layers.0.";
    b.add(lp + "layer_norm1.weight", {D}, fp16_ones(D));
    b.add(lp + "layer_norm1.bias",   {D}, fp16_zeros(D));
    auto W = [&](float s) { return fp16_seq(static_cast<std::size_t>(D) * D, s); };
    b.add(lp + "self_attn.q_proj.weight",   {D, D}, W(0.02f));
    b.add(lp + "self_attn.q_proj.bias",     {D},    fp16_zeros(D));
    b.add(lp + "self_attn.k_proj.weight",   {D, D}, W(0.03f));
    b.add(lp + "self_attn.k_proj.bias",     {D},    fp16_zeros(D));
    b.add(lp + "self_attn.v_proj.weight",   {D, D}, W(0.04f));
    b.add(lp + "self_attn.v_proj.bias",     {D},    fp16_zeros(D));
    b.add(lp + "self_attn.out_proj.weight", {D, D}, W(0.05f));
    b.add(lp + "self_attn.out_proj.bias",   {D},    fp16_zeros(D));
    b.add(lp + "layer_norm2.weight", {D}, fp16_ones(D));
    b.add(lp + "layer_norm2.bias",   {D}, fp16_zeros(D));
    b.add(lp + "mlp.fc1.weight", {F, D},
          fp16_seq(static_cast<std::size_t>(F) * D, 0.01f));
    b.add(lp + "mlp.fc1.bias",   {F}, fp16_zeros(F));
    b.add(lp + "mlp.fc2.weight", {D, F},
          fp16_seq(static_cast<std::size_t>(D) * F, 0.01f));
    b.add(lp + "mlp.fc2.bias",   {D}, fp16_zeros(D));
    b.add(p + "final_layer_norm.weight", {D}, fp16_ones(D));
    b.add(p + "final_layer_norm.bias",   {D}, fp16_zeros(D));

    b.write(dir / "text_encoder" / "model.safetensors");
    write_text(dir / "text_encoder" / "config.json",
        "{\"hidden_size\":" + std::to_string(kPooledDim) +
        ",\"intermediate_size\":" + std::to_string(kClipFFN) +
        ",\"num_attention_heads\":1,\"num_hidden_layers\":1"
        ",\"max_position_embeddings\":" + std::to_string(kClipMaxPos) +
        ",\"vocab_size\":" + std::to_string(kClipVocab) +
        ",\"layer_norm_eps\":1e-05}");
}

void build_t5(const fs::path& dir) {
    Builder b;
    const int V  = kT5Vocab;
    const int D  = kJointDim;     // T5 d_model
    const int FF = kT5FF;
    const int NB = kT5Buckets;
    const int H  = kT5Heads;
    const std::string prefix = "";

    b.add(prefix + "shared.weight", {V, D},
          fp16_seq(static_cast<std::size_t>(V) * D, 0.03f));
    auto WDD = [&](float s) { return fp16_seq(static_cast<std::size_t>(D) * D, s); };
    for (int i = 0; i < kT5Layers; ++i) {
        const std::string p =
            prefix + "encoder.block." + std::to_string(i) + ".";
        b.add(p + "layer.0.layer_norm.weight", {D}, fp16_ones(D));
        const std::string sa = p + "layer.0.SelfAttention.";
        b.add(sa + "q.weight", {D, D}, WDD(0.02f));
        b.add(sa + "k.weight", {D, D}, WDD(0.03f));
        b.add(sa + "v.weight", {D, D}, WDD(0.04f));
        b.add(sa + "o.weight", {D, D}, WDD(0.05f));
        if (i == 0) {
            b.add(sa + "relative_attention_bias.weight", {NB, H},
                  fp16_seq(static_cast<std::size_t>(NB) * H, 0.01f));
        }
        b.add(p + "layer.1.layer_norm.weight", {D}, fp16_ones(D));
        const std::string dr = p + "layer.1.DenseReluDense.";
        b.add(dr + "wi_0.weight", {FF, D},
              fp16_seq(static_cast<std::size_t>(FF) * D, 0.01f));
        b.add(dr + "wi_1.weight", {FF, D},
              fp16_seq(static_cast<std::size_t>(FF) * D, 0.015f));
        b.add(dr + "wo.weight", {D, FF},
              fp16_seq(static_cast<std::size_t>(D) * FF, 0.01f));
    }
    b.add(prefix + "encoder.final_layer_norm.weight", {D}, fp16_ones(D));

    b.write(dir / "text_encoder_2" / "model.safetensors");
    write_text(dir / "text_encoder_2" / "config.json",
        "{\"d_model\":" + std::to_string(kJointDim) +
        ",\"d_ff\":" + std::to_string(kT5FF) +
        ",\"d_kv\":" + std::to_string(kT5DKV) +
        ",\"num_heads\":" + std::to_string(kT5Heads) +
        ",\"num_layers\":" + std::to_string(kT5Layers) +
        ",\"relative_attention_num_buckets\":" + std::to_string(kT5Buckets) +
        ",\"relative_attention_max_distance\":" + std::to_string(kT5MaxDist) +
        ",\"vocab_size\":" + std::to_string(kT5Vocab) +
        ",\"layer_norm_epsilon\":1e-06}");
}

// Emit one VAE resnet's tensors under prefix p (already ending with ".").
void emit_resnet(Builder& b, const std::string& p, int C_in, int C_out) {
    b.add(p + "norm1.weight", {C_in},  fp16_ones(C_in));
    b.add(p + "norm1.bias",   {C_in},  fp16_zeros(C_in));
    b.add(p + "conv1.weight", {C_out, C_in, 3, 3},
          fp16_seq(static_cast<std::size_t>(C_out) * C_in * 9, 0.02f, p.size()));
    b.add(p + "conv1.bias",   {C_out}, fp16_zeros(C_out));
    b.add(p + "norm2.weight", {C_out}, fp16_ones(C_out));
    b.add(p + "norm2.bias",   {C_out}, fp16_zeros(C_out));
    b.add(p + "conv2.weight", {C_out, C_out, 3, 3},
          fp16_seq(static_cast<std::size_t>(C_out) * C_out * 9, 0.02f,
                   p.size() + 1));
    b.add(p + "conv2.bias",   {C_out}, fp16_zeros(C_out));
    if (C_in != C_out) {
        b.add(p + "conv_shortcut.weight", {C_out, C_in, 1, 1},
              fp16_seq(static_cast<std::size_t>(C_out) * C_in, 0.05f,
                       p.size() + 2));
        b.add(p + "conv_shortcut.bias",   {C_out}, fp16_zeros(C_out));
    }
}

void build_vae(const fs::path& dir) {
    Builder b;
    const std::string p = "decoder.";
    const int in_ch   = kLatentChannels;   // VAE latent in == FluxDenoiser LC
    const int out_ch  = 3;
    const int nb      = static_cast<int>(kVaeBlocks.size());
    const int mid_C   = kVaeBlocks.back();
    const int first_C = kVaeBlocks.front();

    b.add(p + "conv_in.weight", {mid_C, in_ch, 3, 3},
          fp16_seq(static_cast<std::size_t>(mid_C) * in_ch * 9, 0.05f));
    b.add(p + "conv_in.bias",   {mid_C}, fp16_zeros(mid_C));

    emit_resnet(b, p + "mid_block.resnets.0.", mid_C, mid_C);
    emit_resnet(b, p + "mid_block.resnets.1.", mid_C, mid_C);

    const std::string ap = p + "mid_block.attentions.0.";
    b.add(ap + "group_norm.weight", {mid_C}, fp16_ones(mid_C));
    b.add(ap + "group_norm.bias",   {mid_C}, fp16_zeros(mid_C));
    auto WC = [&](std::size_t salt) {
        return fp16_seq(static_cast<std::size_t>(mid_C) * mid_C, 0.03f, salt);
    };
    b.add(ap + "query.weight", {mid_C, mid_C}, WC(11));
    b.add(ap + "query.bias",   {mid_C}, fp16_zeros(mid_C));
    b.add(ap + "key.weight",   {mid_C, mid_C}, WC(13));
    b.add(ap + "key.bias",     {mid_C}, fp16_zeros(mid_C));
    b.add(ap + "value.weight", {mid_C, mid_C}, WC(17));
    b.add(ap + "value.bias",   {mid_C}, fp16_zeros(mid_C));
    b.add(ap + "proj_attn.weight", {mid_C, mid_C}, WC(19));
    b.add(ap + "proj_attn.bias",   {mid_C}, fp16_zeros(mid_C));

    int C_prev = mid_C;
    for (int i = 0; i < nb; ++i) {
        int C_block = kVaeBlocks[static_cast<std::size_t>(nb - 1 - i)];
        for (int j = 0; j <= kVaeLayers; ++j) {
            int Ci = (j == 0) ? C_prev : C_block;
            emit_resnet(b, p + "up_blocks." + std::to_string(i) +
                        ".resnets." + std::to_string(j) + ".", Ci, C_block);
        }
        if (i + 1 < nb) {
            const std::string up = p + "up_blocks." + std::to_string(i) +
                                   ".upsamplers.0.conv.";
            b.add(up + "weight", {C_block, C_block, 3, 3},
                  fp16_seq(static_cast<std::size_t>(C_block) * C_block * 9,
                           0.02f, i + 23));
            b.add(up + "bias",   {C_block}, fp16_zeros(C_block));
        }
        C_prev = C_block;
    }

    b.add(p + "conv_norm_out.weight", {first_C}, fp16_ones(first_C));
    b.add(p + "conv_norm_out.bias",   {first_C}, fp16_zeros(first_C));
    b.add(p + "conv_out.weight", {out_ch, first_C, 3, 3},
          fp16_seq(static_cast<std::size_t>(out_ch) * first_C * 9, 0.04f));
    b.add(p + "conv_out.bias",   {out_ch}, fp16_zeros(out_ch));

    // ── Encoder weights (mirror Decoder topology, forward channel order) ──
    // Pipeline now loads the encoder unconditionally from a model dir; the
    // tensors must exist even though Flux img2img isn't supported yet.
    {
        const std::string ep = "encoder.";
        const int twoC = 2 * in_ch;
        b.add(ep + "conv_in.weight", {first_C, out_ch, 3, 3},
              fp16_seq(static_cast<std::size_t>(first_C) * out_ch * 9, 0.05f, 600));
        b.add(ep + "conv_in.bias",   {first_C}, fp16_zeros(first_C));

        int Ce_prev = first_C;
        for (int i = 0; i < nb; ++i) {
            const int C_block = kVaeBlocks[static_cast<std::size_t>(i)];
            for (int j = 0; j < kVaeLayers; ++j) {
                const int Ci = (j == 0) ? Ce_prev : C_block;
                emit_resnet(b, ep + "down_blocks." + std::to_string(i) +
                            ".resnets." + std::to_string(j) + ".",
                            Ci, C_block);
            }
            if (i + 1 < nb) {
                const std::string dp = ep + "down_blocks." +
                                       std::to_string(i) +
                                       ".downsamplers.0.conv.";
                b.add(dp + "weight", {C_block, C_block, 3, 3},
                      fp16_seq(static_cast<std::size_t>(C_block) * C_block * 9,
                               0.02f, static_cast<std::size_t>(700 + i)));
                b.add(dp + "bias",   {C_block}, fp16_zeros(C_block));
            }
            Ce_prev = C_block;
        }

        emit_resnet(b, ep + "mid_block.resnets.0.", mid_C, mid_C);
        emit_resnet(b, ep + "mid_block.resnets.1.", mid_C, mid_C);

        const std::string eap = ep + "mid_block.attentions.0.";
        b.add(eap + "group_norm.weight", {mid_C}, fp16_ones(mid_C));
        b.add(eap + "group_norm.bias",   {mid_C}, fp16_zeros(mid_C));
        b.add(eap + "query.weight",      {mid_C, mid_C}, WC(611));
        b.add(eap + "query.bias",        {mid_C}, fp16_zeros(mid_C));
        b.add(eap + "key.weight",        {mid_C, mid_C}, WC(613));
        b.add(eap + "key.bias",          {mid_C}, fp16_zeros(mid_C));
        b.add(eap + "value.weight",      {mid_C, mid_C}, WC(617));
        b.add(eap + "value.bias",        {mid_C}, fp16_zeros(mid_C));
        b.add(eap + "proj_attn.weight",  {mid_C, mid_C}, WC(619));
        b.add(eap + "proj_attn.bias",    {mid_C}, fp16_zeros(mid_C));

        b.add(ep + "conv_norm_out.weight", {mid_C}, fp16_ones(mid_C));
        b.add(ep + "conv_norm_out.bias",   {mid_C}, fp16_zeros(mid_C));
        b.add(ep + "conv_out.weight", {twoC, mid_C, 3, 3},
              fp16_seq(static_cast<std::size_t>(twoC) * mid_C * 9, 0.04f, 631));
        b.add(ep + "conv_out.bias",   {twoC}, fp16_zeros(twoC));
    }

    b.write(dir / "vae" / "diffusion_pytorch_model.safetensors");
    std::string blocks = "[";
    for (std::size_t i = 0; i < kVaeBlocks.size(); ++i) {
        if (i) blocks += ",";
        blocks += std::to_string(kVaeBlocks[i]);
    }
    blocks += "]";
    write_text(dir / "vae" / "config.json",
        "{\"in_channels\":3,\"out_channels\":3,\"latent_channels\":" +
        std::to_string(kLatentChannels) +
        ",\"block_out_channels\":" + blocks + ",\"layers_per_block\":" +
        std::to_string(kVaeLayers) +
        ",\"norm_num_groups\":" + std::to_string(kVaeGroups) +
        ",\"scaling_factor\":1.0,\"shift_factor\":0.0}");
}

// Tiny HF CLIP-style vocab.json + merges.txt: a couple of byte-level tokens.
void build_clip_tokenizer(const fs::path& dir) {
    write_text(dir / "tokenizer" / "vocab.json",
               "{\"a\":1,\"a</w>\":2}");
    write_text(dir / "tokenizer" / "merges.txt", "#version: test\n");
}

// Tiny Unigram tokenizer.json for T5 (mirrors test_tokenizer_t5's fixture).
void build_t5_tokenizer(const fs::path& dir) {
    const std::string M = "\xE2\x96\x81";  // U+2581 metaspace
    std::string j = "{\"model\":{\"type\":\"Unigram\",\"unk_id\":2,\"vocab\":[";
    j += "[\"<pad>\",0.0],[\"</s>\",0.0],[\"<unk>\",0.0],";
    j += "[\"" + M + "\",-2.0],[\"" + M + "a\",-3.0],[\"a\",-6.0],";
    j += "[\"cat\",-3.5],[\"" + M + "cat\",-3.0],[\"c\",-7.0],[\"at\",-7.0]";
    j += "]}}";
    write_text(dir / "tokenizer_2" / "tokenizer.json", j);
}

void build_model_dir(const fs::path& dir) {
    write_text(dir / "model_index.json", "{\"_class_name\":\"FluxPipeline\"}");
    write_text(dir / "scheduler" / "scheduler_config.json",
        "{\"_class_name\":\"FlowMatchEulerDiscreteScheduler\","
        "\"num_train_timesteps\":1000,\"shift\":3.0,"
        "\"use_dynamic_shifting\":false}");
    write_text(dir / "tokenizer_2" / "tokenizer_config.json",
               "{\"model_max_length\":256}");
    build_flux(dir);
    build_clip(dir);
    build_t5(dir);
    build_vae(dir);
    build_clip_tokenizer(dir);
    build_t5_tokenizer(dir);
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

    fs::path base = fs::temp_directory_path()
                  / ("brodiffusion_flux_pipeline_test_"
                     + std::to_string(std::hash<std::string>{}(
                           std::to_string(reinterpret_cast<std::uintptr_t>(
                               &base)))));
    std::error_code ec;
    fs::remove_all(base, ec);
    fs::create_directories(base);

    try {
        build_model_dir(base);

        // Whole-assembly load: model_config + the right sub-modules + all
        // component weights + tokenizers via the open_component_files() helper.
        pl::Pipeline pipeline = pl::Pipeline::from_model_dir(base.string());

        CHECK(pipeline.config().model_class ==
              brodiffusion::ModelClass::Flux);
        // tokenizer_config.json's model_max_length (256) wins over the
        // 512 default for the T5 sequence length.
        CHECK(pipeline.config().t5_max_length == 256);
        CHECK(pipeline.config().flux.in_channels == kFluxInChannels);

        // Run a few denoising steps at a small image size. Latent dims are
        // H/8, W/8 -> 2x2 here, with image_seq_len = 1x1 token after 2x2
        // patch packing.
        pl::GenerateOptions opts;
        opts.height = 16;
        opts.width  = 16;
        opts.num_inference_steps = 3;
        opts.guidance_scale = 0.0f;   // schnell: guidance_embeds=false
        opts.seed = 7;

        std::vector<float> img = pipeline.generate("a cat", opts);

        const std::size_t expected =
            static_cast<std::size_t>(3) * opts.height * opts.width;
        CHECK(img.size() == expected);

        int nonfinite = 0;
        for (float v : img) {
            if (!std::isfinite(v)) ++nonfinite;
        }
        CHECK(nonfinite == 0);

        // A second generate() with the same seed must be byte-identical
        // (deterministic graph).
        std::vector<float> img2 = pipeline.generate("a cat", opts);
        CHECK(img == img2);

        // ── INT8 W8A16 quantized load (GPU only) ─────────────────────────
        // Re-load the same model dir with quantize=true and compare against
        // the dense run. Per-output-row symmetric INT8 introduces bounded
        // per-layer error; on this tiny synthetic fixture the end-to-end
        // image error stays small. On the CPU backend the flag downgrades
        // to a dense load with a warning, making the outputs identical —
        // the bound holds trivially.
        {
            pl::Pipeline::ModelDirOptions dopts;
            dopts.quantize = true;
            pl::Pipeline qp = pl::Pipeline::from_model_dir(base.string(),
                                                           dopts);
            std::vector<float> qimg = qp.generate("a cat", opts);
            CHECK(qimg.size() == img.size());
            float max_abs_err = 0.0f;
            int nf = 0;
            for (std::size_t i = 0; i < qimg.size(); ++i) {
                if (!std::isfinite(qimg[i])) ++nf;
                const float e = std::fabs(qimg[i] - img[i]);
                if (e > max_abs_err) max_abs_err = e;
            }
            CHECK(nf == 0);
            std::printf("flux quantized vs dense: max_abs_err=%.4f\n",
                        max_abs_err);
            CHECK(max_abs_err < 0.08f);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL exception: %s\n", e.what());
        ++g_failures;
    }

    fs::remove_all(base, ec);

    if (g_failures == 0) std::printf("flux_pipeline: OK\n");
    else std::fprintf(stderr, "flux_pipeline: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
