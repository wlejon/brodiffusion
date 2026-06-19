// Sana model_config parsing test.
//
// Self-contained: writes the real Sana 0.6B diffusers-style config files into a
// unique temp directory, calls load_model_config(), asserts the parsed Sana
// transformer / DC-AE VAE / Gemma-2 / scheduler structs, then cleans up. Does
// NOT depend on downloaded weights.

#include "brodiffusion/model_config.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <variant>
#include <vector>

namespace bd = brodiffusion;
namespace fs = std::filesystem;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

static void write_file(const fs::path& path, const std::string& content) {
    fs::create_directories(path.parent_path());
    std::ofstream f(path, std::ios::binary);
    f << content;
}

static bool approx(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) <= tol;
}

static void test_sana(const fs::path& dir) {
    write_file(dir / "model_index.json",
        R"({"_class_name": "SanaPipeline"})");

    write_file(dir / "transformer" / "config.json", R"({
        "_class_name": "SanaTransformer2DModel",
        "attention_bias": false,
        "attention_head_dim": 32,
        "caption_channels": 2304,
        "cross_attention_dim": 1152,
        "cross_attention_head_dim": 72,
        "dropout": 0.0,
        "in_channels": 32,
        "mlp_ratio": 2.5,
        "norm_elementwise_affine": false,
        "norm_eps": 1e-06,
        "num_attention_heads": 36,
        "num_cross_attention_heads": 16,
        "num_layers": 28,
        "out_channels": 32,
        "patch_size": 1,
        "sample_size": 32
    })");

    write_file(dir / "vae" / "config.json", R"({
        "_class_name": "AutoencoderDC",
        "attention_head_dim": 32,
        "decoder_act_fns": "silu",
        "decoder_block_out_channels": [128, 256, 512, 512, 1024, 1024],
        "decoder_block_types": ["ResBlock", "ResBlock", "ResBlock",
            "EfficientViTBlock", "EfficientViTBlock", "EfficientViTBlock"],
        "decoder_layers_per_block": [3, 3, 3, 3, 3, 3],
        "decoder_norm_types": "rms_norm",
        "decoder_qkv_multiscales": [[], [], [], [5], [5], [5]],
        "downsample_block_type": "Conv",
        "in_channels": 3,
        "latent_channels": 32,
        "scaling_factor": 0.41407,
        "upsample_block_type": "interpolate"
    })");

    write_file(dir / "text_encoder" / "config.json", R"({
        "architectures": ["Gemma2Model"],
        "attention_bias": false,
        "attn_logit_softcapping": 50.0,
        "final_logit_softcapping": 30.0,
        "head_dim": 256,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 2304,
        "intermediate_size": 9216,
        "max_position_embeddings": 8192,
        "model_type": "gemma2",
        "num_attention_heads": 8,
        "num_hidden_layers": 26,
        "num_key_value_heads": 4,
        "query_pre_attn_scalar": 256,
        "rms_norm_eps": 1e-06,
        "rope_theta": 10000.0,
        "sliding_window": 4096,
        "vocab_size": 256000
    })");

    // Real Sana scheduler is DPMSolverMultistepScheduler in flow-prediction
    // mode. (The real file also carries `lambda_min_clipped: -Infinity`, which
    // is not valid JSON; the loader only needs flow_shift / num_train_timesteps
    // here, so the fixture omits the non-finite key.)
    write_file(dir / "scheduler" / "scheduler_config.json", R"({
        "_class_name": "DPMSolverMultistepScheduler",
        "flow_shift": 3.0,
        "num_train_timesteps": 1000,
        "prediction_type": "flow_prediction",
        "solver_order": 2,
        "use_flow_sigmas": true
    })");

    bd::ModelConfig mc = bd::load_model_config(dir.string());

    CHECK(mc.model_class == bd::ModelClass::Sana);

    // Sana transformer.
    CHECK(mc.sana.num_layers == 28);
    CHECK(mc.sana.num_attention_heads == 36);
    CHECK(mc.sana.attention_head_dim == 32);
    CHECK(mc.sana.inner_dim() == 1152);
    CHECK(mc.sana.latent_channels() == 32);
    CHECK(mc.sana.caption_channels == 2304);
    CHECK(mc.sana.cross_attention_dim == 1152);
    CHECK(mc.sana.num_cross_attention_heads == 16);
    CHECK(mc.sana.cross_attention_head_dim == 72);
    CHECK(approx(mc.sana.mlp_ratio, 2.5f));
    CHECK(mc.sana.patch_size == 1);
    CHECK(mc.sana.attention_bias == false);
    CHECK(mc.sana.norm_elementwise_affine == false);

    // DC-AE VAE.
    CHECK(mc.dcae.latent_channels == 32);
    CHECK(mc.dcae.image_channels == 3);
    CHECK(mc.dcae.attention_head_dim == 32);
    CHECK(approx(mc.dcae.scaling_factor, 0.41407f));
    CHECK(mc.dcae.block_out_channels
          == std::vector<int>({128, 256, 512, 512, 1024, 1024}));
    CHECK(mc.dcae.layers_per_block
          == std::vector<int>({3, 3, 3, 3, 3, 3}));
    // 6 stages; the last 3 are EfficientViTBlock (attention).
    CHECK(mc.dcae.is_attention.size() == 6);
    CHECK(mc.dcae.is_attention
          == std::vector<bool>({false, false, false, true, true, true}));
    CHECK(mc.dcae.qkv_multiscales.size() == 6);
    CHECK(mc.dcae.qkv_multiscales[0].empty());
    CHECK(mc.dcae.qkv_multiscales[3] == std::vector<int>({5}));
    CHECK(mc.dcae.qkv_multiscales[5] == std::vector<int>({5}));

    // Gemma-2 text encoder.
    CHECK(mc.gemma.vocab_size == 256000);
    CHECK(mc.gemma.hidden_size == 2304);
    CHECK(mc.gemma.intermediate_size == 9216);
    CHECK(mc.gemma.num_hidden_layers == 26);
    CHECK(mc.gemma.num_attention_heads == 8);
    CHECK(mc.gemma.num_key_value_heads == 4);
    CHECK(mc.gemma.head_dim == 256);
    CHECK(approx(mc.gemma.rms_norm_eps, 1e-6f, 1e-9f));
    CHECK(approx(mc.gemma.query_pre_attn_scalar, 256.0f));
    CHECK(mc.gemma.sliding_window == 4096);
    CHECK(approx(mc.gemma.attn_logit_softcapping, 50.0f));
    CHECK(approx(mc.gemma.final_logit_softcapping, 30.0f));
    CHECK(mc.gemma.max_position_embeddings == 8192);

    // Scheduler: DPMSolverMultistep(flow) -> FlowMatchConfig with shift=flow_shift.
    CHECK(std::holds_alternative<bd::scheduler::FlowMatchConfig>(mc.scheduler));
    if (std::holds_alternative<bd::scheduler::FlowMatchConfig>(mc.scheduler)) {
        const auto& fc = std::get<bd::scheduler::FlowMatchConfig>(mc.scheduler);
        CHECK(approx(fc.shift, 3.0f));
        CHECK(fc.num_train_timesteps == 1000);
    }
}

int main() {
    fs::path base = fs::temp_directory_path()
                  / ("brodiffusion_sana_config_test_"
                     + std::to_string(
                           std::hash<std::string>{}(
                               std::to_string(
                                   reinterpret_cast<std::uintptr_t>(&base)))));
    fs::create_directories(base);

    try {
        test_sana(base / "sana");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL exception: %s\n", e.what());
        ++g_failures;
    }

    std::error_code ec;
    fs::remove_all(base, ec);

    if (g_failures == 0) {
        std::fprintf(stderr, "test_sana_config: all checks passed\n");
    }
    return g_failures == 0 ? 0 : 1;
}
