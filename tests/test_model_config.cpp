// model_config + JSON parser test.
//
// Self-contained: writes synthetic diffusers-style fixture files into a unique
// temp directory, calls load_model_config(), asserts the parsed structs, then
// cleans up. Does NOT depend on downloaded weights.

#include "brodiffusion/model_config.h"
#include "brodiffusion/detail/json.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <variant>

namespace bd  = brodiffusion;
namespace fs  = std::filesystem;
namespace json = brodiffusion::detail::json;

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

// ── JSON parser unit check ─────────────────────────────────────────────────
static void test_json_parser() {
    const std::string text = R"({
        "name": "demo",
        "nested": { "a": 1, "b": 2.5 },
        "arr": [320, 640, 1280],
        "scale": -1.5e-05,
        "big": 3.0,
        "flag_t": true,
        "flag_f": false,
        "missing_val": null,
        "esc": "line\nbreak\tA"
    })";

    json::Value v = json::parse(text);
    CHECK(v.is_object());
    CHECK(v.contains("name"));
    CHECK(!v.contains("nope"));

    CHECK(v.at("name").as_string() == "demo");
    CHECK(v.find("nope") == nullptr);
    CHECK(v.find("name") != nullptr);

    // Nested object.
    const json::Value& nested = v.at("nested");
    CHECK(nested.is_object());
    CHECK(nested.get_int("a", -1) == 1);
    CHECK(approx(nested.get_float("b", 0.0f), 2.5f));

    // Array.
    const json::Value& arr = v.at("arr");
    CHECK(arr.is_array());
    CHECK(arr.as_array().size() == 3);
    CHECK(arr.as_array()[1].as_number() == 640.0);

    // Float with exponent.
    CHECK(approx(v.get_float("scale", 0.0f), -1.5e-05f, 1e-9f));
    CHECK(approx(v.get_float("big", 0.0f), 3.0f));

    // Bool.
    CHECK(v.get_bool("flag_t", false) == true);
    CHECK(v.get_bool("flag_f", true) == false);

    // null OR absent -> default.
    CHECK(v.get_int("missing_val", 99) == 99);
    CHECK(v.get_int("absent_key", 77) == 77);
    CHECK(v.get_float("missing_val", 1.25f) == 1.25f);
    CHECK(v.get_bool("missing_val", true) == true);
    CHECK(v.get_string("missing_val", "dflt") == "dflt");

    // Escapes.
    CHECK(v.get_string("esc", "") == "line\nbreak\tA");

    // get_int_array convenience + default.
    std::vector<int> ia = v.get_int_array("arr", {});
    CHECK(ia.size() == 3 && ia[0] == 320 && ia[2] == 1280);
    std::vector<int> dflt = v.get_int_array("missing_val", {1, 2});
    CHECK(dflt.size() == 2 && dflt[0] == 1 && dflt[1] == 2);

    // Type-mismatch accessors throw.
    bool threw = false;
    try { v.at("name").as_number(); } catch (const std::exception&) { threw = true; }
    CHECK(threw);
}

// ── SD1.5 synthetic dir ────────────────────────────────────────────────────
static void test_sd15(const fs::path& base) {
    const fs::path dir = base / "sd15";

    write_file(dir / "model_index.json",
        R"({"_class_name": "StableDiffusionPipeline"})");
    write_file(dir / "unet" / "config.json", R"({
        "in_channels": 4, "out_channels": 4,
        "block_out_channels": [320, 640, 1280, 1280],
        "layers_per_block": 2, "norm_num_groups": 32,
        "cross_attention_dim": 768,
        "attention_head_dim": 8,
        "time_cond_proj_dim": null
    })");
    write_file(dir / "vae" / "config.json", R"({
        "in_channels": 3, "out_channels": 3,
        "latent_channels": 4,
        "block_out_channels": [128, 256, 512, 512],
        "layers_per_block": 2, "norm_num_groups": 32,
        "scaling_factor": 0.18215
    })");
    write_file(dir / "text_encoder" / "config.json", R"({
        "hidden_size": 768, "intermediate_size": 3072,
        "num_attention_heads": 12, "num_hidden_layers": 12,
        "max_position_embeddings": 77, "vocab_size": 49408,
        "layer_norm_eps": 1e-05
    })");
    write_file(dir / "scheduler" / "scheduler_config.json",
        R"({"_class_name": "PNDMScheduler", "num_train_timesteps": 1000,
            "beta_start": 0.00085, "beta_end": 0.012, "steps_offset": 1})");

    bd::ModelConfig mc = bd::load_model_config(dir.string());

    CHECK(mc.model_class == bd::ModelClass::StableDiffusion);

    CHECK(mc.unet.block_out_channels
          == std::vector<int>({320, 640, 1280, 1280}));
    CHECK(mc.unet.cross_attention_dim == 768);
    CHECK(mc.unet.attention_head_dim == 8);
    CHECK(mc.unet.time_cond_proj_dim == 0);  // null -> default

    // VAE: in_channels comes from latent_channels.
    CHECK(mc.vae.in_channels == 4);
    CHECK(mc.vae.out_channels == 3);
    CHECK(approx(mc.vae.shift_factor, 0.0f));
    CHECK(approx(mc.vae.scaling_factor, 0.18215f));

    CHECK(mc.text_encoder.hidden_dim == 768);
    CHECK(mc.text_encoder.num_layers == 12);

    CHECK(std::holds_alternative<bd::scheduler::DDIMConfig>(mc.scheduler));
}

// ── LCM variant ────────────────────────────────────────────────────────────
static void test_lcm(const fs::path& base) {
    const fs::path dir = base / "lcm";

    write_file(dir / "model_index.json",
        R"({"_class_name": "StableDiffusionPipeline"})");
    write_file(dir / "unet" / "config.json", R"({
        "in_channels": 4, "out_channels": 4,
        "block_out_channels": [320, 640, 1280, 1280],
        "time_cond_proj_dim": 256
    })");
    write_file(dir / "vae" / "config.json",
        R"({"latent_channels": 4, "out_channels": 3})");
    write_file(dir / "scheduler" / "scheduler_config.json",
        R"({"_class_name": "LCMScheduler", "num_train_timesteps": 1000,
            "original_inference_steps": 50, "timestep_scaling": 10.0})");

    bd::ModelConfig mc = bd::load_model_config(dir.string());

    CHECK(mc.model_class == bd::ModelClass::StableDiffusion);
    CHECK(mc.unet.time_cond_proj_dim == 256);
    CHECK(std::holds_alternative<bd::scheduler::LCMConfig>(mc.scheduler));
    if (std::holds_alternative<bd::scheduler::LCMConfig>(mc.scheduler)) {
        const auto& lc = std::get<bd::scheduler::LCMConfig>(mc.scheduler);
        CHECK(lc.original_inference_steps == 50);
        CHECK(approx(lc.timestep_scaling, 10.0f));
    }
}

// ── Flux synthetic dir ─────────────────────────────────────────────────────
static void test_flux(const fs::path& base) {
    const fs::path dir = base / "flux";

    write_file(dir / "model_index.json",
        R"({"_class_name": "FluxPipeline"})");
    write_file(dir / "vae" / "config.json", R"({
        "in_channels": 3, "out_channels": 3,
        "latent_channels": 16,
        "scaling_factor": 0.3611,
        "shift_factor": 0.1159
    })");
    write_file(dir / "scheduler" / "scheduler_config.json", R"({
        "_class_name": "FlowMatchEulerDiscreteScheduler",
        "num_train_timesteps": 1000,
        "shift": 3.0,
        "use_dynamic_shifting": false
    })");
    write_file(dir / "transformer" / "config.json", R"({
        "in_channels": 64,
        "num_layers": 19,
        "num_single_layers": 38,
        "attention_head_dim": 128,
        "num_attention_heads": 24,
        "joint_attention_dim": 4096,
        "pooled_projection_dim": 768,
        "guidance_embeds": false,
        "axes_dims_rope": [16, 56, 56]
    })");
    write_file(dir / "text_encoder_2" / "config.json", R"({
        "d_model": 4096,
        "d_ff": 10240,
        "d_kv": 64,
        "num_heads": 64,
        "num_layers": 24,
        "relative_attention_num_buckets": 32,
        "relative_attention_max_distance": 128,
        "vocab_size": 32128,
        "layer_norm_epsilon": 1e-06
    })");

    bd::ModelConfig mc = bd::load_model_config(dir.string());

    CHECK(mc.model_class == bd::ModelClass::Flux);
    CHECK(mc.vae.in_channels == 16);
    CHECK(approx(mc.vae.shift_factor, 0.1159f));
    CHECK(std::holds_alternative<bd::scheduler::FlowMatchConfig>(mc.scheduler));
    if (std::holds_alternative<bd::scheduler::FlowMatchConfig>(mc.scheduler)) {
        const auto& fc = std::get<bd::scheduler::FlowMatchConfig>(mc.scheduler);
        CHECK(approx(fc.shift, 3.0f));
    }

    // Flux transformer config.
    CHECK(mc.flux.in_channels == 64);
    CHECK(mc.flux.num_layers == 19);
    CHECK(mc.flux.num_single_layers == 38);
    CHECK(mc.flux.guidance_embeds == false);
    CHECK(mc.flux.axes_dims_rope == std::vector<int>({16, 56, 56}));

    // T5 (text_encoder_2) config.
    CHECK(mc.t5.d_model == 4096);
    CHECK(mc.t5.num_layers == 24);
    CHECK(mc.t5.num_heads == 64);
    CHECK(approx(mc.t5.layer_norm_eps, 1e-6f, 1e-9f));
}

int main() {
    test_json_parser();

    fs::path base = fs::temp_directory_path()
                  / ("brodiffusion_model_config_test_"
                     + std::to_string(
                           std::hash<std::string>{}(
                               std::to_string(reinterpret_cast<std::uintptr_t>(&base)))));
    fs::create_directories(base);

    try {
        test_sd15(base);
        test_lcm(base);
        test_flux(base);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL exception: %s\n", e.what());
        ++g_failures;
    }

    // Cleanup.
    std::error_code ec;
    fs::remove_all(base, ec);

    return g_failures == 0 ? 0 : 1;
}
