// model_config — diffusers model-directory JSON config loader.
//
// See model_config.h. Uses <filesystem> for path joins / existence checks and
// <fstream> to read files, then brodiffusion::detail::json to parse them.

#include "brodiffusion/model_config.h"

#include "brodiffusion/detail/json.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace brodiffusion {

namespace {

namespace fs = std::filesystem;
namespace json = brodiffusion::detail::json;

// Read an entire file into a string. Throws on failure.
std::string read_file(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("model_config: cannot open '"
                                 + path.string() + "'");
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Parse a JSON file into a Value. Throws on missing file or malformed JSON.
json::Value parse_file(const fs::path& path) {
    std::string text = read_file(path);
    try {
        return json::parse(text);
    } catch (const std::exception& e) {
        throw std::runtime_error("model_config: parsing '" + path.string()
                                 + "': " + e.what());
    }
}

bool contains_ci(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

void populate_unet(const json::Value& cfg, unet::UNetConfig& out) {
    out.in_channels        = cfg.get_int("in_channels", out.in_channels);
    out.out_channels       = cfg.get_int("out_channels", out.out_channels);
    out.block_out_channels = cfg.get_int_array("block_out_channels",
                                               out.block_out_channels);
    out.layers_per_block   = cfg.get_int("layers_per_block", out.layers_per_block);
    out.norm_num_groups    = cfg.get_int("norm_num_groups", out.norm_num_groups);
    out.cross_attention_dim =
        cfg.get_int("cross_attention_dim", out.cross_attention_dim);

    // attention_head_dim may be stored as an int or an array; take element 0.
    if (const json::Value* ah = cfg.find("attention_head_dim");
        ah && !ah->is_null()) {
        if (ah->is_array()) {
            const auto& arr = ah->as_array();
            if (!arr.empty()) {
                out.attention_head_dim =
                    static_cast<int>(arr.front().as_number());
            }
        } else {
            out.attention_head_dim = static_cast<int>(ah->as_number());
        }
    }

    // time_cond_proj_dim: JSON null -> leave at default (0); an int -> use it.
    out.time_cond_proj_dim =
        cfg.get_int("time_cond_proj_dim", out.time_cond_proj_dim);
}

void populate_vae(const json::Value& cfg, vae::DecoderConfig& out) {
    // diffusers VAE in_channels/out_channels are *image* channels; our
    // DecoderConfig.in_channels is the *latent* channel count -> latent_channels.
    out.in_channels        = cfg.get_int("latent_channels", out.in_channels);
    out.out_channels       = cfg.get_int("out_channels", out.out_channels);
    out.block_out_channels = cfg.get_int_array("block_out_channels",
                                               out.block_out_channels);
    out.layers_per_block   = cfg.get_int("layers_per_block", out.layers_per_block);
    out.norm_num_groups    = cfg.get_int("norm_num_groups", out.norm_num_groups);
    out.scaling_factor     = cfg.get_float("scaling_factor", out.scaling_factor);
    out.shift_factor       = cfg.get_float("shift_factor", out.shift_factor);
    out.force_upcast       = cfg.get_bool("force_upcast", out.force_upcast);
}

void populate_text_encoder(const json::Value& cfg, brolm::clip::TextEncoderConfig& out) {
    out.hidden_dim       = cfg.get_int("hidden_size", out.hidden_dim);
    out.intermediate_dim = cfg.get_int("intermediate_size", out.intermediate_dim);
    out.num_heads        = cfg.get_int("num_attention_heads", out.num_heads);
    out.num_layers       = cfg.get_int("num_hidden_layers", out.num_layers);
    out.max_position     = cfg.get_int("max_position_embeddings", out.max_position);
    out.vocab_size       = cfg.get_int("vocab_size", out.vocab_size);
    out.layer_norm_eps   = cfg.get_float("layer_norm_eps", out.layer_norm_eps);
    out.eos_token_id     = cfg.get_int("eos_token_id", out.eos_token_id);
}

scheduler::DDIMConfig parse_ddim(const json::Value& cfg) {
    scheduler::DDIMConfig c;
    c.num_train_timesteps =
        cfg.get_int("num_train_timesteps", c.num_train_timesteps);
    c.beta_start       = cfg.get_float("beta_start", c.beta_start);
    c.beta_end         = cfg.get_float("beta_end", c.beta_end);
    c.steps_offset     = cfg.get_int("steps_offset", c.steps_offset);
    c.set_alpha_to_one = cfg.get_bool("set_alpha_to_one", c.set_alpha_to_one);
    return c;
}

scheduler::LCMConfig parse_lcm(const json::Value& cfg) {
    scheduler::LCMConfig c;
    c.num_train_timesteps =
        cfg.get_int("num_train_timesteps", c.num_train_timesteps);
    c.beta_start = cfg.get_float("beta_start", c.beta_start);
    c.beta_end   = cfg.get_float("beta_end", c.beta_end);
    c.original_inference_steps =
        cfg.get_int("original_inference_steps", c.original_inference_steps);
    c.timestep_scaling = cfg.get_float("timestep_scaling", c.timestep_scaling);
    c.set_alpha_to_one = cfg.get_bool("set_alpha_to_one", c.set_alpha_to_one);
    return c;
}

void populate_flux(const json::Value& cfg, dit::FluxConfig& out) {
    out.in_channels         = cfg.get_int("in_channels", out.in_channels);
    out.num_layers          = cfg.get_int("num_layers", out.num_layers);
    out.num_single_layers   = cfg.get_int("num_single_layers",
                                          out.num_single_layers);
    out.attention_head_dim  = cfg.get_int("attention_head_dim",
                                          out.attention_head_dim);
    out.num_attention_heads = cfg.get_int("num_attention_heads",
                                          out.num_attention_heads);
    out.joint_attention_dim = cfg.get_int("joint_attention_dim",
                                          out.joint_attention_dim);
    out.pooled_projection_dim = cfg.get_int("pooled_projection_dim",
                                            out.pooled_projection_dim);
    out.guidance_embeds     = cfg.get_bool("guidance_embeds",
                                           out.guidance_embeds);
    out.axes_dims_rope      = cfg.get_int_array("axes_dims_rope",
                                               out.axes_dims_rope);
}

void populate_t5(const json::Value& cfg, brolm::t5::T5Config& out) {
    out.d_model    = cfg.get_int("d_model", out.d_model);
    out.d_ff       = cfg.get_int("d_ff", out.d_ff);
    out.d_kv       = cfg.get_int("d_kv", out.d_kv);
    out.num_heads  = cfg.get_int("num_heads", out.num_heads);
    out.num_layers = cfg.get_int("num_layers", out.num_layers);
    out.relative_attention_num_buckets =
        cfg.get_int("relative_attention_num_buckets",
                    out.relative_attention_num_buckets);
    out.relative_attention_max_distance =
        cfg.get_int("relative_attention_max_distance",
                    out.relative_attention_max_distance);
    out.vocab_size = cfg.get_int("vocab_size", out.vocab_size);
    out.layer_norm_eps =
        cfg.get_float("layer_norm_epsilon", out.layer_norm_eps);
}

scheduler::FlowMatchConfig parse_flow_match(const json::Value& cfg) {
    scheduler::FlowMatchConfig c;
    c.num_train_timesteps =
        cfg.get_int("num_train_timesteps", c.num_train_timesteps);
    c.shift = cfg.get_float("shift", c.shift);
    c.use_dynamic_shifting =
        cfg.get_bool("use_dynamic_shifting", c.use_dynamic_shifting);
    c.base_shift = cfg.get_float("base_shift", c.base_shift);
    c.max_shift  = cfg.get_float("max_shift", c.max_shift);
    c.base_image_seq_len =
        cfg.get_int("base_image_seq_len", c.base_image_seq_len);
    c.max_image_seq_len =
        cfg.get_int("max_image_seq_len", c.max_image_seq_len);
    return c;
}

}  // namespace

ModelConfig load_model_config(const std::string& model_dir) {
    ModelConfig out;
    out.model_dir = model_dir;

    const fs::path root(model_dir);

    // --- model_index.json (required) ---
    const fs::path index_path = root / "model_index.json";
    if (!fs::exists(index_path)) {
        throw std::runtime_error("model_config: missing model_index.json in '"
                                 + model_dir + "'");
    }
    const json::Value index = parse_file(index_path);
    const std::string class_name = index.get_string("_class_name", "");
    if (contains_ci(class_name, "StableDiffusion")) {
        out.model_class = ModelClass::StableDiffusion;
    } else if (contains_ci(class_name, "Flux")) {
        out.model_class = ModelClass::Flux;
    } else {
        out.model_class = ModelClass::Unknown;
    }

    const bool is_flux = (out.model_class == ModelClass::Flux);

    // --- unet/config.json (StableDiffusion only) ---
    if (!is_flux) {
        const fs::path unet_cfg = root / "unet" / "config.json";
        if (fs::exists(unet_cfg)) {
            populate_unet(parse_file(unet_cfg), out.unet);
        }
    }

    // --- transformer/config.json + text_encoder_2/config.json (Flux only) ---
    if (is_flux) {
        const fs::path tf_cfg = root / "transformer" / "config.json";
        if (fs::exists(tf_cfg)) {
            populate_flux(parse_file(tf_cfg), out.flux);
        }
        const fs::path t5_cfg = root / "text_encoder_2" / "config.json";
        if (fs::exists(t5_cfg)) {
            populate_t5(parse_file(t5_cfg), out.t5);
        }
        // T5 sequence length: prefer tokenizer_2's model_max_length when present
        // and sane (<= 512); leave the 512 default otherwise.
        const fs::path tok2_cfg =
            root / "tokenizer_2" / "tokenizer_config.json";
        if (fs::exists(tok2_cfg)) {
            const json::Value tc = parse_file(tok2_cfg);
            const int mml = tc.get_int("model_max_length", out.t5_max_length);
            if (mml > 0 && mml <= 512) out.t5_max_length = mml;
        }
    }

    // --- vae/config.json (always) ---
    {
        const fs::path vae_cfg = root / "vae" / "config.json";
        if (fs::exists(vae_cfg)) {
            populate_vae(parse_file(vae_cfg), out.vae);
        }
    }

    // --- text_encoder/config.json (when present) ---
    {
        const fs::path te_cfg = root / "text_encoder" / "config.json";
        if (fs::exists(te_cfg)) {
            populate_text_encoder(parse_file(te_cfg), out.text_encoder);
        }
    }

    // --- scheduler/scheduler_config.json ---
    {
        const fs::path sched_cfg =
            root / "scheduler" / "scheduler_config.json";
        if (fs::exists(sched_cfg)) {
            const json::Value cfg = parse_file(sched_cfg);
            const std::string sched_class = cfg.get_string("_class_name", "");
            if (sched_class == "FlowMatchEulerDiscreteScheduler") {
                out.scheduler = parse_flow_match(cfg);
            } else if (sched_class == "LCMScheduler") {
                out.scheduler = parse_lcm(cfg);
            } else {
                out.scheduler = parse_ddim(cfg);
            }
        } else {
            // Missing scheduler config: default per model class.
            if (is_flux) {
                out.scheduler = scheduler::FlowMatchConfig{};
            } else {
                out.scheduler = scheduler::DDIMConfig{};
            }
        }
    }

    return out;
}

}  // namespace brodiffusion
