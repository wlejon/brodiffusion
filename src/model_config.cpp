// model_config — diffusers model-directory JSON config loader.
//
// See model_config.h. Uses <filesystem> for path joins / existence checks and
// <fstream> to read files, then brodiffusion::detail::json to parse them.

#include "brodiffusion/model_config.h"

#include "brodiffusion/detail/json.h"

#include <cmath>
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

// Sana ships a DPMSolverMultistepScheduler in flow-prediction mode. A dedicated
// Flow-DPM-Solver scheduler is a later chunk; for now map it onto the existing
// FlowMatchConfig, taking the flow shift from the `flow_shift` key (the
// DPM-Solver config spells it differently than FlowMatchEuler's `shift`).
scheduler::FlowMatchConfig parse_flow_dpm(const json::Value& cfg) {
    scheduler::FlowMatchConfig c;
    c.num_train_timesteps =
        cfg.get_int("num_train_timesteps", c.num_train_timesteps);
    c.shift = cfg.get_float("flow_shift", c.shift);
    return c;
}

// PixArt-Sigma ships a DPMSolverMultistepScheduler in epsilon-prediction mode
// (dpmsolver++, order 2). Distinct from Sana's flow-prediction DPM, which is
// mapped onto FlowMatch above.
scheduler::DPMSolverConfig parse_dpm_solver(const json::Value& cfg) {
    scheduler::DPMSolverConfig c;
    c.num_train_timesteps =
        cfg.get_int("num_train_timesteps", c.num_train_timesteps);
    c.beta_start        = cfg.get_float("beta_start", c.beta_start);
    c.beta_end          = cfg.get_float("beta_end", c.beta_end);
    c.solver_order      = cfg.get_int("solver_order", c.solver_order);
    c.lower_order_final = cfg.get_bool("lower_order_final", c.lower_order_final);
    c.steps_offset      = cfg.get_int("steps_offset", c.steps_offset);
    return c;
}

// Sana-Sprint ships an SCMScheduler (TrigFlow few-step sampler).
scheduler::SCMConfig parse_scm(const json::Value& cfg) {
    scheduler::SCMConfig c;
    c.num_train_timesteps =
        cfg.get_int("num_train_timesteps", c.num_train_timesteps);
    c.sigma_data = cfg.get_float("sigma_data", c.sigma_data);
    return c;
}

void populate_sana(const json::Value& cfg, dit::SanaConfig& out) {
    out.in_channels         = cfg.get_int("in_channels", out.in_channels);
    out.out_channels        = cfg.get_int("out_channels", out.out_channels);
    out.num_layers          = cfg.get_int("num_layers", out.num_layers);
    out.attention_head_dim  = cfg.get_int("attention_head_dim",
                                          out.attention_head_dim);
    out.num_attention_heads = cfg.get_int("num_attention_heads",
                                          out.num_attention_heads);
    out.num_cross_attention_heads =
        cfg.get_int("num_cross_attention_heads", out.num_cross_attention_heads);
    out.cross_attention_head_dim =
        cfg.get_int("cross_attention_head_dim", out.cross_attention_head_dim);
    out.cross_attention_dim =
        cfg.get_int("cross_attention_dim", out.cross_attention_dim);
    out.caption_channels    = cfg.get_int("caption_channels",
                                          out.caption_channels);
    out.mlp_ratio           = cfg.get_float("mlp_ratio", out.mlp_ratio);
    out.patch_size          = cfg.get_int("patch_size", out.patch_size);
    out.sample_size         = cfg.get_int("sample_size", out.sample_size);
    out.attention_bias      = cfg.get_bool("attention_bias", out.attention_bias);
    out.norm_elementwise_affine =
        cfg.get_bool("norm_elementwise_affine", out.norm_elementwise_affine);
    out.norm_eps            = cfg.get_float("norm_eps", out.norm_eps);
    // Sana-Sprint extras. guidance_embeds gates the combined timestep+guidance
    // embedding (and disables CFG); qk_norm is a string in the config
    // ("rms_norm_across_heads") — the only variant Sana ships — mapped to a bool.
    out.guidance_embeds     = cfg.get_bool("guidance_embeds", out.guidance_embeds);
    out.guidance_embeds_scale =
        cfg.get_float("guidance_embeds_scale", out.guidance_embeds_scale);
    out.qk_norm             = !cfg.get_string("qk_norm", "").empty();
}

void populate_pixart(const json::Value& cfg, dit::PixArtConfig& out) {
    out.in_channels         = cfg.get_int("in_channels", out.in_channels);
    out.out_channels        = cfg.get_int("out_channels", out.out_channels);
    out.num_layers          = cfg.get_int("num_layers", out.num_layers);
    out.attention_head_dim  = cfg.get_int("attention_head_dim",
                                          out.attention_head_dim);
    out.num_attention_heads = cfg.get_int("num_attention_heads",
                                          out.num_attention_heads);
    out.cross_attention_dim = cfg.get_int("cross_attention_dim",
                                          out.cross_attention_dim);
    out.caption_channels    = cfg.get_int("caption_channels",
                                          out.caption_channels);
    out.patch_size          = cfg.get_int("patch_size", out.patch_size);
    out.sample_size         = cfg.get_int("sample_size", out.sample_size);
    out.interpolation_scale = cfg.get_int("interpolation_scale",
                                          out.interpolation_scale);
    out.norm_eps            = cfg.get_float("norm_eps", out.norm_eps);
}

void populate_dcae(const json::Value& cfg, dcae::DecoderConfig& out) {
    out.latent_channels = cfg.get_int("latent_channels", out.latent_channels);
    out.image_channels  = cfg.get_int("in_channels", out.image_channels);
    out.attention_head_dim =
        cfg.get_int("attention_head_dim", out.attention_head_dim);
    out.scaling_factor  = cfg.get_float("scaling_factor", out.scaling_factor);

    out.block_out_channels =
        cfg.get_int_array("decoder_block_out_channels", out.block_out_channels);
    out.layers_per_block =
        cfg.get_int_array("decoder_layers_per_block", out.layers_per_block);

    // decoder_block_types: array of strings; EfficientViTBlock -> attention.
    if (const json::Value* bt = cfg.find("decoder_block_types");
        bt && bt->is_array()) {
        std::vector<bool> is_attn;
        for (const auto& e : bt->as_array()) {
            is_attn.push_back(e.as_string() == "EfficientViTBlock");
        }
        out.is_attention = std::move(is_attn);
    }

    // decoder_qkv_multiscales: array of int arrays, one per stage.
    if (const json::Value* qm = cfg.find("decoder_qkv_multiscales");
        qm && qm->is_array()) {
        std::vector<std::vector<int>> scales;
        for (const auto& stage : qm->as_array()) {
            std::vector<int> s;
            if (stage.is_array()) {
                for (const auto& v : stage.as_array()) {
                    s.push_back(static_cast<int>(v.as_number()));
                }
            }
            scales.push_back(std::move(s));
        }
        out.qkv_multiscales = std::move(scales);
    }
}

void populate_gemma(const json::Value& cfg, brolm::gemma::Gemma2Config& out) {
    out.vocab_size          = cfg.get_int("vocab_size", out.vocab_size);
    out.hidden_size         = cfg.get_int("hidden_size", out.hidden_size);
    out.intermediate_size   = cfg.get_int("intermediate_size",
                                          out.intermediate_size);
    out.num_hidden_layers   = cfg.get_int("num_hidden_layers",
                                          out.num_hidden_layers);
    out.num_attention_heads = cfg.get_int("num_attention_heads",
                                          out.num_attention_heads);
    out.num_key_value_heads = cfg.get_int("num_key_value_heads",
                                          out.num_key_value_heads);
    out.head_dim            = cfg.get_int("head_dim", out.head_dim);
    out.rms_norm_eps        = cfg.get_float("rms_norm_eps", out.rms_norm_eps);
    out.rope_theta          = cfg.get_float("rope_theta", out.rope_theta);
    out.tie_word_embeddings =
        cfg.get_bool("tie_word_embeddings", out.tie_word_embeddings);
    out.query_pre_attn_scalar =
        cfg.get_float("query_pre_attn_scalar", out.query_pre_attn_scalar);
    out.sliding_window      = cfg.get_int("sliding_window", out.sliding_window);
    out.attn_logit_softcapping =
        cfg.get_float("attn_logit_softcapping", out.attn_logit_softcapping);
    out.final_logit_softcapping =
        cfg.get_float("final_logit_softcapping", out.final_logit_softcapping);
    out.max_position_embeddings =
        cfg.get_int("max_position_embeddings", out.max_position_embeddings);
}

// Parse a JSON float array (get_int_array has no float sibling).
std::vector<float> get_float_array(const json::Value& cfg, const std::string& key,
                                   std::vector<float> dflt) {
    const json::Value* v = cfg.find(key);
    if (!v || !v->is_array()) return dflt;
    std::vector<float> out;
    for (const auto& e : v->as_array()) {
        out.push_back(static_cast<float>(e.as_number()));
    }
    return out;
}

// Parse a JSON bool array.
std::vector<bool> get_bool_array(const json::Value& cfg, const std::string& key,
                                 std::vector<bool> dflt) {
    const json::Value* v = cfg.find(key);
    if (!v || !v->is_array()) return dflt;
    std::vector<bool> out;
    for (const auto& e : v->as_array()) out.push_back(e.as_bool());
    return out;
}

void populate_krea2_transformer(const json::Value& cfg, dit::Krea2Config& out) {
    out.in_channels          = cfg.get_int("in_channels", out.in_channels);
    out.num_layers           = cfg.get_int("num_layers", out.num_layers);
    out.attention_head_dim   = cfg.get_int("attention_head_dim",
                                           out.attention_head_dim);
    out.num_attention_heads  = cfg.get_int("num_attention_heads",
                                           out.num_attention_heads);
    out.num_key_value_heads  = cfg.get_int("num_key_value_heads",
                                           out.num_key_value_heads);
    out.intermediate_size    = cfg.get_int("intermediate_size",
                                           out.intermediate_size);
    out.timestep_embed_dim   = cfg.get_int("timestep_embed_dim",
                                           out.timestep_embed_dim);
    out.text_hidden_dim      = cfg.get_int("text_hidden_dim", out.text_hidden_dim);
    out.num_text_layers      = cfg.get_int("num_text_layers", out.num_text_layers);
    out.text_num_attention_heads =
        cfg.get_int("text_num_attention_heads", out.text_num_attention_heads);
    out.text_num_key_value_heads =
        cfg.get_int("text_num_key_value_heads", out.text_num_key_value_heads);
    out.text_intermediate_size =
        cfg.get_int("text_intermediate_size", out.text_intermediate_size);
    out.num_layerwise_text_blocks =
        cfg.get_int("num_layerwise_text_blocks", out.num_layerwise_text_blocks);
    out.num_refiner_text_blocks =
        cfg.get_int("num_refiner_text_blocks", out.num_refiner_text_blocks);
    out.axes_dims_rope       = cfg.get_int_array("axes_dims_rope",
                                                 out.axes_dims_rope);
    out.rope_theta           = cfg.get_float("rope_theta", out.rope_theta);
    out.norm_eps             = cfg.get_float("norm_eps", out.norm_eps);
}

void populate_krea2_vae(const json::Value& cfg, vae_qwenimage::Config& out) {
    out.base_dim        = cfg.get_int("base_dim", out.base_dim);
    out.z_dim           = cfg.get_int("z_dim", out.z_dim);
    out.dim_mult        = cfg.get_int_array("dim_mult", out.dim_mult);
    out.num_res_blocks  = cfg.get_int("num_res_blocks", out.num_res_blocks);
    out.attn_scales     = get_float_array(cfg, "attn_scales", out.attn_scales);
    out.temperal_downsample =
        get_bool_array(cfg, "temperal_downsample", out.temperal_downsample);
    out.dropout         = cfg.get_float("dropout", out.dropout);
    out.input_channels  = cfg.get_int("input_channels", out.input_channels);
    out.latents_mean    = get_float_array(cfg, "latents_mean", out.latents_mean);
    out.latents_std     = get_float_array(cfg, "latents_std", out.latents_std);
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
    } else if (contains_ci(class_name, "Sana")) {
        out.model_class = ModelClass::Sana;
    } else if (contains_ci(class_name, "PixArt")) {
        out.model_class = ModelClass::PixArt;
    } else if (contains_ci(class_name, "Krea2")) {
        out.model_class = ModelClass::Krea2;
    } else {
        out.model_class = ModelClass::Unknown;
    }

    const bool is_flux   = (out.model_class == ModelClass::Flux);
    const bool is_sana   = (out.model_class == ModelClass::Sana);
    const bool is_pixart = (out.model_class == ModelClass::PixArt);
    const bool is_krea2  = (out.model_class == ModelClass::Krea2);

    // --- Krea 2: transformer + VAE + Qwen3-VL text encoder configs ---
    if (is_krea2) {
        // Raw vs Turbo: both ship _class_name "Krea2Pipeline"; is_distilled
        // separates them (false = Raw / real CFG, true = Turbo / no CFG).
        out.krea2.is_distilled = index.get_bool("is_distilled", false);
        out.krea2.patch_size   = index.get_int("patch_size", 2);

        const fs::path tf_cfg = root / "transformer" / "config.json";
        if (fs::exists(tf_cfg)) {
            populate_krea2_transformer(parse_file(tf_cfg), out.krea2.transformer);
        }
        const fs::path te_cfg = root / "text_encoder" / "config.json";
        if (fs::exists(te_cfg)) {
            out.krea2.text = brolm::qwen3vl::Qwen3VLConfig::load(te_cfg.string());
        }
    }

    // --- unet/config.json (StableDiffusion only) ---
    if (!is_flux && !is_sana && !is_pixart && !is_krea2) {
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

    // --- transformer/config.json + text_encoder/config.json (Sana only) ---
    if (is_sana) {
        const fs::path tf_cfg = root / "transformer" / "config.json";
        if (fs::exists(tf_cfg)) {
            populate_sana(parse_file(tf_cfg), out.sana);
        }
        const fs::path gemma_cfg = root / "text_encoder" / "config.json";
        if (fs::exists(gemma_cfg)) {
            populate_gemma(parse_file(gemma_cfg), out.gemma);
        }
    }

    // --- transformer/config.json + text_encoder/config.json (PixArt only) ---
    if (is_pixart) {
        const fs::path tf_cfg = root / "transformer" / "config.json";
        if (fs::exists(tf_cfg)) {
            populate_pixart(parse_file(tf_cfg), out.pixart);
        }
        // PixArt's text encoder is T5-XXL (text_encoder/config.json is a T5
        // config). The encoder weights aren't bundled here — but the config is,
        // so the T5 hyper-params come from it.
        const fs::path t5_cfg = root / "text_encoder" / "config.json";
        if (fs::exists(t5_cfg)) {
            populate_t5(parse_file(t5_cfg), out.t5);
        }
        out.t5_max_length = 300;  // PixArt-Sigma caption cap (alpha was 120)
    }

    // --- vae/config.json (AutoencoderDC for Sana, AutoencoderKLQwenImage for
    //     Krea 2, AutoencoderKL otherwise) ---
    {
        const fs::path vae_cfg = root / "vae" / "config.json";
        if (fs::exists(vae_cfg)) {
            if (is_sana) {
                populate_dcae(parse_file(vae_cfg), out.dcae);
            } else if (is_krea2) {
                populate_krea2_vae(parse_file(vae_cfg), out.krea2.vae);
            } else {
                populate_vae(parse_file(vae_cfg), out.vae);
            }
        }
    }

    // --- text_encoder/config.json (CLIP; Sana's Gemma-2, PixArt's T5, and
    //     Krea 2's Qwen3-VL are handled above) ---
    if (!is_sana && !is_pixart && !is_krea2) {
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
                scheduler::FlowMatchConfig fm = parse_flow_match(cfg);
                // Krea 2 Turbo (distilled) uses a FIXED dynamic-shift mu = max_shift
                // instead of the resolution-derived one. A dynamic shift with a
                // fixed mu is algebraically identical to a static shift = exp(mu)
                // (both give sigma' = e^mu·s / (1 + (e^mu-1)·s)), so realise it as
                // a static shift and leave the scheduler's dynamic-shifting path —
                // and every other model's behaviour — untouched.
                if (is_krea2 && out.krea2.is_distilled) {
                    fm.use_dynamic_shifting = false;
                    fm.shift = std::exp(fm.max_shift);
                }
                out.scheduler = fm;
            } else if (sched_class == "DPMSolverMultistepScheduler") {
                // Two flavours share this class name: PixArt-Sigma's epsilon
                // DPM-Solver++ (a real discrete-time DPM) and Sana's
                // flow-prediction DPM (mapped onto FlowMatch for now). Branch
                // on prediction_type.
                const std::string pred =
                    cfg.get_string("prediction_type", "epsilon");
                if (pred == "epsilon" || pred == "v_prediction") {
                    out.scheduler = parse_dpm_solver(cfg);
                } else {
                    out.scheduler = parse_flow_dpm(cfg);
                }
            } else if (sched_class == "LCMScheduler") {
                out.scheduler = parse_lcm(cfg);
            } else if (sched_class == "SCMScheduler") {
                out.scheduler = parse_scm(cfg);
            } else {
                out.scheduler = parse_ddim(cfg);
            }
        } else {
            // Missing scheduler config: default per model class.
            if (is_flux || is_sana || is_krea2) {
                out.scheduler = scheduler::FlowMatchConfig{};
            } else if (is_pixart) {
                out.scheduler = scheduler::DPMSolverConfig{};
            } else {
                out.scheduler = scheduler::DDIMConfig{};
            }
        }
    }

    return out;
}

}  // namespace brodiffusion
