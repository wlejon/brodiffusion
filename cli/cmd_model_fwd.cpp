// Single-component forward drivers.
//
// Each of these runs ONE sub-module (text encoder, VAE, DiT) on fixed inputs
// read from raw little-endian float32 files and dumps the result the same way,
// so the matching scripts/*_parity.sh can diff it against a diffusers
// reference. They are deliberately absent from usage() — they exist for the
// parity gates, not for end users.

#include "commands.h"

#include "brodiffusion/detail/compute.h"
#include "brodiffusion/detail/jit_fusion.h"
#include "brodiffusion/detail/json.h"
#include "brodiffusion/dit/krea2.h"
#include "brodiffusion/dit/pixart.h"
#include "brodiffusion/dit/qwenimage21.h"
#include "brodiffusion/krea2_text.h"
#include "brodiffusion/qwenimage21_text.h"
#include "brodiffusion/vae_qwenimage.h"
#include "brodiffusion/vae_qwenimage21.h"
#include "brodiffusion/detail/safetensors_dir.h"

#include "brolm/qwen3vl_config.h"
#include "brolm/qwen3vl_text.h"
#include "brolm/qwen3vl_tokenizer.h"
#include "brolm/t5.h"
#include "brolm/tokenizer_t5.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace brodiffusion::cli {

namespace st = brotensor::safetensors;

namespace {

// Parse a Krea2Transformer2DModel config.json into a Krea2Config.
brodiffusion::dit::Krea2Config load_krea2_config(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open config: " + path);
    std::string text((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    namespace j = brodiffusion::detail::json;
    j::Value v = j::parse(text);
    brodiffusion::dit::Krea2Config c;
    c.in_channels = v.get_int("in_channels", c.in_channels);
    c.num_layers = v.get_int("num_layers", c.num_layers);
    c.attention_head_dim = v.get_int("attention_head_dim", c.attention_head_dim);
    c.num_attention_heads = v.get_int("num_attention_heads", c.num_attention_heads);
    c.num_key_value_heads = v.get_int("num_key_value_heads", c.num_key_value_heads);
    c.intermediate_size = v.get_int("intermediate_size", c.intermediate_size);
    c.timestep_embed_dim = v.get_int("timestep_embed_dim", c.timestep_embed_dim);
    c.text_hidden_dim = v.get_int("text_hidden_dim", c.text_hidden_dim);
    c.num_text_layers = v.get_int("num_text_layers", c.num_text_layers);
    c.text_num_attention_heads =
        v.get_int("text_num_attention_heads", c.text_num_attention_heads);
    c.text_num_key_value_heads =
        v.get_int("text_num_key_value_heads", c.text_num_key_value_heads);
    c.text_intermediate_size =
        v.get_int("text_intermediate_size", c.text_intermediate_size);
    c.num_layerwise_text_blocks =
        v.get_int("num_layerwise_text_blocks", c.num_layerwise_text_blocks);
    c.num_refiner_text_blocks =
        v.get_int("num_refiner_text_blocks", c.num_refiner_text_blocks);
    c.axes_dims_rope = v.get_int_array("axes_dims_rope", c.axes_dims_rope);
    c.rope_theta = v.get_float("rope_theta", c.rope_theta);
    c.norm_eps = v.get_float("norm_eps", c.norm_eps);
    return c;
}

// Parse a QwenImage21Transformer2DModel config.json into a QwenImage21Config.
brodiffusion::dit::QwenImage21Config load_qi21_config(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open config: " + path);
    std::string text((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    namespace j = brodiffusion::detail::json;
    j::Value v = j::parse(text);
    brodiffusion::dit::QwenImage21Config c;
    c.patch_size = v.get_int("patch_size", c.patch_size);
    c.in_channels = v.get_int("in_channels", c.in_channels);
    c.out_channels = v.get_int("out_channels", c.out_channels);
    c.num_layers = v.get_int("num_layers", c.num_layers);
    c.attention_head_dim = v.get_int("attention_head_dim", c.attention_head_dim);
    c.num_attention_heads = v.get_int("num_attention_heads", c.num_attention_heads);
    c.context_in_dim = v.get_int("context_in_dim", c.context_in_dim);
    c.mlp_ratio = v.get_int("mlp_ratio", c.mlp_ratio);
    c.axes_dims_rope = v.get_int_array("axes_dims_rope", c.axes_dims_rope);
    c.eps = v.get_float("eps", c.eps);
    c.causal_condition = v.get_bool("causal_condition", c.causal_condition);
    return c;
}

// Open a transformer component directory's safetensors (single file or the
// shard set named by its .index.json).
std::vector<st::File> open_transformer_shards(const std::string& wd) {
    std::vector<st::File> files;
    const std::string index = wd + "/diffusion_pytorch_model.safetensors.index.json";
    std::ifstream idxf(index, std::ios::binary);
    if (idxf) {
        std::string text((std::istreambuf_iterator<char>(idxf)),
                         std::istreambuf_iterator<char>());
        namespace j = brodiffusion::detail::json;
        j::Value v = j::parse(text);
        const auto& wm = v.at("weight_map");
        std::vector<std::string> names;
        for (const auto& m : wm.as_object()) {
            const std::string s = m.second.as_string();
            if (std::find(names.begin(), names.end(), s) == names.end())
                names.push_back(s);
        }
        std::sort(names.begin(), names.end());
        files.reserve(names.size());
        for (const std::string& n : names) files.push_back(st::File::open(wd + "/" + n));
    } else {
        files.push_back(st::File::open(wd + "/diffusion_pytorch_model.safetensors"));
    }
    return files;
}

}  // namespace

// Load the T5-XXL encoder, encode a prompt, run one forward, print stats.
// Doubles as the load/works check for the t5-xxl weight download and the
// manual FP16-vs-INT8 comparison driver (run once with and once without
// --quantize and diff the printed mean / L2 norm).
int run_t5(int argc, char** argv) {
    const char* weights_path = arg_after(argc, argv, "--weights");
    const char* tok_path     = arg_after(argc, argv, "--tokenizer");
    const char* prompt       = arg_after(argc, argv, "--prompt");
    const char* maxlen_s     = arg_after(argc, argv, "--max-length");
    const char* dump_path    = arg_after(argc, argv, "--out");
    bool quantize = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--quantize") == 0) quantize = true;
    }
    if (!weights_path || !tok_path || !prompt) {
        std::fprintf(stderr,
            "t5: --weights, --tokenizer, --prompt are required\n");
        return 2;
    }
    const int max_length = maxlen_s ? std::atoi(maxlen_s) : 128;

    brotensor::init();

    auto tok = brolm::t5::Tokenizer::load(tok_path);
    std::vector<std::int32_t> ids = tok.encode(prompt, max_length);

    brolm::t5::T5Config cfg;
    cfg.quantize_weights = quantize;
    brolm::t5::TextEncoder enc(cfg);

    std::printf("Loading T5-XXL weights: %s%s\n", weights_path,
                quantize ? "  (INT8 W8A16)" : "");
    auto file = st::File::open(weights_path);

    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now();
    enc.load_weights(file, "");
    brotensor::sync_all();
    auto t1 = clk::now();
    std::printf("Loaded (%s) in %.2f s\n",
                quantize ? "quantized to INT8" : "FP16",
                std::chrono::duration<double>(t1 - t0).count());

    brotensor::Tensor out;
    auto t2 = clk::now();
    enc.forward(ids.data(), static_cast<int>(ids.size()), out);
    brotensor::sync_all();
    auto t3 = clk::now();

    const int L = out.rows;
    const int D = out.cols;
    std::vector<float> vals;
    if (out.dtype == brotensor::Dtype::FP16) {
        std::vector<std::uint16_t> bits(static_cast<std::size_t>(L) * D);
        out.copy_to_host_fp16(bits.data());
        brotensor::sync_all();
        vals.resize(bits.size());
        for (std::size_t i = 0; i < bits.size(); ++i) {
            vals[i] = brotensor::fp16_bits_to_fp32(bits[i]);
        }
    } else if (out.dtype == brotensor::Dtype::BF16) {
        std::vector<std::uint16_t> bits(static_cast<std::size_t>(L) * D);
        out.copy_to_host_bf16(bits.data());
        brotensor::sync_all();
        vals.resize(bits.size());
        for (std::size_t i = 0; i < bits.size(); ++i) {
            vals[i] = brotensor::bf16_bits_to_fp32(bits[i]);
        }
    } else {
        vals = out.to_host_vector();
    }

    int nonfinite = 0;
    double sum = 0.0, sumsq = 0.0;
    for (float v : vals) {
        if (!std::isfinite(v)) { ++nonfinite; continue; }
        sum += v;
        sumsq += static_cast<double>(v) * static_cast<double>(v);
    }
    const double mean = vals.empty() ? 0.0 : sum / static_cast<double>(vals.size());
    const double l2   = std::sqrt(sumsq);
    std::printf("Output: (%d, %d)  tokens=%d  forward=%.1f ms\n",
                L, D, static_cast<int>(ids.size()),
                std::chrono::duration<double, std::milli>(t3 - t2).count());
    std::printf("  non-finite : %d\n", nonfinite);
    std::printf("  mean       : %.6f\n", mean);
    std::printf("  L2 norm    : %.4f\n", l2);
    if (vals.size() >= 5) {
        std::printf("  first 5    : %.5f %.5f %.5f %.5f %.5f\n",
                    vals[0], vals[1], vals[2], vals[3], vals[4]);
    }
    if (dump_path) {
        std::ofstream of(dump_path, std::ios::binary | std::ios::trunc);
        of.write(reinterpret_cast<const char*>(vals.data()),
                 static_cast<std::streamsize>(vals.size() * sizeof(float)));
        std::printf("  dumped (%d,%d) to %s\n", L, D, dump_path);
    }
    return nonfinite == 0 ? 0 : 1;
}

// Run ONE PixArtDenoiser forward on fixed inputs read from raw float32 files,
// dump the epsilon. Diffed against scripts/pixart_ref.py.
int run_pixart_fwd(int argc, char** argv) {
    const char* w  = arg_after(argc, argv, "--weights");
    const char* lp = arg_after(argc, argv, "--latent");
    const char* cp = arg_after(argc, argv, "--ctx");
    const char* op = arg_after(argc, argv, "--out");
    const char* ts = arg_after(argc, argv, "--t");
    const char* Hs = arg_after(argc, argv, "--H");
    const char* Ws = arg_after(argc, argv, "--W");
    const char* Ls = arg_after(argc, argv, "--L");
    if (!w || !lp || !cp || !op || !ts || !Hs || !Ws || !Ls) {
        std::fprintf(stderr, "pixart-fwd: need --weights --latent --ctx --out "
                             "--t --H --W --L\n");
        return 2;
    }
    const int H = std::atoi(Hs), W = std::atoi(Ws), L = std::atoi(Ls);
    const float t = static_cast<float>(std::atof(ts));
    brotensor::init();
    brodiffusion::dit::PixArtConfig cfg;
    brodiffusion::dit::PixArtDenoiser den(cfg);
    auto f = st::File::open(w);
    den.load_weights(f, "");
    den.finalize_weights();

    auto lat_h = load_latent_f32(lp, cfg.in_channels * H * W);
    auto ctx_h = load_latent_f32(cp, L * cfg.caption_channels);
    brotensor::Tensor lat =
        brotensor::Tensor::from_host(lat_h.data(), 1, cfg.in_channels * H * W)
            .to(brotensor::default_device());
    brotensor::Tensor ctx =
        brotensor::Tensor::from_host(ctx_h.data(), L, cfg.caption_channels)
            .to(brotensor::default_device());
    // Cast latent to the denoiser compute dtype (forward also handles this).
    brodiffusion::Conditioning cond;
    cond.text_embeddings = ctx;
    cond.has_uncond = false;
    auto prep = den.prepare(cond);
    brotensor::Tensor out;
    den.forward(lat, H, W, t, prep, brodiffusion::Branch::Cond, out);
    brotensor::sync_all();
    dump_latent_f32(op, out);
    std::printf("pixart-fwd: wrote epsilon (%d,%d) to %s\n", out.rows, out.cols, op);
    return 0;
}

// Run ONE Qwen-Image VAE (Krea 2) decode on a fixed latent read from a raw
// float32 file, dump the RGB output. Diffed against scripts/krea2_vae_ref.py.
int run_krea2_vae_fwd(int argc, char** argv) {
    const char* w  = arg_after(argc, argv, "--weights");
    const char* lp = arg_after(argc, argv, "--latent");
    const char* op = arg_after(argc, argv, "--out");
    const char* Hs = arg_after(argc, argv, "--H");
    const char* Ws = arg_after(argc, argv, "--W");
    if (!w || !lp || !op || !Hs || !Ws) {
        std::fprintf(stderr, "krea2-vae-fwd: need --weights --latent --out --H --W\n");
        return 2;
    }
    const int H = std::atoi(Hs), W = std::atoi(Ws);
    brotensor::init();

    namespace vq = brodiffusion::vae_qwenimage;
    vq::Config cfg;
    vq::Decoder dec(cfg);
    auto f = st::File::open(w);
    dec.load_weights(f, "");

    auto lat_h = load_latent_f32(lp, cfg.z_dim * H * W);
    brotensor::Tensor lat =
        brotensor::Tensor::from_host(lat_h.data(), 1, cfg.z_dim * H * W)
            .to(brotensor::default_device());

    brotensor::Tensor out;
    dec.decode(lat, H, W, out);
    brotensor::sync_all();

    // Normalize to FP32 before dumping — force_upcast may put `out` at BF16
    // on a CUDA backend.
    if (out.dtype != brotensor::Dtype::FP32) {
        brotensor::Tensor out_f32;
        brotensor::cast(out, out_f32, brotensor::Dtype::FP32);
        brotensor::sync_all();
        out = std::move(out_f32);
    }
    dump_latent_f32(op, out);
    std::printf("krea2-vae-fwd: wrote image (%d,%d) to %s\n", out.rows, out.cols, op);
    return 0;
}

// One Qwen-Image 2.1 VAE pass on raw float32 files.
//   decode: --weights <vae.safetensors> --latent <f32> --out <f32> --H --W
//           (latent (z_dim,H,W) raw pipeline scale -> RGBA (4,16H,16W))
//   encode: --weights <vae.safetensors> --image <f32> --out <f32> --H --W
//           (RGBA (4,H,W) in [-1,1] -> latent (z_dim,H/16,W/16), mean sample)
// Diffed against scripts/qwenimage21_vae_ref.py.
int run_qi21_vae_fwd(int argc, char** argv) {
    const char* w  = arg_after(argc, argv, "--weights");
    const char* lp = arg_after(argc, argv, "--latent");
    const char* ip = arg_after(argc, argv, "--image");
    const char* op = arg_after(argc, argv, "--out");
    const char* Hs = arg_after(argc, argv, "--H");
    const char* Ws = arg_after(argc, argv, "--W");
    if (!w || (!lp && !ip) || !op || !Hs || !Ws) {
        std::fprintf(stderr,
            "qi21-vae-fwd: need --weights (--latent | --image) --out --H --W\n");
        return 2;
    }
    const int H = std::atoi(Hs), W = std::atoi(Ws);
    brotensor::init();

    namespace vq = brodiffusion::vae_qwenimage21;
    vq::Config cfg;
    auto f = st::File::open(w);
    brotensor::Tensor out;
    if (lp) {
        vq::Decoder dec(cfg);
        dec.load_weights(f, "");
        auto lat_h = load_latent_f32(lp, cfg.z_dim * H * W);
        brotensor::Tensor lat =
            brotensor::Tensor::from_host(lat_h.data(), 1, cfg.z_dim * H * W)
                .to(brotensor::default_device());
        dec.decode(lat, H, W, out);
    } else {
        vq::Encoder enc(cfg);
        enc.load_weights(f, "");
        auto img_h = load_latent_f32(ip, cfg.in_channels * H * W);
        brotensor::Tensor img =
            brodiffusion::detail::upload_host(img_h.data(), 1, cfg.in_channels * H * W);
        enc.encode(img, H, W, nullptr, out);
    }
    brotensor::sync_all();
    if (out.dtype != brotensor::Dtype::FP32) {
        brotensor::Tensor out_f32;
        brotensor::cast(out, out_f32, brotensor::Dtype::FP32);
        brotensor::sync_all();
        out = std::move(out_f32);
    }
    dump_latent_f32(op, out);
    std::printf("qi21-vae-fwd: wrote (%d,%d) to %s\n", out.rows, out.cols, op);
    return 0;
}

// Run the Krea 2 text-conditioning pathway (Qwen3-VL tapped hidden states →
// fusion input tensors) for one prompt and dump prompt_embeds (and optionally
// the validity mask) as raw float32. Diffed against scripts/krea2_text_ref.py.
int run_krea2_text_fwd(int argc, char** argv) {
    const char* wdir = arg_after(argc, argv, "--weights-dir");
    const char* tdir = arg_after(argc, argv, "--tokenizer-dir");
    const char* prompt = arg_after(argc, argv, "--prompt");
    const char* op = arg_after(argc, argv, "--out");
    const char* mp = arg_after(argc, argv, "--mask-out");
    if (!wdir || !tdir || !prompt || !op) {
        std::fprintf(stderr,
            "krea2-text-fwd: need --weights-dir --tokenizer-dir --prompt --out "
            "[--mask-out]\n");
        return 2;
    }
    brotensor::init();

    const std::string wd = wdir, td = tdir;
    auto tok = brolm::qwen3vl::Tokenizer::load(td + "/vocab.json",
                                               td + "/merges.txt");
    auto cfg = brolm::qwen3vl::Qwen3VLConfig::load(wd + "/text_encoder/config.json");
    brolm::qwen3vl::TextModel model(cfg.text);
    auto f = st::File::open(wd + "/text_encoder/model.safetensors");
    model.load_weights(f, "language_model.");

    auto cond = brodiffusion::krea2::encode_prompt(tok, model, prompt);
    brotensor::sync_all();

    dump_latent_f32(op, cond.prompt_embeds);
    std::printf("krea2-text-fwd: wrote prompt_embeds (%d,%d) to %s\n",
                cond.prompt_embeds.rows, cond.prompt_embeds.cols, op);
    if (mp) {
        dump_latent_f32(mp, cond.prompt_embeds_mask);
        std::printf("krea2-text-fwd: wrote mask (%d,%d) to %s\n",
                    cond.prompt_embeds_mask.rows, cond.prompt_embeds_mask.cols, mp);
    }
    return 0;
}

// Run the Qwen-Image 2.1 text-conditioning pathway (prompt -> the Qwen3-VL-8B
// last-hidden-state rows the DiT's txt_in consumes) and dump the embeddings,
// the validity mask and the token ids. Diffed against
// scripts/qwenimage21_text_ref.py.
int run_qi21_text_fwd(int argc, char** argv) {
    const char* wdir = arg_after(argc, argv, "--weights-dir");
    const char* tdir = arg_after(argc, argv, "--tokenizer-dir");
    const char* prompt = arg_after(argc, argv, "--prompt");
    const char* op = arg_after(argc, argv, "--out");
    const char* mp = arg_after(argc, argv, "--mask-out");
    const char* ip = arg_after(argc, argv, "--ids-out");
    bool quantize = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--quantize") == 0) quantize = true;
    }
    if (!wdir || !prompt || !op) {
        std::fprintf(stderr,
            "qi21-text-fwd: need --weights-dir --prompt --out "
            "[--tokenizer-dir D] [--mask-out F] [--ids-out F] [--quantize]\n");
        return 2;
    }
    brotensor::init();

    const std::string wd = wdir;
    // The tokenizer assets live under processor/ in a Qwen-Image 2.1 dir
    // (there is no tokenizer/ subdirectory).
    const std::string td = tdir ? std::string(tdir) : wd + "/processor";

    auto tok = brolm::qwen3vl::Tokenizer::load(td + "/vocab.json",
                                               td + "/merges.txt");
    auto cfg = brolm::qwen3vl::Qwen3VLConfig::load(wd + "/text_encoder/config.json");
    cfg.text.quantize_weights = quantize;
    // We only ever read hidden states from this backbone, never logits, so the
    // untied lm_head (151936x4096 — 1.2 GiB at BF16) is dead weight. Claiming
    // tied embeddings makes brolm skip loading it entirely.
    cfg.text.tie_word_embeddings = true;
    brolm::qwen3vl::TextModel model(cfg.text);

    auto te_files = brodiffusion::detail::open_component_files(
        wd + "/text_encoder");
    std::vector<const st::File*> te_ptrs;
    for (const auto& f : te_files) te_ptrs.push_back(&f);
    model.load_weights(te_ptrs, "model.language_model.");
    brotensor::sync_all();

    auto cond = brodiffusion::qwenimage21::encode_prompt(tok, model, prompt);
    brotensor::sync_all();

    brotensor::Tensor emb = cond.embeds;
    if (emb.dtype != brotensor::Dtype::FP32) {
        brotensor::Tensor f32;
        brotensor::cast(emb, f32, brotensor::Dtype::FP32);
        brotensor::sync_all();
        emb = std::move(f32);
    }
    dump_latent_f32(op, emb);
    std::printf("qi21-text-fwd: wrote prompt_embeds (%d,%d) to %s\n",
                emb.rows, emb.cols, op);
    if (mp) {
        dump_latent_f32(mp, cond.mask);
        std::printf("qi21-text-fwd: wrote mask (%d,%d) to %s\n",
                    cond.mask.rows, cond.mask.cols, mp);
    }
    if (ip) {
        // Only the SURVIVING ids — the reference's post-drop sequence.
        std::vector<std::int32_t> kept;
        for (std::size_t i = static_cast<std::size_t>(cond.drop_idx);
             i < cond.token_ids.size(); ++i) {
            kept.push_back(static_cast<std::int32_t>(cond.token_ids[i]));
        }
        std::ofstream f(ip, std::ios::binary | std::ios::trunc);
        if (!f) {
            throw std::runtime_error(std::string("cannot open --ids-out: ") + ip);
        }
        f.write(reinterpret_cast<const char*>(kept.data()),
                static_cast<std::streamsize>(kept.size() * sizeof(std::int32_t)));
        std::printf("qi21-text-fwd: wrote %zu token ids to %s\n",
                    kept.size(), ip);
    }
    return 0;
}

// Run ONE Krea2 DiT forward on fixed inputs (packed latent, stage-1
// conditioning, timestep, packed grid) read from raw float32 files, dump the
// velocity. Diffed against scripts/krea2_dit_ref.py. Sharded checkpoint
// (--weights-dir holds config.json + the diffusion_pytorch_model shards).
int run_krea2_fwd(int argc, char** argv) {
    const char* wdir = arg_after(argc, argv, "--weights-dir");
    const char* lp = arg_after(argc, argv, "--latent");
    const char* ep = arg_after(argc, argv, "--embeds");
    const char* mp = arg_after(argc, argv, "--mask");
    const char* op = arg_after(argc, argv, "--out");
    const char* ts = arg_after(argc, argv, "--t");
    const char* hps = arg_after(argc, argv, "--hp");
    const char* wps = arg_after(argc, argv, "--wp");
    const char* seqs = arg_after(argc, argv, "--seq");
    bool quantize = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--quantize") == 0) quantize = true;
    }
    if (!wdir || !lp || !ep || !mp || !op || !ts || !hps || !wps) {
        std::fprintf(stderr, "krea2-fwd: need --weights-dir --latent --embeds "
                             "--mask --out --t --hp --wp [--seq] [--quantize]\n");
        return 2;
    }
    const int hp = std::atoi(hps), wp = std::atoi(wps);
    const int text_seq = seqs ? std::atoi(seqs) : 512;
    const float t = static_cast<float>(std::atof(ts));
    brotensor::init();

    const std::string wd = wdir;
    auto cfg = load_krea2_config(wd + "/config.json");
    cfg.quantize_weights = quantize;
    brodiffusion::dit::Krea2Transformer2DModel model(cfg);

    std::vector<st::File> files = open_transformer_shards(wd);
    std::vector<const st::File*> shards;
    for (const st::File& f : files) shards.push_back(&f);
    model.load_weights(shards, "");

    const int img_len = hp * wp;
    auto lat_h = load_latent_f32(lp, img_len * cfg.in_channels);
    auto emb_h = load_latent_f32(ep, text_seq * cfg.num_text_layers * cfg.text_hidden_dim);
    auto msk_h = load_latent_f32(mp, text_seq);
    brotensor::Tensor lat =
        brotensor::Tensor::from_host(lat_h.data(), img_len, cfg.in_channels)
            .to(brotensor::default_device());
    brotensor::Tensor emb =
        brotensor::Tensor::from_host(emb_h.data(),
                                     text_seq * cfg.num_text_layers,
                                     cfg.text_hidden_dim)
            .to(brotensor::default_device());
    brotensor::Tensor msk =
        brotensor::Tensor::from_host(msk_h.data(), text_seq, 1)
            .to(brotensor::default_device());

    brotensor::Tensor out;
    model.forward(lat, hp, wp, emb, msk, t, out);
    brotensor::sync_all();
    // The compute dtype is BF16 on a CUDA backend; normalize before dumping.
    if (out.dtype != brotensor::Dtype::FP32) {
        brotensor::Tensor out_f32;
        brotensor::cast(out, out_f32, brotensor::Dtype::FP32);
        brotensor::sync_all();
        out = std::move(out_f32);
    }
    dump_latent_f32(op, out);
    std::printf("krea2-fwd: wrote velocity (%d,%d) to %s\n", out.rows, out.cols, op);
    return 0;
}

// Run ONE (or two consecutive) Qwen-Image 2.1 DiT forward(s) on fixed inputs —
// packed latent, Qwen3-VL text hidden states, timestep, latent token grid —
// read from raw float32 files, dump the velocity. Diffed against
// scripts/qwenimage21_dit_ref.py.
//
// With --steps 2 the first forward extracts the prefix KV cache and the second
// decodes from it: --out receives step 1's velocity and --out2 step 2's, so
// parity can cover the cached path. --no-cache runs without a cache at all
// (full prefill every step), which is what the cached result must match.
int run_qi21_fwd(int argc, char** argv) {
    const char* wdir = arg_after(argc, argv, "--weights-dir");
    const char* cfgp = arg_after(argc, argv, "--config");
    const char* lp = arg_after(argc, argv, "--latent");
    const char* ep = arg_after(argc, argv, "--embeds");
    const char* op = arg_after(argc, argv, "--out");
    const char* op2 = arg_after(argc, argv, "--out2");
    const char* ts = arg_after(argc, argv, "--t");
    const char* ts2 = arg_after(argc, argv, "--t2");
    const char* hps = arg_after(argc, argv, "--hp");
    const char* wps = arg_after(argc, argv, "--wp");
    const char* seqs = arg_after(argc, argv, "--seq");
    const char* steps_s = arg_after(argc, argv, "--steps");
    const char* bench_s = arg_after(argc, argv, "--bench");
    const char* bench_ab_s = arg_after(argc, argv, "--bench-ab");
    bool quantize = false, no_cache = false, synthetic = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--quantize") == 0) quantize = true;
        if (std::strcmp(argv[i], "--no-cache") == 0) no_cache = true;
        if (std::strcmp(argv[i], "--synthetic") == 0) synthetic = true;
    }
    // --synthetic makes its own inputs, so the parity files and the dump are
    // all optional: it exists for step timing, where what the tensors contain
    // is irrelevant and only their shape is.
    const bool need_io = !synthetic;
    if (!wdir || !ts || !hps || !wps || !seqs || (need_io && (!lp || !ep || !op))) {
        std::fprintf(stderr,
            "qi21-fwd: need --weights-dir --latent --embeds --out --t --hp "
            "--wp --seq [--config F] [--steps N] [--out2 F] [--t2 T] "
            "[--quantize] [--no-cache] [--synthetic] [--bench N]\n");
        return 2;
    }
    const int hp = std::atoi(hps), wp = std::atoi(wps);
    const int text_seq = std::atoi(seqs);
    const int steps = steps_s ? std::atoi(steps_s) : 1;
    const float t = static_cast<float>(std::atof(ts));
    const float t2 = ts2 ? static_cast<float>(std::atof(ts2)) : t;
    brotensor::init();

    const std::string wd = wdir;
    auto cfg = load_qi21_config(cfgp ? std::string(cfgp) : wd + "/config.json");
    cfg.quantize_weights = quantize;
    brodiffusion::dit::QwenImage21Transformer2DModel model(cfg);

    std::vector<st::File> files = open_transformer_shards(wd);
    std::vector<const st::File*> shards;
    for (const st::File& f : files) shards.push_back(&f);
    std::size_t free_before = 0, total_dev = 0;
    brotensor::device_mem_info(brotensor::default_device(), free_before,
                               total_dev);
    model.load_weights(shards, "");
    brotensor::sync_all();
    {
        std::size_t free_after = 0, total_after = 0;
        if (brotensor::device_mem_info(brotensor::default_device(), free_after,
                                       total_after) &&
            free_before > free_after) {
            std::printf("qi21-fwd: weights resident %.2f GiB (device %.2f GiB "
                        "free of %.2f)\n",
                        (free_before - free_after) / 1073741824.0,
                        free_after / 1073741824.0, total_after / 1073741824.0);
        }
    }

    const int img_len = hp * wp;
    // A deterministic unit-ish spread, from a splitmix64 stream so the same
    // command always denoises the same tensor and two builds are comparable.
    auto synth = [](std::size_t n, uint64_t seed) {
        std::vector<float> v(n);
        uint64_t s = seed;
        for (std::size_t i = 0; i < n; ++i) {
            s += 0x9e3779b97f4a7c15ull;
            uint64_t z = s;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
            z ^= z >> 31;
            v[i] = static_cast<float>(static_cast<int64_t>(z >> 40) - 8388608) /
                   8388608.0f;
        }
        return v;
    };
    auto lat_h = synthetic
                     ? synth(static_cast<std::size_t>(img_len) * cfg.in_channels, 1)
                     : load_latent_f32(lp, img_len * cfg.in_channels);
    auto emb_h =
        synthetic
            ? synth(static_cast<std::size_t>(text_seq) * cfg.context_in_dim, 2)
            : load_latent_f32(ep, text_seq * cfg.context_in_dim);
    brotensor::Tensor lat =
        brotensor::Tensor::from_host(lat_h.data(), img_len, cfg.in_channels)
            .to(brotensor::default_device());
    brotensor::Tensor emb =
        brotensor::Tensor::from_host(emb_h.data(), text_seq, cfg.context_in_dim)
            .to(brotensor::default_device());

    brotensor::Tensor txt;
    model.encode_text(emb, txt);

    brodiffusion::dit::QwenImage21PrefixCache cache;
    auto* cache_ptr = no_cache ? nullptr : &cache;

    auto dump = [&](const char* path, brotensor::Tensor& out) {
        brotensor::sync_all();
        if (out.dtype != brotensor::Dtype::FP32) {
            brotensor::Tensor out_f32;
            brotensor::cast(out, out_f32, brotensor::Dtype::FP32);
            brotensor::sync_all();
            out = std::move(out_f32);
        }
        dump_latent_f32(path, out);
        std::printf("qi21-fwd: wrote velocity (%d,%d) to %s\n", out.rows,
                    out.cols, path);
    };

    brotensor::Tensor out;
    model.forward(lat, hp, wp, txt, t, cache_ptr, out);
    if (op) {
        dump(op, out);
    } else {
        brotensor::sync_all();
        std::printf("qi21-fwd: velocity (%d,%d)\n", out.rows, out.cols);
    }

    if (steps >= 2) {
        if (!op2) {
            std::fprintf(stderr, "qi21-fwd: --steps 2 needs --out2\n");
            return 2;
        }
        brotensor::Tensor out2;
        model.forward(lat, hp, wp, txt, t2, cache_ptr, out2);
        dump(op2, out2);
    }

    // --bench N: N further forwards off the (already extracted) prefix cache,
    // reporting the steady-state per-step cost the denoising loop pays, plus
    // the peak device residency including the cache and activations.
    // --bench-ab N: the same N steps run with the trace JIT on and with it
    // off, alternating one step at a time inside a single process. A desktop
    // GPU that is also driving a display drifts — clocks ramp, another process
    // grabs the SMs — and two separate runs minutes apart cannot be subtracted
    // safely. Alternating puts both configurations under the same drift, and
    // the per-step minimum of each is then comparable.
    if (bench_ab_s) {
        const int n = std::atoi(bench_ab_s);
        const bool was_on = brodiffusion::detail::jit_enabled();
        brotensor::Tensor bo;
        for (int i = 0; i < 4; ++i) {   // spin the clocks up before timing
            model.forward(lat, hp, wp, txt, 0.5f, cache_ptr, bo);
        }
        brotensor::sync_all();
        double best[2] = {0.0, 0.0}, sum[2] = {0.0, 0.0};
        for (int i = 0; i < 2 * n; ++i) {
            const int which = i & 1;    // 0 = jit on, 1 = jit off
            brodiffusion::detail::set_jit_enabled(which == 0);
            const auto s0 = std::chrono::steady_clock::now();
            model.forward(lat, hp, wp, txt, 0.5f, cache_ptr, bo);
            brotensor::sync_all();
            const double ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - s0).count();
            sum[which] += ms;
            if (i < 2 || ms < best[which]) best[which] = ms;
        }
        brodiffusion::detail::set_jit_enabled(was_on);
        std::printf("qi21-fwd: %dx%d %d steps each, interleaved\n", hp, wp, n);
        std::printf("  jit on : %.1f ms/step min, %.1f mean\n", best[0],
                    sum[0] / n);
        std::printf("  jit off: %.1f ms/step min, %.1f mean\n", best[1],
                    sum[1] / n);
        std::printf("  delta  : %.1f ms/step (%.1f%% of the eager step)\n",
                    best[1] - best[0],
                    100.0 * (best[1] - best[0]) / (best[1] > 0 ? best[1] : 1.0));
    }

    if (bench_s) {
        const int n = std::atoi(bench_s);
        brotensor::Tensor bo;
        model.forward(lat, hp, wp, txt, 0.5f, cache_ptr, bo);   // warm-up
        brotensor::sync_all();
        // Each step is timed on its own and the MINIMUM reported alongside the
        // mean. A desktop GPU that is also driving a display hands out
        // multi-millisecond stalls that land in whichever step is unlucky; the
        // mean tracks the interference, the min tracks the kernel work, and
        // it is the kernel work a before/after comparison is about.
        double best = 0.0, total = 0.0;
        for (int i = 0; i < n; ++i) {
            const auto s0 = std::chrono::steady_clock::now();
            model.forward(lat, hp, wp, txt, 0.5f, cache_ptr, bo);
            brotensor::sync_all();
            const double ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - s0).count();
            total += ms;
            if (i == 0 || ms < best) best = ms;
        }
        const double mean = total / (n > 0 ? n : 1);
        std::size_t free_now = 0, total_now = 0;
        brotensor::device_mem_info(brotensor::default_device(), free_now,
                                   total_now);
        std::printf("qi21-fwd: %d cached steps at %dx%d: %.1f ms/step min, "
                    "%.1f mean (device %.2f GiB in use)\n",
                    n, hp, wp, best, mean,
                    (total_now - free_now) / 1073741824.0);
    }
    return 0;
}

}  // namespace brodiffusion::cli
