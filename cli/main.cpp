#include "brodiffusion/pipeline.h"
#include "brodiffusion/safetensors.h"
#include "brodiffusion/tokenizer.h"
#include "brodiffusion/unet.h"
#include "brodiffusion/version.h"

#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace pl   = brodiffusion::pipeline;
namespace st   = brodiffusion::safetensors;
namespace clip = brodiffusion::clip;

static int usage() {
    std::printf(
        "brodiffusion %s\n"
        "\n"
        "Usage:\n"
        "  brodiffusion --version\n"
        "  brodiffusion txt2img --text <st> --unet <st> --vae <st>\n"
        "                       --vocab <vocab.json> --merges <merges.txt>\n"
        "                       --prompt <text> --out <ppm>\n"
        "                       [--negative <text>] [--steps N] [--cfg F]\n"
        "                       [--width N] [--height N] [--seed N]\n"
        "  brodiffusion capture-inlet --text <st> --unet <st> --vae <st>\n"
        "                       --vocab <vocab.json> --merges <merges.txt>\n"
        "                       --prompts <file> --out <dir>\n"
        "                       [--seed N] [--steps N] [--cfg F]\n"
        "                       [--negative <text>] [--width N] [--height N]\n"
        "\n"
        "Writes an uncompressed PPM (P6) — a dev convenience for sanity-checking\n"
        "the generation pipeline. Proper PNG/JPEG encoding lives outside this\n"
        "library (bro's image-api on integration).\n",
        brodiffusion::version_string());
    return 0;
}

namespace {

const char* arg_after(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc - 1; ++i) {
        if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
    }
    return nullptr;
}

int run_txt2img(int argc, char** argv) {
    const char* text_path   = arg_after(argc, argv, "--text");
    const char* unet_path   = arg_after(argc, argv, "--unet");
    const char* vae_path    = arg_after(argc, argv, "--vae");
    const char* vocab_path  = arg_after(argc, argv, "--vocab");
    const char* merges_path = arg_after(argc, argv, "--merges");
    const char* prompt      = arg_after(argc, argv, "--prompt");
    const char* out_path    = arg_after(argc, argv, "--out");
    const char* neg         = arg_after(argc, argv, "--negative");
    const char* steps_s     = arg_after(argc, argv, "--steps");
    const char* cfg_s       = arg_after(argc, argv, "--cfg");
    const char* width_s     = arg_after(argc, argv, "--width");
    const char* height_s    = arg_after(argc, argv, "--height");
    const char* seed_s      = arg_after(argc, argv, "--seed");

    if (!text_path || !unet_path || !vae_path ||
        !vocab_path || !merges_path || !prompt || !out_path) {
        std::fprintf(stderr,
            "txt2img: --text, --unet, --vae, --vocab, --merges, --prompt, --out are required\n");
        return 2;
    }

    pl::GenerateOptions opts;
    if (neg)     opts.negative_prompt = neg;
    if (steps_s) opts.num_inference_steps = std::atoi(steps_s);
    if (cfg_s)   opts.guidance_scale = static_cast<float>(std::atof(cfg_s));
    if (width_s) opts.width  = std::atoi(width_s);
    if (height_s)opts.height = std::atoi(height_s);
    if (seed_s)  opts.seed = static_cast<std::uint64_t>(std::strtoull(seed_s, nullptr, 10));

    brotensor::cuda_init();

    auto tok = clip::Tokenizer::load(vocab_path, merges_path);

    pl::PipelineConfig cfg;
    pl::Pipeline pipeline(cfg, std::move(tok));

    std::printf("Loading weights:\n  text: %s\n  unet: %s\n  vae:  %s\n",
                text_path, unet_path, vae_path);
    auto text_file = st::File::open(text_path);
    auto unet_file = st::File::open(unet_path);
    auto vae_file  = st::File::open(vae_path);
    pipeline.load_weights(text_file, unet_file, vae_file);

    std::printf("Generating %dx%d, %d steps, CFG=%.1f, seed=%llu\n",
                opts.width, opts.height, opts.num_inference_steps,
                static_cast<double>(opts.guidance_scale),
                static_cast<unsigned long long>(opts.seed));

    auto img = pipeline.generate(prompt, opts);  // (3*H*W) NCHW, FP32 in [-1, 1]

    // Convert planar [-1,1] FP32 → interleaved RGB8.
    const int H = opts.height, W = opts.width;
    const int plane = H * W;
    std::vector<std::uint8_t> rgb(static_cast<std::size_t>(3) * plane);
    for (int i = 0; i < plane; ++i) {
        for (int c = 0; c < 3; ++c) {
            float v = (img[c * plane + i] + 1.0f) * 127.5f;
            v = std::clamp(v, 0.0f, 255.0f);
            rgb[3 * i + c] = static_cast<std::uint8_t>(v + 0.5f);
        }
    }

    std::ofstream f(out_path, std::ios::binary | std::ios::trunc);
    if (!f) {
        std::fprintf(stderr, "txt2img: cannot open output %s\n", out_path);
        return 1;
    }
    f << "P6\n" << W << " " << H << "\n255\n";
    f.write(reinterpret_cast<const char*>(rgb.data()),
            static_cast<std::streamsize>(rgb.size()));
    std::printf("Wrote %s\n", out_path);
    return 0;
}

// ─── capture-inlet ─────────────────────────────────────────────────────────
//
// For each prompt × denoising step × CFG branch (cond/uncond) the teacher
// UNet is run forward; via an InletCaptureHook we snapshot (sample, t_emb_raw,
// ctx, s0..s11) to host memory and emit one safetensors file per call. The
// trainer downstream consumes these files as its (input, target) corpus.

class CaptureWriter : public brodiffusion::unet::InletCaptureHook {
public:
    void set_target(const std::string& path) {
        path_ = path;
        have_inputs_ = false;
        have_skips_  = false;
    }

    void on_inputs(const brotensor::GpuTensor& sample,
                   const brotensor::GpuTensor& t_emb_raw,
                   const brotensor::GpuTensor& ctx) override {
        copy_to_host_(sample,    in_sample_);
        copy_to_host_(t_emb_raw, in_temb_);
        copy_to_host_(ctx,       in_ctx_);
        in_sample_shape_ = {1, 4, 64, 64};
        in_temb_shape_   = {1, static_cast<int64_t>(t_emb_raw.cols)};
        in_ctx_shape_    = {static_cast<int64_t>(ctx.rows),
                            static_cast<int64_t>(ctx.cols)};
        have_inputs_ = true;
    }

    void on_skips(const std::array<const brotensor::GpuTensor*, 12>& skips) override {
        // Teacher skip channel widths × spatial dims (matches Inlet header doc):
        //   s0..s2:  C=320 @ 64x64
        //   s3:      C=320 @ 32x32
        //   s4..s5:  C=640 @ 32x32
        //   s6:      C=640 @ 16x16
        //   s7..s8:  C=1280 @ 16x16
        //   s9..s11: C=1280 @ 8x8
        static const int kC[12] = {320,320,320,320, 640,640,640,
                                   1280,1280,1280,1280,1280};
        static const int kS[12] = { 64, 64, 64, 32,  32, 32, 16,
                                     16, 16,  8,  8,  8};
        for (int i = 0; i < 12; ++i) {
            copy_to_host_(*skips[static_cast<std::size_t>(i)],
                          skip_buf_[static_cast<std::size_t>(i)]);
            skip_shape_[static_cast<std::size_t>(i)] = {
                1, kC[i], kS[i], kS[i]};
            // Sanity: tensor element count should match.
            const int64_t n = static_cast<int64_t>(kC[i]) *
                              static_cast<int64_t>(kS[i]) *
                              static_cast<int64_t>(kS[i]);
            if (static_cast<int64_t>(skips[static_cast<std::size_t>(i)]->size()) != n) {
                throw std::runtime_error("capture-inlet: skip " +
                    std::to_string(i) + " element count mismatch");
            }
        }
        have_skips_ = true;
        flush_();
    }

private:
    void copy_to_host_(const brotensor::GpuTensor& g, std::vector<uint16_t>& host) {
        host.resize(static_cast<std::size_t>(g.size()));
        brotensor::download_fp16(g, host.data());
    }

    void flush_() {
        if (!have_inputs_ || !have_skips_) {
            throw std::runtime_error("capture-inlet: flush before both halves seen");
        }
        namespace st = brodiffusion::safetensors;
        std::vector<st::WriteEntry> ents;
        ents.reserve(15);

        auto push = [&](const char* name,
                        const std::vector<uint16_t>& h,
                        const std::vector<int64_t>& shape) {
            st::WriteEntry e;
            e.name      = name;
            e.dtype     = st::Dtype::F16;
            e.shape     = shape;
            e.host_data = h.data();
            e.bytes     = h.size() * sizeof(uint16_t);
            ents.push_back(std::move(e));
        };
        push("sample",    in_sample_, in_sample_shape_);
        push("t_emb_raw", in_temb_,   in_temb_shape_);
        push("ctx",       in_ctx_,    in_ctx_shape_);
        for (int i = 0; i < 12; ++i) {
            std::string name = "s" + std::to_string(i);
            push(name.c_str(),
                 skip_buf_[static_cast<std::size_t>(i)],
                 skip_shape_[static_cast<std::size_t>(i)]);
        }
        // names hold strings through ents lifetime — WriteEntry takes a copy
        // of `name` (std::string member), so the local std::string above is
        // safe to destroy after push_back.
        st::write_file(path_, ents);
        have_inputs_ = false;
        have_skips_  = false;
    }

    std::string path_;
    bool have_inputs_ = false, have_skips_ = false;

    std::vector<uint16_t>             in_sample_, in_temb_, in_ctx_;
    std::vector<int64_t>              in_sample_shape_, in_temb_shape_, in_ctx_shape_;
    std::array<std::vector<uint16_t>, 12> skip_buf_;
    std::array<std::vector<int64_t>, 12>  skip_shape_;
};

// Drives a single Pipeline::generate call but plumbs prompt/step/branch
// tagging into the capture path. The Pipeline always runs cond first, then
// uncond (when CFG is enabled); we track that ordering with a counter.
class CaptureDriver : public brodiffusion::unet::InletCaptureHook {
public:
    CaptureDriver(const std::string& out_dir, int prompt_index, int total_steps, bool do_cfg)
        : out_dir_(out_dir), prompt_index_(prompt_index),
          total_steps_(total_steps), do_cfg_(do_cfg) {}

    void on_inputs(const brotensor::GpuTensor& sample,
                   const brotensor::GpuTensor& t_emb_raw,
                   const brotensor::GpuTensor& ctx) override {
        // Compute current (step, branch) from call_idx_.
        const int per_step = do_cfg_ ? 2 : 1;
        const int step     = call_idx_ / per_step;
        const bool uncond  = (do_cfg_ && (call_idx_ % per_step) == 1);
        // Filename: pNNNN_stepS_<cond|uncond>.safetensors
        char fn[256];
        std::snprintf(fn, sizeof(fn), "p%04d_step%d_%s.safetensors",
                      prompt_index_, step, uncond ? "uncond" : "cond");
        std::filesystem::path p = std::filesystem::path(out_dir_) / fn;
        last_path_ = p.string();
        writer_.set_target(last_path_);
        writer_.on_inputs(sample, t_emb_raw, ctx);
        if (step >= total_steps_) {
            throw std::runtime_error("capture-inlet: step counter overflow");
        }
    }

    void on_skips(const std::array<const brotensor::GpuTensor*, 12>& skips) override {
        writer_.on_skips(skips);
        files_written_++;
        // Approximate bytes (no fs call): each capture file is sample 32KB +
        // t_emb 2.56KB + ctx ≈ 118.27KB + 12 skips. Compute exactly.
        bytes_written_ += last_file_bytes_();
        call_idx_++;
    }

    int  files_written() const { return files_written_; }
    uint64_t bytes_written() const { return bytes_written_; }
    const std::string& last_path() const { return last_path_; }

private:
    static uint64_t last_file_bytes_() {
        // sample (4*64*64) + t_emb (1280) + ctx (77*768) + s0..s11
        const int64_t per_tap[12] = {
            320LL*64*64, 320LL*64*64, 320LL*64*64, 320LL*32*32,
            640LL*32*32, 640LL*32*32, 640LL*16*16,
            1280LL*16*16, 1280LL*16*16, 1280LL*8*8, 1280LL*8*8, 1280LL*8*8};
        int64_t n = 4LL*64*64 + 1280LL + 77LL*768LL;
        for (int i = 0; i < 12; ++i) n += per_tap[i];
        return static_cast<uint64_t>(n) * 2ULL;  // FP16
    }

    CaptureWriter writer_;
    std::string   out_dir_;
    std::string   last_path_;
    int           prompt_index_;
    int           total_steps_;
    bool          do_cfg_;
    int           call_idx_ = 0;
    int           files_written_ = 0;
    uint64_t      bytes_written_ = 0;
};

int run_capture_inlet(int argc, char** argv) {
    const char* text_path   = arg_after(argc, argv, "--text");
    const char* unet_path   = arg_after(argc, argv, "--unet");
    const char* vae_path    = arg_after(argc, argv, "--vae");
    const char* vocab_path  = arg_after(argc, argv, "--vocab");
    const char* merges_path = arg_after(argc, argv, "--merges");
    const char* prompts_path= arg_after(argc, argv, "--prompts");
    const char* out_path    = arg_after(argc, argv, "--out");
    const char* seed_s      = arg_after(argc, argv, "--seed");
    const char* steps_s     = arg_after(argc, argv, "--steps");
    const char* cfg_s       = arg_after(argc, argv, "--cfg");
    const char* neg         = arg_after(argc, argv, "--negative");
    const char* width_s     = arg_after(argc, argv, "--width");
    const char* height_s    = arg_after(argc, argv, "--height");

    if (!text_path || !unet_path || !vae_path || !vocab_path || !merges_path ||
        !prompts_path || !out_path) {
        std::fprintf(stderr,
            "capture-inlet: --text, --unet, --vae, --vocab, --merges, --prompts, --out required\n");
        return 2;
    }

    const std::uint64_t base_seed = seed_s ? std::strtoull(seed_s, nullptr, 10) : 0ULL;
    const int   steps  = steps_s  ? std::atoi(steps_s)  : 5;
    const float guid   = cfg_s    ? static_cast<float>(std::atof(cfg_s)) : 7.5f;
    const std::string neg_prompt = neg ? std::string(neg) : std::string();
    const int   width  = width_s  ? std::atoi(width_s)  : 512;
    const int   height = height_s ? std::atoi(height_s) : 512;
    const bool  do_cfg = (guid != 1.0f);

    // Read prompt list (skip blank lines).
    std::vector<std::string> prompts;
    {
        std::ifstream f(prompts_path);
        if (!f) {
            std::fprintf(stderr, "capture-inlet: cannot open prompts file '%s'\n", prompts_path);
            return 1;
        }
        std::string line;
        while (std::getline(f, line)) {
            // Strip trailing CR (Windows line endings) and leading/trailing whitespace.
            while (!line.empty() &&
                   (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) {
                line.pop_back();
            }
            std::size_t start = 0;
            while (start < line.size() && (line[start] == ' ' || line[start] == '\t')) ++start;
            if (start >= line.size()) continue;
            prompts.push_back(line.substr(start));
        }
    }
    if (prompts.empty()) {
        std::fprintf(stderr, "capture-inlet: no prompts in '%s'\n", prompts_path);
        return 1;
    }

    std::error_code ec;
    std::filesystem::create_directories(out_path, ec);
    if (ec) {
        std::fprintf(stderr, "capture-inlet: cannot create output dir '%s': %s\n",
                     out_path, ec.message().c_str());
        return 1;
    }

    brotensor::cuda_init();
    auto tok = clip::Tokenizer::load(vocab_path, merges_path);
    pl::PipelineConfig cfg;
    // Teacher path only — capture hook is only meaningful when the teacher
    // down path runs (inlet is the *target* model, not the source).
    cfg.unet.enable_inlet = false;
    pl::Pipeline pipeline(cfg, std::move(tok));

    std::fprintf(stderr, "capture-inlet: loading weights...\n");
    std::fflush(stderr);
    pipeline.load_weights(st::File::open(text_path),
                          st::File::open(unet_path),
                          st::File::open(vae_path));

    pl::GenerateOptions opts;
    opts.num_inference_steps = steps;
    opts.guidance_scale      = guid;
    opts.negative_prompt     = neg_prompt;
    opts.width               = width;
    opts.height              = height;

    std::fprintf(stderr,
        "capture-inlet: %zu prompts × %d steps × %d branches → '%s' (CFG=%.2f)\n",
        prompts.size(), steps, do_cfg ? 2 : 1, out_path,
        static_cast<double>(guid));
    std::fflush(stderr);

    int total_files = 0;
    uint64_t total_bytes = 0;
    std::string last_written;
    for (std::size_t i = 0; i < prompts.size(); ++i) {
        opts.seed = base_seed + static_cast<std::uint64_t>(i);
        CaptureDriver drv(out_path, static_cast<int>(i), steps, do_cfg);
        pipeline.set_inlet_capture_hook(&drv);
        try {
            (void)pipeline.generate(prompts[i], opts);
        } catch (...) {
            pipeline.set_inlet_capture_hook(nullptr);
            throw;
        }
        pipeline.set_inlet_capture_hook(nullptr);
        total_files += drv.files_written();
        total_bytes += drv.bytes_written();
        last_written = drv.last_path();
        std::fprintf(stderr, "  p%04zu (\"%.40s%s\"): %d files\n",
                     i, prompts[i].c_str(),
                     prompts[i].size() > 40 ? "..." : "",
                     drv.files_written());
        std::fflush(stderr);
    }

    std::fprintf(stderr,
        "capture-inlet: wrote %d files, %.2f MB on disk (%llu bytes payload)\n",
        total_files, static_cast<double>(total_bytes) / (1024.0 * 1024.0),
        static_cast<unsigned long long>(total_bytes));
    std::fflush(stderr);

    // ── Verification: reopen the last file with the reader; dump keys+shapes.
    if (!last_written.empty()) {
        try {
            auto f = st::File::open(last_written);
            std::fprintf(stderr, "capture-inlet: verify '%s' (%zu tensors):\n",
                         last_written.c_str(), f.size());
            for (const auto& tv : f.tensors()) {
                std::fprintf(stderr, "    %-12s dtype=%s shape=[",
                             tv.name.c_str(), st::dtype_name(tv.dtype));
                for (std::size_t j = 0; j < tv.shape.size(); ++j) {
                    std::fprintf(stderr, "%s%lld",
                                 j ? "," : "",
                                 static_cast<long long>(tv.shape[j]));
                }
                std::fprintf(stderr, "] bytes=%zu\n", tv.nbytes);
            }
            std::fflush(stderr);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "capture-inlet: verify failed: %s\n", e.what());
            return 1;
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) return usage();
    if (std::strcmp(argv[1], "--version") == 0 || std::strcmp(argv[1], "-v") == 0) {
        std::printf("brodiffusion %s\n", brodiffusion::version_string());
        return 0;
    }
    if (std::strcmp(argv[1], "txt2img") == 0) {
        try {
            return run_txt2img(argc, argv);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "txt2img: %s\n", e.what());
            return 1;
        }
    }
    if (std::strcmp(argv[1], "capture-inlet") == 0) {
        try {
            return run_capture_inlet(argc, argv);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "capture-inlet: %s\n", e.what());
            return 1;
        }
    }
    if (std::strcmp(argv[1], "bench") == 0) {
        try {
            const char* text_path   = arg_after(argc, argv, "--text");
            const char* unet_path   = arg_after(argc, argv, "--unet");
            const char* vae_path    = arg_after(argc, argv, "--vae");
            const char* vocab_path  = arg_after(argc, argv, "--vocab");
            const char* merges_path = arg_after(argc, argv, "--merges");
            const char* steps_s     = arg_after(argc, argv, "--steps");
            const char* iters_s     = arg_after(argc, argv, "--iters");
            const char* warmup_s    = arg_after(argc, argv, "--warmup");
            if (!text_path || !unet_path || !vae_path || !vocab_path || !merges_path) {
                std::fprintf(stderr, "bench: --text, --unet, --vae, --vocab, --merges required\n");
                return 2;
            }
            const int steps  = steps_s  ? std::atoi(steps_s)  : 5;
            const int iters  = iters_s  ? std::atoi(iters_s)  : 5;
            const int warmup = warmup_s ? std::atoi(warmup_s) : 1;

            brotensor::cuda_init();
            auto tok = clip::Tokenizer::load(vocab_path, merges_path);
            pl::PipelineConfig cfg;
            // Optional: replace the teacher UNet down path with the inlet
            // module. Weights stay zero-init (ceiling bench) unless --inlet
            // is also provided.
            const char* inlet_env = std::getenv("BRODIFFUSION_INLET");
            const bool enable_inlet = (inlet_env && inlet_env[0] == '1');
            if (enable_inlet) {
                cfg.unet.enable_inlet = true;
                std::fprintf(stderr, "[bench] BRODIFFUSION_INLET=1 — inlet enabled (zero-init unless --inlet given)\n");
                std::fflush(stderr);
            }
            pl::Pipeline pipeline(cfg, std::move(tok));
            pipeline.load_weights(st::File::open(text_path),
                                  st::File::open(unet_path),
                                  st::File::open(vae_path));
            pl::GenerateOptions opts;
            opts.num_inference_steps = steps;
            opts.guidance_scale = 1.0f;   // skip uncond pass — bench unet+vae core
            opts.width = 512; opts.height = 512; opts.seed = 1;

            for (int i = 0; i < warmup; ++i) (void)pipeline.generate("an astronaut", opts);
            using clk = std::chrono::high_resolution_clock;
            double sum_ms = 0.0, mn = 1e30, mx = 0.0;
            for (int i = 0; i < iters; ++i) {
                auto t0 = clk::now();
                (void)pipeline.generate("an astronaut", opts);
                auto t1 = clk::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                sum_ms += ms; mn = std::min(mn, ms); mx = std::max(mx, ms);
                std::printf("  iter %d: %.2f ms (%.2f steps/s)\n", i, ms, 1000.0 * steps / ms);
            }
            const double avg = sum_ms / iters;
            std::printf("bench: steps=%d iters=%d  avg=%.2f ms  min=%.2f  max=%.2f\n",
                        steps, iters, avg, mn, mx);
            std::printf("       per-step=%.2f ms  throughput=%.2f (steps=%d)/s  (target 24/s @ %d steps)\n",
                        avg / steps, 1000.0 / avg, steps, steps);
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "bench: %s\n", e.what());
            return 1;
        }
    }
    return usage();
}
