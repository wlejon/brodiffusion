// Terrain-diffusion performance harness.
//
// Four views on the same pipeline:
//
//   * `unet`  — per-net microbenchmark at the shapes the pipeline actually
//               runs (coarse 1x64^2, base Bx64^2, decoder Bx512^2), swept over
//               batch size. This is what says whether a stage is saturating
//               the GPU or paying launch overhead.
//
//   * `kernels` — the primitives under those nets: a 3x3 / 1x1 convolution
//               against a plain GEMM of identical M/N/K, plus the data movement
//               a residual block does around them (clone, the NCHW<->sequence
//               round trip in the channel norm, the per-channel modulation).
//               Says how much of a stage's time is convolution and how far that
//               convolution is from the tensor cores' own GEMM rate.
//
//   * `world` — end-to-end elevation over a square region, with the DAG's
//               three stages timed separately by materializing them in order
//               (coarse first, then latent on a warm coarse cache, then the
//               residual, then the Laplacian reconstruction). Reports the tile
//               counts each stage ran, plus throughput in cells/s and km^2/s.
//
//   * `stream` — the shape a real world generator has: one long-lived pipeline
//               asked for a raster of adjacent regions. Per-region timings show
//               whether the tile cache is paying for the overlap or thrashing
//               and recomputing it.
//
// Usage:
//   bench_terrain unet    --weights <dir> [--batches 1,2,4,8,16] [--iters N]
//   bench_terrain kernels [--iters N]                     (no weights needed)
//   bench_terrain world   --weights <dir> [--seed N] [--sizes 512,1024,2048]
//                         [--repeat N]
//   bench_terrain stream  --weights <dir> [--seed N] [--region 512] [--grid 4]
//
// Everything is a pure function of (seed, position), so a `world` run at one
// size tells you nothing about the next size's cache state — each size builds
// a fresh WorldPipeline.

#include "brodiffusion/terrain/laplacian.h"
#include "brodiffusion/terrain/mp_unet.h"
#include "brodiffusion/terrain/world_pipeline.h"
#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace td = brodiffusion::terrain;
namespace bt = brotensor;

namespace {

using Clock = std::chrono::steady_clock;

double since(const Clock::time_point& t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

const char* arg_after(int argc, char** argv, const char* key) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
    }
    return nullptr;
}

std::vector<int> int_list(const char* s, std::vector<int> dflt) {
    if (!s) return dflt;
    std::vector<int> out;
    const char* p = s;
    while (*p) {
        char* end = nullptr;
        const long v = std::strtol(p, &end, 10);
        if (end == p) break;
        out.push_back(static_cast<int>(v));
        p = (*end == ',') ? end + 1 : end;
    }
    return out.empty() ? dflt : out;
}

// Upload host FP32 at the pipeline compute dtype (FP16 on CUDA, FP32 on CPU).
bt::Tensor upload(const std::vector<float>& h, int rows, int cols) {
    if (bt::compute_dtype() == bt::Dtype::FP16) {
        std::vector<std::uint16_t> bits(h.size());
        for (std::size_t i = 0; i < h.size(); ++i) bits[i] = bt::fp32_to_fp16_bits(h[i]);
        return bt::Tensor::from_host_fp16(bits.data(), rows, cols);
    }
    return bt::Tensor::from_host(h.data(), rows, cols);
}

// ─── unet microbenchmark ───────────────────────────────────────────────────

// Rough forward FLOP count for one MPUNet evaluation at spatial size S: every
// conv in the enc/dec ladder, counted as 2*C_in*C_out*k^2 per output pixel.
// Attention and the elementwise tail are ignored — they are single-digit
// percent at these shapes, and the number exists to put a roofline under the
// measured time, not to be exact.
double unet_gflops(const td::MPUNetConfig& c, int S) {
    const int L = static_cast<int>(c.model_channel_mults.size());
    double f = 0.0;
    auto conv = [&](long long cin, long long cout, int k, int s) {
        f += 2.0 * static_cast<double>(cin) * static_cast<double>(cout) *
             k * k * static_cast<double>(s) * s;
    };

    int cout = c.in_channels + 1;
    int sz = S;
    std::vector<int> skips;
    for (int level = 0; level < L; ++level) {
        const int ch = c.model_channels * c.model_channel_mults[level];
        if (level == 0) {
            conv(cout, ch, 3, sz);
            cout = ch;
            skips.push_back(cout);
        } else {
            sz /= 2;                       // the 'down' block subsamples
            skips.push_back(cout);
        }
        for (int i = 0; i < c.layers_per_block[level]; ++i) {
            if (cout != ch) conv(cout, ch, 1, sz);
            conv(cout, ch, 3, sz);
            conv(ch, ch, 3, sz);
            cout = ch;
            skips.push_back(cout);
        }
    }
    for (int level = L - 1; level >= 0; --level) {
        const int ch = c.model_channels * c.model_channel_mults[level];
        if (level == L - 1) {
            conv(cout, cout, 3, sz);       // in0
            conv(cout, cout, 3, sz);
            conv(cout, cout, 3, sz);       // in1
            conv(cout, cout, 3, sz);
        } else {
            sz *= 2;
        }
        for (int i = 0; i <= c.layers_per_block[level]; ++i) {
            const int cin = cout + (skips.empty() ? 0 : skips.back());
            if (!skips.empty()) skips.pop_back();
            if (cin != ch) conv(cin, ch, 1, sz);
            conv(cin, ch, 3, sz);
            conv(ch, ch, 3, sz);
            cout = ch;
        }
    }
    conv(cout, c.out_channels, 3, sz);
    return f / 1e9;
}

struct StageSpec {
    const char* name;
    int         S;        // spatial extent the pipeline runs this stage at
    int         steps;    // sequential UNet evaluations per tile
};

void bench_unet(const std::string& dir, const std::vector<int>& batches, int iters) {
    const StageSpec stages[] = {
        {"coarse",  64, 20},   // DPM-Solver++, 20 steps
        {"base",    64,  2},   // TrigFlow, 2 steps
        {"decoder", 512, 1},   // TrigFlow, 1 step
    };

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    for (const StageSpec& sp : stages) {
        const auto t_cfg = Clock::now();
        auto cfg = td::MPUNetConfig::from_config_json(dir + "/config.json", sp.name);
        td::MPUNet net(cfg);
        {
            auto f = bt::safetensors::File::open(dir + "/" + sp.name + ".safetensors");
            net.load_weights(f);
        }
        bt::sync_all();
        const double load = since(t_cfg);

        const double gf = unet_gflops(cfg, sp.S);
        std::printf("\n== %s  S=%d  in=%d out=%d ch=%d  %.1f GFLOP/eval  "
                    "(load %.2fs)\n",
                    sp.name, sp.S, cfg.in_channels, cfg.out_channels,
                    cfg.model_channels, gf, load);
        std::printf("   %6s %10s %10s %10s %10s\n",
                    "batch", "ms/eval", "ms/tile", "TFLOP/s", "tiles/s");

        for (int B : batches) {
            // A 512^2 decoder tile is 5 channels in, ~1.3 GB of activations at
            // batch 8; keep the sweep inside a 24 GB card.
            if (sp.S >= 512 && B > 8) continue;

            const std::size_t n =
                static_cast<std::size_t>(B) * cfg.in_channels * sp.S * sp.S;
            std::vector<float> h(n);
            for (auto& v : h) v = nd(rng);
            bt::Tensor x = upload(h, B, cfg.in_channels * sp.S * sp.S);

            std::vector<float> labels(static_cast<std::size_t>(B), 0.7f);
            std::vector<std::vector<float>> cond;
            for (const auto& ci : cfg.conditional_inputs) {
                const std::size_t m = (ci.kind == "tensor")
                                          ? static_cast<std::size_t>(B) * ci.dim
                                          : static_cast<std::size_t>(B);
                std::vector<float> c(m);
                for (auto& v : c) v = nd(rng) * 0.1f;
                cond.push_back(std::move(c));
            }

            bt::Tensor y;
            bool ok = true;
            try {
                net.forward(x, B, sp.S, labels.data(), cond, y);
                bt::sync_all();
            } catch (const std::exception& e) {
                std::printf("   %6d  FAILED: %s\n", B, e.what());
                ok = false;
            }
            if (!ok) continue;

            const auto t0 = Clock::now();
            for (int i = 0; i < iters; ++i) net.forward(x, B, sp.S, labels.data(), cond, y);
            bt::sync_all();
            const double per_batch = since(t0) / iters;

            const double per_eval = per_batch / B;
            const double per_tile = per_eval * sp.steps;
            std::printf("   %6d %10.2f %10.2f %10.2f %10.2f\n",
                        B, per_eval * 1e3, per_tile * 1e3,
                        gf * B / per_batch / 1e3, 1.0 / per_tile);
        }
    }
}

// ─── primitive attribution ─────────────────────────────────────────────────
//
// mp_unet.cpp's residual block is a convolution sandwich wrapped in a lot of
// data movement: a clone per branch, a channel-norm that transposes NCHW into
// sequence layout and back, a per-channel modulation that does the same, and a
// magnitude-preserving sum that clones again. This mode times each of those
// primitives at the (N, C, S) shapes the two heavy stages actually run, so the
// gap between a stage's measured TFLOP/s and its convolutions' TFLOP/s can be
// attributed rather than guessed at.

template <typename F>
double time_op(F&& f, int iters) {
    f();
    bt::sync_all();
    const auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) f();
    bt::sync_all();
    return since(t0) / iters;
}

bt::Tensor randn_dev(int rows, int cols, std::mt19937& rng) {
    std::normal_distribution<float> d(0.0f, 1.0f);
    std::vector<float> h(static_cast<std::size_t>(rows) * cols);
    for (auto& v : h) v = d(rng);
    return upload(h, rows, cols);
}

// One (N, C, S) rung of a UNet ladder.
void bench_rung(const char* stage, int N, int C, int S, int iters,
                std::mt19937& rng) {
    const int L = S * S;
    bt::Tensor x  = randn_dev(N, C * L, rng);
    bt::Tensor w3 = randn_dev(C, C * 9, rng);
    bt::Tensor w1 = randn_dev(C, C, rng);
    bt::Tensor emb = randn_dev(N, C, rng);

    const double conv3_gf = 2.0 * C * C * 9.0 * L * N / 1e9;

    bt::Tensor y, seq, nrm, prod;
    const double t_conv3 = time_op([&] {
        bt::conv2d_forward(x, w3, nullptr, N, C, S, S, C, 3, 3, 1, 1, 1, 1, 1, 1, 1, y);
    }, iters);
    const double t_conv1 = time_op([&] {
        bt::conv2d_forward(x, w1, nullptr, N, C, S, S, C, 1, 1, 1, 1, 0, 0, 1, 1, 1, y);
    }, iters);
    // clone(): the allocate-and-copy mp_sum2 / mp_concat2 / the residual branch
    // each do at least once per block.
    const double t_clone = time_op([&] {
        bt::Tensor o = bt::Tensor::zeros_on(x.device, x.rows, x.cols, x.dtype);
        bt::copy_d2d(x, 0, o, 0, x.rows * x.cols);
    }, iters);
    // pixel_norm_channels(): transpose in, normalise, transpose back.
    const double t_pnorm = time_op([&] {
        bt::nchw_to_sequence(x, N, C, S, S, seq);
        bt::pixel_norm_forward(seq, 1e-4f, nrm);
        bt::Tensor o;
        bt::sequence_to_nchw(nrm, N, C, S, S, o);
    }, iters);
    // mul_per_channel(): same round trip, plus a per-batch-item loop when N > 1.
    const double t_mulc = time_op([&] {
        bt::Tensor o = x;
        bt::nchw_to_sequence(o, N, C, S, S, seq);
        if (N == 1) {
            bt::broadcast_mul(seq, emb, prod);
        } else {
            prod = bt::Tensor::zeros_on(seq.device, N * L, C, seq.dtype);
            bt::Tensor row = bt::Tensor::zeros_on(emb.device, 1, C, emb.dtype);
            bt::Tensor blk = bt::Tensor::zeros_on(seq.device, L, C, seq.dtype);
            bt::Tensor sub;
            for (int n = 0; n < N; ++n) {
                bt::copy_d2d(emb, n * C, row, 0, C);
                bt::copy_d2d(seq, n * L * C, blk, 0, L * C);
                bt::broadcast_mul(blk, row, sub);
                bt::copy_d2d(sub, 0, prod, n * L * C, L * C);
            }
        }
        bt::sequence_to_nchw(prod, N, C, S, S, o);
    }, iters);
    const double t_silu = time_op([&] {
        bt::Tensor o = bt::Tensor::zeros_on(x.device, x.rows, x.cols, x.dtype);
        bt::silu_forward(x, o);
        bt::scale_inplace(o, 1.0f / 0.596f);
    }, iters);

    const double overhead = t_clone + t_pnorm + t_mulc + t_silu;
    std::printf("   %-8s %3d %5d %5d | %8.3f %8.3f | %8.3f %8.3f %8.3f %8.3f "
                "| %8.3f %7.2fx\n",
                stage, N, C, S,
                t_conv3 * 1e3, t_conv1 * 1e3,
                t_clone * 1e3, t_pnorm * 1e3, t_mulc * 1e3, t_silu * 1e3,
                conv3_gf / t_conv3 / 1e3, overhead / t_conv3);
}

// What the same tensor cores do on a plain GEMM, as a ceiling for the conv
// numbers. The second set is shaped like the convolutions themselves — an
// implicit-GEMM conv is a (N*H*W) x C_out x (C_in*k*k) matrix product, and the
// terrain ladders make that very skinny in N (C_out is 64 at the decoder's
// widest level), which is a different regime from a square GEMM.
void bench_gemm_roofline(int iters, std::mt19937& rng) {
    struct Shape { const char* what; int M, N, K; };
    const Shape shapes[] = {
        {"square 4096",      4096,   4096, 4096},
        {"square 8192",      8192,   8192, 8192},
        {"dec L0 conv3x3", 262144,     64,  576},   // 1x64x512^2, 3x3
        {"dec L1 conv3x3",  65536,    128, 1152},
        {"base L0 conv3x3", 65536,    192, 1728},   // 16x192x64^2, 3x3
        {"base L3 conv3x3",  1024,    768, 6912},
    };
    std::printf("\n== GEMM roofline (same tensor cores, no conv addressing)\n");
    std::printf("   %-15s %8s %6s %6s | %9s %9s\n", "shape", "M", "N", "K",
                "ms", "TFLOP/s");
    for (const Shape& s : shapes) {
        bt::Tensor A = randn_dev(s.M, s.K, rng);
        bt::Tensor B = randn_dev(s.K, s.N, rng);
        bt::Tensor C;
        const double gf = 2.0 * s.M * s.N * static_cast<double>(s.K) / 1e9;
        const double t = time_op([&] { bt::matmul(A, B, C); }, iters);
        std::printf("   %-15s %8d %6d %6d | %9.3f %9.2f\n",
                    s.what, s.M, s.N, s.K, t * 1e3, gf / t / 1e3);
    }
}

void bench_kernels(int iters) {
    std::mt19937 rng(99);
    bench_gemm_roofline(iters, rng);
    std::printf("\n== primitives at the shapes the ladders run\n");
    std::printf("   ms per call; 'ovh/conv3' is (clone + pnorm + mulc + silu) "
                "divided by one 3x3 conv\n\n");
    std::printf("   %-8s %3s %5s %5s | %8s %8s | %8s %8s %8s %8s | %8s %8s\n",
                "stage", "N", "C", "S", "conv3x3", "conv1x1", "clone", "pnorm",
                "mul_ch", "silu", "TFLOP/s", "ovh/conv3");

    // decoder: batch 1, 512^2 down to 64^2.
    bench_rung("decoder", 1,  64, 512, iters, rng);
    bench_rung("decoder", 1, 128, 256, iters, rng);
    bench_rung("decoder", 1, 192, 128, iters, rng);
    bench_rung("decoder", 1, 256,  64, iters, rng);
    // base: batch 16, 64^2 down to 8^2.
    bench_rung("base",   16, 192,  64, iters, rng);
    bench_rung("base",   16, 384,  32, iters, rng);
    bench_rung("base",   16, 576,  16, iters, rng);
    bench_rung("base",   16, 768,   8, iters, rng);
    // base at the batch the DAG would use if a request straddled fewer windows.
    bench_rung("base",    1, 192,  64, iters, rng);
    bench_rung("base",    1, 768,   8, iters, rng);
    // coarse: batch 1, single level.
    bench_rung("coarse",  1, 128,  64, iters, rng);
}

// ─── end-to-end world benchmark ────────────────────────────────────────────

// The pipeline's tiling constants, duplicated here so the bench can report how
// many tiles each stage ran without world_pipeline.cpp having to export them.
constexpr int kCoarseTileSize    = 64,  kCoarseTileStride    = 48;
constexpr int kLatentTileSize    = 64,  kLatentTileStride    = 32;
constexpr int kDecoderTileSize   = 512, kDecoderTileStride   = 384;

std::int64_t floor_div_i(std::int64_t a, std::int64_t b) {
    const std::int64_t q = a / b, r = a % b;
    return (r != 0 && ((r < 0) != (b < 0))) ? q - 1 : q;
}
std::int64_t ceil_div_i(std::int64_t a, std::int64_t b) { return -floor_div_i(-a, b); }

// Windows of one axis touching [start, stop) for a given size/stride.
std::int64_t axis_windows(std::int64_t start, std::int64_t stop, int size, int stride) {
    const std::int64_t low  = ceil_div_i(start - size + 1, stride);
    const std::int64_t high = floor_div_i(stop - 1, stride);
    return high >= low ? high - low + 1 : 0;
}

// The CPU tail of elevation(): two Laplacian passes over the padded region, in
// double precision. Timed on synthetic buffers of exactly the shapes the real
// call uses, because measuring it through elevation() would fold in whatever
// the tile cache decided to recompute.
double time_laplacian(int ph, int pw, int lh, int lw) {
    std::vector<double> res(static_cast<std::size_t>(ph) * pw);
    std::vector<double> low(static_cast<std::size_t>(lh) * lw);
    for (std::size_t i = 0; i < res.size(); ++i) res[i] = 0.001 * static_cast<double>(i % 977);
    for (std::size_t i = 0; i < low.size(); ++i) low[i] = 0.01 * static_cast<double>(i % 331);

    const auto t0 = Clock::now();
    std::vector<double> new_low, elev;
    td::laplacian_denoise(res.data(), ph, pw, low.data(), lh, lw, 5.0, new_low);
    td::laplacian_decode(res.data(), ph, pw, new_low.data(), lh, lw, false, elev);
    return since(t0);
}

void bench_world(const std::string& dir, std::uint64_t seed,
                 const std::vector<int>& sizes, int repeat) {
    std::printf("\n== world  seed=%llu  tile cache %.0f MB (WorldPipeline's fixed default)\n",
                static_cast<unsigned long long>(seed),
                static_cast<double>(td::MemoryTileStore::kDefaultCacheBytes) /
                    (1024.0 * 1024.0));
    std::printf("   tiles are per-stage window counts; 'sum' is the stages plus the\n"
                "   Laplacian, i.e. the cost with every tile computed exactly once.\n"
                "   'elev' is one elevation() call on a cold pipeline — the number a\n"
                "   caller actually pays. elev > sum means the tile cache is thrashing.\n\n");
    std::printf("   %6s %7s %7s %7s | %7s %8s %8s %7s %7s %7s | %7s %10s %9s\n",
                "N", "coarse", "latent", "dec", "load s", "coarse s", "latent s",
                "dec s", "lap s", "sum s", "elev s", "cells/s", "km2/s");

    for (int N : sizes) {
        for (int r = 0; r < repeat; ++r) {
            const auto t_load = Clock::now();
            td::WorldPipeline pipe(dir, seed);
            bt::sync_all();
            const double load = since(t_load);
            const auto& cfg = pipe.config();

            const std::int64_t scale = cfg.latent_compression;
            // Mirror WorldPipeline::elevation's outward pad so the tile counts
            // and the timed regions match what elevation() will ask for.
            const int  kernel_size  = (static_cast<int>(5.0 * 2) / 2) * 2 + 1;
            const std::int64_t pad_hr = (kernel_size / 2 + 1) * scale;
            const std::int64_t pi1 = floor_div_i(0 - pad_hr, scale) * scale;
            const std::int64_t pj1 = pi1;
            const std::int64_t pi2 = ceil_div_i(N + pad_hr, scale) * scale;
            const std::int64_t pj2 = pi2;

            // One coarse cell spans latent_compression*32 native cells.
            const std::int64_t coarse_span = scale * 32;
            const std::int64_t ci1 = floor_div_i(pi1, coarse_span) - 1;
            const std::int64_t ci2 = ceil_div_i(pi2, coarse_span) + 1;

            const std::int64_t n_coarse =
                axis_windows(ci1, ci2, kCoarseTileSize, kCoarseTileStride);
            const std::int64_t n_latent =
                axis_windows(pi1 / scale, pi2 / scale, kLatentTileSize, kLatentTileStride);
            const std::int64_t n_dec =
                axis_windows(pi1, pi2, kDecoderTileSize, kDecoderTileStride);

            // Stage by stage on a warm upstream cache, so each column is that
            // stage's own cost rather than its cost plus everything under it.
            const auto t_c = Clock::now();
            pipe.coarse(ci1, ci1, ci2, ci2);
            bt::sync_all();
            const double t_coarse = since(t_c);

            const auto t_l = Clock::now();
            pipe.latent(pi1 / scale, pj1 / scale, pi2 / scale, pj2 / scale);
            bt::sync_all();
            const double t_latent = since(t_l);

            const auto t_d = Clock::now();
            pipe.residual(pi1, pj1, pi2, pj2);
            bt::sync_all();
            const double t_dec = since(t_d);

            const double t_lap = time_laplacian(static_cast<int>(pi2 - pi1),
                                                static_cast<int>(pj2 - pj1),
                                                static_cast<int>((pi2 - pi1) / scale),
                                                static_cast<int>((pj2 - pj1) / scale));

            // The honest end-to-end number: a second, cold pipeline asked for
            // the same region in one call.
            td::WorldPipeline fresh(dir, seed);
            const auto t_e = Clock::now();
            td::TileBuffer e = fresh.elevation(0, 0, N, N);
            const double t_elev = since(t_e);

            const double sum = t_coarse + t_latent + t_dec + t_lap;
            const double cells = static_cast<double>(N) * N;
            const double km2 = cells * cfg.native_resolution * cfg.native_resolution / 1e6;

            std::printf("   %6d %7lld %7lld %7lld | %7.2f %8.2f %8.2f %7.2f %7.2f %7.2f "
                        "| %7.2f %10.0f %9.2f\n",
                        N,
                        static_cast<long long>(n_coarse * n_coarse),
                        static_cast<long long>(n_latent * n_latent),
                        static_cast<long long>(n_dec * n_dec),
                        load, t_coarse, t_latent, t_dec, t_lap, sum,
                        t_elev, cells / t_elev, km2 / t_elev);
            std::fflush(stdout);
            (void)e;
        }
    }
}

// Raster a grid of adjacent regions out of one pipeline. Region (r, c) shares
// its whole west edge with (r, c-1) and its north edge with (r-1, c), so with a
// cache big enough to hold the overlap the later regions should come in under
// the first. If they do not, the cache is thrashing.
void bench_stream(const std::string& dir, std::uint64_t seed, int region, int grid) {
    td::WorldPipeline pipe(dir, seed);
    bt::sync_all();

    std::printf("\n== stream  seed=%llu  region=%d^2  grid=%dx%d\n",
                static_cast<unsigned long long>(seed), region, grid, grid);
    std::printf("   %6s %6s %10s %12s\n", "row", "col", "s", "cells/s");

    double first = 0.0, rest = 0.0;
    for (int r = 0; r < grid; ++r) {
        for (int c = 0; c < grid; ++c) {
            const std::int64_t i1 = static_cast<std::int64_t>(r) * region;
            const std::int64_t j1 = static_cast<std::int64_t>(c) * region;
            const auto t0 = Clock::now();
            td::TileBuffer e = pipe.elevation(i1, j1, i1 + region, j1 + region);
            const double dt = since(t0);
            (void)e;
            if (r == 0 && c == 0) first = dt; else rest += dt;
            std::printf("   %6d %6d %10.2f %12.0f\n", r, c, dt,
                        static_cast<double>(region) * region / dt);
            std::fflush(stdout);
        }
    }
    const int n_rest = grid * grid - 1;
    if (n_rest > 0) {
        std::printf("   first %.2fs, mean of the remaining %d: %.2fs\n",
                    first, n_rest, rest / n_rest);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: bench_terrain unet|world --weights <dir> [...]\n");
        return 2;
    }
    const std::string mode = argv[1];
    const char* w = arg_after(argc, argv, "--weights");
    if (!w && mode != "kernels") {   // kernels runs on synthetic tensors
        std::fprintf(stderr, "bench_terrain: --weights <converted dir> is required\n");
        return 2;
    }

    bt::init();
    std::printf("bench_terrain: device=%s dtype=%s\n",
                bt::default_device() == bt::Device::CPU ? "CPU" : "GPU",
                bt::compute_dtype() == bt::Dtype::FP16 ? "FP16" : "FP32");

    try {
        if (mode == "unet") {
            const char* it = arg_after(argc, argv, "--iters");
            bench_unet(w, int_list(arg_after(argc, argv, "--batches"), {1, 2, 4, 8, 16}),
                       it ? std::atoi(it) : 5);
        } else if (mode == "world") {
            const char* sd = arg_after(argc, argv, "--seed");
            const char* rp = arg_after(argc, argv, "--repeat");
            bench_world(w, sd ? std::strtoull(sd, nullptr, 10) : 42ull,
                        int_list(arg_after(argc, argv, "--sizes"), {512, 1024, 2048}),
                        rp ? std::atoi(rp) : 1);
        } else if (mode == "kernels") {
            const char* it = arg_after(argc, argv, "--iters");
            bench_kernels(it ? std::atoi(it) : 20);
        } else if (mode == "stream") {
            const char* sd = arg_after(argc, argv, "--seed");
            const char* rg = arg_after(argc, argv, "--region");
            const char* gd = arg_after(argc, argv, "--grid");
            bench_stream(w, sd ? std::strtoull(sd, nullptr, 10) : 42ull,
                         rg ? std::atoi(rg) : 512, gd ? std::atoi(gd) : 4);
        } else {
            std::fprintf(stderr, "bench_terrain: unknown mode '%s'\n", mode.c_str());
            return 2;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "bench_terrain: %s\n", e.what());
        return 1;
    }
    return 0;
}
