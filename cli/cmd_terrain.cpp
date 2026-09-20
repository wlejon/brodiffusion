// terrain-diffusion subcommands — infinite-world map generation.
//
// The map UNet forward, the tile-seeded portable RNG, the infinite-tensor
// evaluator, the synthetic climate map, the Laplacian primitives, the coarse
// world pipeline and the per-stage samplers. Each drives the matching
// scripts/terrain_*_parity.sh against the PyTorch reference.

#include "commands.h"

#include "brodiffusion/detail/compute.h"
#include "brodiffusion/terrain/mp_unet.h"
#include "brodiffusion/terrain/portable_rng.h"
#include "brodiffusion/terrain/sampler.h"
#include "brodiffusion/terrain/infinite_tensor.h"
#include "brodiffusion/terrain/synthetic_map.h"
#include "brodiffusion/terrain/laplacian.h"
#include "brodiffusion/terrain/world_pipeline.h"

#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace brodiffusion::cli {

namespace st = brotensor::safetensors;

namespace {

std::vector<float> load_raw_f32(const char* path, size_t n) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error(std::string("cannot open ") + path);
    std::vector<float> v(n);
    in.read(reinterpret_cast<char*>(v.data()),
            static_cast<std::streamsize>(n * sizeof(float)));
    if (!in) throw std::runtime_error(std::string("short read from ") + path);
    return v;
}

}  // namespace

int run_terrain_unet_fwd(int argc, char** argv) {
    const char* w   = arg_after(argc, argv, "--weights");   // converted -bro dir
    const char* stg = arg_after(argc, argv, "--stage");     // coarse | base | decoder
    const char* xp  = arg_after(argc, argv, "--x");         // (1, C_in, S, S) raw f32
    const char* np_ = arg_after(argc, argv, "--noise");     // (1,) raw f32
    const char* cp  = arg_after(argc, argv, "--cond");      // flattened conditioning
    const char* ss  = arg_after(argc, argv, "--size");
    const char* op  = arg_after(argc, argv, "--out");
    if (!w || !stg || !xp || !np_ || !ss || !op) {
        std::fprintf(stderr, "terrain-unet-fwd: need --weights --stage --x "
                             "--noise --size --out [--cond]\n");
        return 2;
    }
    const int S = std::atoi(ss);
    const std::string dir = w;

    brotensor::init();
    auto cfg = brodiffusion::terrain::MPUNetConfig::from_config_json(
        dir + "/config.json", stg);
    brodiffusion::terrain::MPUNet net(cfg);
    auto f = st::File::open(dir + "/" + stg + ".safetensors");
    net.load_weights(f);

    auto xh = load_raw_f32(xp, static_cast<size_t>(cfg.in_channels) * S * S);
    auto nh = load_raw_f32(np_, 1);
    brotensor::Tensor x = brodiffusion::detail::upload_host(
        xh.data(), 1, cfg.in_channels * S * S);

    // The flat --cond file is split per conditional input: one scalar for a
    // 'float' input, `dim` values for a 'tensor' input.
    std::vector<std::vector<float>> cond;
    if (!cfg.conditional_inputs.empty()) {
        if (!cp) throw std::runtime_error("terrain-unet-fwd: --cond is required for this stage");
        size_t total = 0;
        for (const auto& ci : cfg.conditional_inputs) {
            total += (ci.kind == "tensor") ? static_cast<size_t>(ci.dim) : 1u;
        }
        auto ch = load_raw_f32(cp, total);
        size_t off = 0;
        for (const auto& ci : cfg.conditional_inputs) {
            const size_t n = (ci.kind == "tensor") ? static_cast<size_t>(ci.dim) : 1u;
            cond.emplace_back(ch.begin() + static_cast<std::ptrdiff_t>(off),
                              ch.begin() + static_cast<std::ptrdiff_t>(off + n));
            off += n;
        }
    }

    brotensor::Tensor out;
    net.forward(x, /*N=*/1, S, nh.data(), cond, out);
    brotensor::sync_all();
    dump_latent_f32(op, out);
    std::printf("terrain-unet-fwd: stage=%s size=%d in=%d out=%d -> (1,%d,%d,%d)\n",
                stg, S, cfg.in_channels, cfg.out_channels,
                cfg.out_channels, S, S);
    return 0;
}

// Sample a window of the infinite tile-seeded Gaussian noise field, and
// optionally self-check the invariants the whole InfiniteDiffusion design rests
// on. See brodiffusion/terrain/portable_rng.h.
int run_terrain_rng(int argc, char** argv) {
    namespace td = brodiffusion::terrain;
    const char* op = arg_after(argc, argv, "--out");
    const char* sd = arg_after(argc, argv, "--seed");
    const char* y0 = arg_after(argc, argv, "--y0");
    const char* x0 = arg_after(argc, argv, "--x0");
    const char* hh = arg_after(argc, argv, "--h");
    const char* ww = arg_after(argc, argv, "--w");
    const char* cc = arg_after(argc, argv, "--channels");
    const char* tl = arg_after(argc, argv, "--tile");
    bool selfcheck = false;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--selfcheck") == 0) selfcheck = true;
    }

    const std::uint64_t seed = sd ? std::strtoull(sd, nullptr, 10) : 12345ULL;
    const std::int64_t  Y = y0 ? std::strtoll(y0, nullptr, 10) : 0;
    const std::int64_t  X = x0 ? std::strtoll(x0, nullptr, 10) : 0;
    const int H = hh ? std::atoi(hh) : 64;
    const int W = ww ? std::atoi(ww) : 64;
    const int C = cc ? std::atoi(cc) : 1;
    const int T = tl ? std::atoi(tl) : 256;

    std::vector<float> scratch(static_cast<size_t>(C) * T * T);
    std::vector<float> out(static_cast<size_t>(C) * H * W);
    td::gaussian_noise_patch(seed, Y, X, H, W, C, T, T, out.data(), scratch.data());

    if (op) {
        std::FILE* f = std::fopen(op, "wb");
        if (!f) throw std::runtime_error("terrain-rng: cannot open --out");
        std::fwrite(out.data(), sizeof(float), out.size(), f);
        std::fclose(f);
    }

    if (selfcheck) {
        // The load-bearing invariant: two windows that overlap must agree
        // EXACTLY on their intersection, or blended tiles show seams. Offset a
        // second window so it straddles a different set of tile boundaries.
        const std::int64_t oy = Y + H / 3, ox = X + W / 4;
        std::vector<float> other(static_cast<size_t>(C) * H * W);
        td::gaussian_noise_patch(seed, oy, ox, H, W, C, T, T, other.data(), scratch.data());
        size_t compared = 0, mismatched = 0;
        for (int c = 0; c < C; ++c) {
            for (std::int64_t y = oy; y < Y + H; ++y) {
                for (std::int64_t x = ox; x < X + W; ++x) {
                    const float a = out[(static_cast<size_t>(c) * H + (y - Y)) * W + (x - X)];
                    const float b = other[(static_cast<size_t>(c) * H + (y - oy)) * W + (x - ox)];
                    ++compared;
                    // Bit-exact, not approximate — the field is a pure function
                    // of world position, so anything but equality is a bug.
                    if (a != b) ++mismatched;
                }
            }
        }
        std::printf("terrain-rng: seam overlap %zu px, %zu mismatched\n", compared, mismatched);
        if (mismatched != 0) {
            std::fprintf(stderr, "terrain-rng: SEAM CHECK FAILED\n");
            return 1;
        }
        std::printf("terrain-rng: SEAM CHECK OK\n");
    }

    double mean = 0.0, m2 = 0.0;
    for (float v : out) mean += v;
    mean /= static_cast<double>(out.size());
    for (float v : out) m2 += (v - mean) * (v - mean);
    std::printf("terrain-rng: seed=%llu (%lld,%lld) %dx%dx%d tile=%d mean=%+.6f std=%.6f\n",
                static_cast<unsigned long long>(seed),
                static_cast<long long>(Y), static_cast<long long>(X),
                C, H, W, T, mean, std::sqrt(m2 / static_cast<double>(out.size())));
    return 0;
}

// Run the infinite-tensor evaluator over a fixed two-node DAG whose compute
// functions return only small integers, so every value — including the sums
// produced when overlapping windows accumulate — is exact in float32 and can be
// compared bit-for-bit against scripts/terrain_itensor_ref.py. See
// brodiffusion/terrain/infinite_tensor.h.
//
// The scenario mirrors terrain's real coarse->latent edge in miniature: A's
// output windows overlap, B's overlap too, and B reads A through a window with a
// negative offset and unit stride.
int run_terrain_itensor(int argc, char** argv) {
    namespace td = brodiffusion::terrain;
    const char* op = arg_after(argc, argv, "--out");
    const char* bs = arg_after(argc, argv, "--batch");
    const char* cs = arg_after(argc, argv, "--case");
    const std::int64_t batch = bs ? std::strtoll(bs, nullptr, 10) : 1;
    const int          kase  = cs ? std::atoi(cs) : 0;

    // Floored modulo. C++ `%` truncates toward zero, so it goes NEGATIVE for the
    // negative window indices these cases deliberately exercise, while Python's
    // `%` floors and stays non-negative. Using `%` here would make the C++ and
    // the reference disagree on window content — a bug in the test rather than
    // in the library, and a confusing one to chase.
    auto floor_mod = [](std::int64_t a, std::int64_t m) -> std::int64_t {
        const std::int64_t r = a % m;
        return (r < 0) ? r + m : r;
    };

    auto f_a = [&](const std::vector<std::vector<std::int64_t>>& widx,
                   const std::vector<std::vector<td::TileBuffer>>&) {
        std::vector<td::TileBuffer> out;
        for (const auto& w : widx) {
            td::TileBuffer t;
            t.shape = {2, 4, 4};
            t.data.resize(2 * 4 * 4);
            const std::int64_t v = floor_mod(w[1] * 7 + w[2] * 13, 32);
            for (int c = 0; c < 2; ++c)
                for (int y = 0; y < 4; ++y)
                    for (int x = 0; x < 4; ++x)
                        t.data[(static_cast<std::size_t>(c) * 4 + y) * 4 + x] =
                            static_cast<float>(v + c * 64 + y * 2 + x);
            out.push_back(std::move(t));
        }
        return out;
    };

    auto f_b = [&](const std::vector<std::vector<std::int64_t>>& widx,
                   const std::vector<std::vector<td::TileBuffer>>& args) {
        std::vector<td::TileBuffer> out;
        for (std::size_t b = 0; b < widx.size(); ++b) {
            double s = 0.0;
            for (float v : args.at(0).at(b).data) s += v;
            td::TileBuffer t;
            t.shape = {2, 4, 4};
            t.data.resize(2 * 4 * 4);
            for (int c = 0; c < 2; ++c)
                for (int y = 0; y < 4; ++y)
                    for (int x = 0; x < 4; ++x)
                        t.data[(static_cast<std::size_t>(c) * 4 + y) * 4 + x] =
                            static_cast<float>(s + c * 8 + y + x * 3);
            out.push_back(std::move(t));
        }
        return out;
    };

    struct Case { std::int64_t c0, c1, y0, y1, x0, x1; };
    static const Case kCases[] = {
        {0, 2,   0, 10,   0, 10},
        {0, 2,  -7,  5, -13, -1},
        {0, 2,   5,  9,   5,  9},
        {0, 2, -40, -32,  24, 32},
    };
    if (kase < 0 || kase >= static_cast<int>(sizeof(kCases) / sizeof(kCases[0])))
        throw std::runtime_error("terrain-itensor: --case out of range");
    const Case& K = kCases[kase];

    td::MemoryTileStore store;
    td::InfiniteTensor A({2, -1, -1}, f_a, td::TensorWindow({2, 4, 4}, {2, 3, 3}),
                         {}, {}, 1, &store, "A");
    td::InfiniteTensor B({2, -1, -1}, f_b, td::TensorWindow({2, 4, 4}, {2, 2, 2}),
                         {&A}, {td::TensorWindow({2, 3, 3}, {2, 1, 1}, {0, -1, -1})},
                         batch, &store, "B");

    const std::vector<td::Slice> req = {{K.c0, K.c1}, {K.y0, K.y1}, {K.x0, K.x1}};
    td::TileBuffer r = B(req);
    // Read the identical range again. Every window is already processed, so this
    // must return exactly the same values — if accumulation happened at write
    // time rather than in read_pixels, the second read would come back doubled.
    td::TileBuffer r2 = B(req);
    if (r.data != r2.data) {
        std::fprintf(stderr, "terrain-itensor: REPEATED READ NOT IDEMPOTENT\n");
        return 1;
    }

    if (op) {
        std::FILE* f = std::fopen(op, "wb");
        if (!f) throw std::runtime_error("terrain-itensor: cannot open --out");
        std::fwrite(r.data.data(), sizeof(float), r.data.size(), f);
        std::fclose(f);
    }

    double sum = 0.0, lo = r.data.empty() ? 0.0 : r.data[0], hi = lo;
    for (float v : r.data) { sum += v; if (v < lo) lo = v; if (v > hi) hi = v; }
    std::printf("terrain-itensor: case %d batch %lld shape [%lld,%lld,%lld] "
                "sum %.1f min %.1f max %.1f\n",
                kase, static_cast<long long>(batch),
                static_cast<long long>(r.shape[0]), static_cast<long long>(r.shape[1]),
                static_cast<long long>(r.shape[2]), sum, lo, hi);
    return 0;
}

// Sample a window of the five-channel synthetic climate map that conditions the
// coarse stage. See brodiffusion/terrain/synthetic_map.h.
int run_terrain_synth(int argc, char** argv) {
    namespace td = brodiffusion::terrain;
    const char* sp = arg_after(argc, argv, "--stats");
    const char* op = arg_after(argc, argv, "--out");
    const char* sd = arg_after(argc, argv, "--seed");
    const char* a1 = arg_after(argc, argv, "--i1");
    const char* b1 = arg_after(argc, argv, "--j1");
    const char* a2 = arg_after(argc, argv, "--i2");
    const char* b2 = arg_after(argc, argv, "--j2");
    bool raw = false;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--raw") == 0) raw = true;
    }
    if (!sp) throw std::runtime_error("terrain-synth: --stats is required");

    const std::int64_t I1 = a1 ? std::strtoll(a1, nullptr, 10) : 0;
    const std::int64_t J1 = b1 ? std::strtoll(b1, nullptr, 10) : 0;
    const std::int64_t I2 = a2 ? std::strtoll(a2, nullptr, 10) : 64;
    const std::int64_t J2 = b2 ? std::strtoll(b2, nullptr, 10) : 64;
    if (I2 <= I1 || J2 <= J1) throw std::runtime_error("terrain-synth: empty window");

    td::SyntheticMapConfig cfg;
    cfg.seed = sd ? static_cast<std::uint32_t>(std::strtoul(sd, nullptr, 10)) : 0u;

    td::SyntheticMap map(td::SyntheticMapStats::load(sp), cfg);

    const std::size_t plane = static_cast<std::size_t>(I2 - I1) * static_cast<std::size_t>(J2 - J1);
    std::vector<float> out(plane * td::kSyntheticChannels);
    if (raw) map.sample_raw(I1, J1, I2, J2, out.data());
    else     map.sample(I1, J1, I2, J2, out.data());

    if (op) {
        std::FILE* f = std::fopen(op, "wb");
        if (!f) throw std::runtime_error("terrain-synth: cannot open --out");
        std::fwrite(out.data(), sizeof(float), out.size(), f);
        std::fclose(f);
    }

    static const char* kNames[td::kSyntheticChannels] = {
        "elev", "temp", "temp_std", "precip", "precip_std"};
    std::printf("terrain-synth: seed=%u (%lld,%lld)-(%lld,%lld)%s\n",
                cfg.seed, static_cast<long long>(I1), static_cast<long long>(J1),
                static_cast<long long>(I2), static_cast<long long>(J2),
                raw ? " raw" : "");
    for (int c = 0; c < td::kSyntheticChannels; ++c) {
        const float* p = out.data() + static_cast<std::size_t>(c) * plane;
        double mean = 0.0, lo = p[0], hi = p[0];
        for (std::size_t i = 0; i < plane; ++i) {
            mean += p[i];
            if (p[i] < lo) lo = p[i];
            if (p[i] > hi) hi = p[i];
        }
        std::printf("  %-11s mean %+12.4f  range [%+12.4f, %+12.4f]\n",
                    kNames[c], mean / static_cast<double>(plane), lo, hi);
    }
    return 0;
}

// Exercise the Laplacian primitives on a deterministic input so
// scripts/terrain_laplacian_parity.sh can drive the same data through
// torchvision. These need a gate of their own: the composed elevation output is
// nearly insensitive to them, because the sigma=5 blur that follows removes most
// of what the resampling choice affects. A wrong resampler shows up here at 1e-1
// and in the elevation at 1e-3, under its bar.
int run_terrain_laplacian(int argc, char** argv) {
    namespace td = brodiffusion::terrain;
    const char* op = arg_after(argc, argv, "--out");
    const char* wh = arg_after(argc, argv, "--op");
    if (!op || !wh) {
        std::fprintf(stderr, "terrain-laplacian: need --op <resize|blur|extrap|denoise> --out F "
                             "[--ih N --iw N --oh N --ow N]\n");
        return 2;
    }
    auto opt = [&](const char* k, int d) { const char* v = arg_after(argc, argv, k);
                                          return v ? std::atoi(v) : d; };
    const int ih = opt("--ih", 64), iw = opt("--iw", 64);
    const int oh = opt("--oh", 8),  ow = opt("--ow", 8);

    // A deterministic, non-separable, non-symmetric input: anything symmetric
    // would hide an axis swap, and anything separable would hide a mixed-up
    // horizontal/vertical pass.
    std::vector<double> src(static_cast<std::size_t>(ih) * iw);
    for (int y = 0; y < ih; ++y) {
        for (int x = 0; x < iw; ++x) {
            src[static_cast<std::size_t>(y) * iw + x] =
                std::sin(0.13 * x + 0.29 * y) + 0.4 * std::cos(0.07 * x * y + 1.0) +
                0.001 * (x * x - y);
        }
    }

    std::vector<double> out;
    const std::string what = wh;
    if (what == "resize") {
        td::resize_bilinear(src.data(), ih, iw, oh, ow, out);
    } else if (what == "blur") {
        td::gaussian_blur(src.data(), ih, iw, 11, 5.0, out);
    } else if (what == "extrap") {
        td::resize_extrapolated(src.data(), ih, iw, oh, ow, out);
    } else if (what == "denoise") {
        // residual at (ih, iw), low band at (oh, ow)
        std::vector<double> low(static_cast<std::size_t>(oh) * ow);
        for (int y = 0; y < oh; ++y) {
            for (int x = 0; x < ow; ++x) {
                low[static_cast<std::size_t>(y) * ow + x] =
                    std::cos(0.31 * x - 0.17 * y) + 0.2 * x;
            }
        }
        std::vector<double> nl;
        td::laplacian_denoise(src.data(), ih, iw, low.data(), oh, ow, 5.0, nl);
        std::vector<double> dec;
        td::laplacian_decode(src.data(), ih, iw, nl.data(), oh, ow, false, dec);
        out = dec;
    } else {
        std::fprintf(stderr, "terrain-laplacian: unknown --op %s\n", wh);
        return 2;
    }

    std::vector<float> f(out.begin(), out.end());
    std::ofstream o(op, std::ios::binary);
    if (!o) { std::fprintf(stderr, "terrain-laplacian: cannot write output\n"); return 1; }
    o.write(reinterpret_cast<const char*>(f.data()),
            static_cast<std::streamsize>(f.size() * sizeof(float)));
    std::printf("terrain-laplacian: %s -> %zu values\n", wh, f.size());
    return 0;
}

// The coarse world map over a requested cell range, as raw LE f32. Emits the
// weight-normalized 6-channel form by default (what you would look at); --raw
// emits the 7-channel weighted form the next stage consumes. Drives
// scripts/terrain_coarse_parity.sh against the PyTorch WorldPipeline.
int run_terrain_coarse(int argc, char** argv) {
    namespace td = brodiffusion::terrain;
    const char* w  = arg_after(argc, argv, "--weights");
    const char* sd = arg_after(argc, argv, "--seed");
    const char* op = arg_after(argc, argv, "--out");
    if (!w || !sd || !op) {
        std::fprintf(stderr, "terrain-coarse: need --weights --seed --out "
                             "[--i1 --j1 --i2 --j2] [--raw]\n");
        return 2;
    }
    bool raw = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--raw") == 0) raw = true;
    }
    auto opt = [&](const char* k, std::int64_t dflt) -> std::int64_t {
        const char* v = arg_after(argc, argv, k);
        return v ? std::strtoll(v, nullptr, 10) : dflt;
    };
    const std::int64_t i1 = opt("--i1", 0), j1 = opt("--j1", 0);
    const std::int64_t i2 = opt("--i2", 64), j2 = opt("--j2", 64);
    if (i2 <= i1 || j2 <= j1) {
        std::fprintf(stderr, "terrain-coarse: empty range\n");
        return 2;
    }

    brotensor::init();
    td::WorldPipeline pipe(w, std::strtoull(sd, nullptr, 10));
    bool latent = false, latent_init = false, residual = false, elev = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--latent") == 0) latent = true;
        if (std::strcmp(argv[i], "--latent-init") == 0) latent_init = true;
        if (std::strcmp(argv[i], "--residual") == 0) residual = true;
        if (std::strcmp(argv[i], "--elev") == 0) elev = true;
    }
    if (elev) {
        td::TileBuffer e = pipe.elevation(i1, j1, i2, j2);
        std::ofstream ef(op, std::ios::binary);
        if (!ef) { std::fprintf(stderr, "terrain-coarse: cannot write file\n"); return 1; }
        ef.write(reinterpret_cast<const char*>(e.data.data()),
                 static_cast<std::streamsize>(e.data.size() * sizeof(float)));
        std::printf("terrain-coarse: elev (%lld, %lld) -> %s\n",
                    static_cast<long long>(e.shape[1]),
                    static_cast<long long>(e.shape[2]), op);
        return 0;
    }
    if (latent_init) {
        td::TileBuffer li = pipe.latent_init(i1, j1, i2, j2);
        std::ofstream lf(op, std::ios::binary);
        if (!lf) { std::fprintf(stderr, "terrain-coarse: cannot write %s\n", op); return 1; }
        // Weight-normalize here rather than adding another public entry point;
        // the raw form of the intermediate step has no consumer.
        const std::size_t pl = static_cast<std::size_t>(li.shape[1]) * li.shape[2];
        const float* wsum = li.data.data() + 5 * pl;
        std::vector<float> nrm(5 * pl);
        for (int c = 0; c < 5; ++c) {
            for (std::size_t k = 0; k < pl; ++k) nrm[c * pl + k] = li.data[c * pl + k] / wsum[k];
        }
        lf.write(reinterpret_cast<const char*>(nrm.data()),
                 static_cast<std::streamsize>(nrm.size() * sizeof(float)));
        std::printf("terrain-coarse: latent-init (5, %lld, %lld) -> %s\n",
                    static_cast<long long>(li.shape[1]),
                    static_cast<long long>(li.shape[2]), op);
        return 0;
    }
    if (residual) {
        td::TileBuffer r = raw ? pipe.residual(i1, j1, i2, j2)
                               : pipe.residual_normalized(i1, j1, i2, j2);
        std::ofstream rf(op, std::ios::binary);
        if (!rf) { std::fprintf(stderr, "terrain-coarse: cannot write %s\n", op); return 1; }
        rf.write(reinterpret_cast<const char*>(r.data.data()),
                 static_cast<std::streamsize>(r.data.size() * sizeof(float)));
        std::printf("terrain-coarse: residual (%lld, %lld, %lld) -> %s\n",
                    static_cast<long long>(r.shape[0]),
                    static_cast<long long>(r.shape[1]),
                    static_cast<long long>(r.shape[2]), op);
        return 0;
    }
    td::TileBuffer out =
        latent ? (raw ? pipe.latent(i1, j1, i2, j2)
                      : pipe.latent_normalized(i1, j1, i2, j2))
               : (raw ? pipe.coarse(i1, j1, i2, j2)
                      : pipe.coarse_normalized(i1, j1, i2, j2));

    std::ofstream f(op, std::ios::binary);
    if (!f) { std::fprintf(stderr, "terrain-coarse: cannot write %s\n", op); return 1; }
    f.write(reinterpret_cast<const char*>(out.data.data()),
            static_cast<std::streamsize>(out.data.size() * sizeof(float)));
    std::printf("terrain-coarse: (%lld, %lld, %lld) -> %s\n",
                static_cast<long long>(out.shape[0]),
                static_cast<long long>(out.shape[1]),
                static_cast<long long>(out.shape[2]), op);
    return 0;
}

// Full denoise for one terrain-diffusion stage: DPM-Solver++ for `coarse`,
// TrigFlow consistency for `base` / `decoder`. Inputs are supplied as raw LE
// f32 so scripts/terrain_sampler_parity.sh can drive the same noise through the
// PyTorch reference and this port.
int run_terrain_sample(int argc, char** argv) {
    namespace td = brodiffusion::terrain;
    const char* w   = arg_after(argc, argv, "--weights");   // converted -bro dir
    const char* stg = arg_after(argc, argv, "--stage");     // coarse | base | decoder
    const char* np_ = arg_after(argc, argv, "--noise");     // per-step noise field(s)
    const char* cp  = arg_after(argc, argv, "--cond");      // coarse conditioning image
    const char* cip = arg_after(argc, argv, "--condin");    // conditional_inputs, flat
    const char* lp  = arg_after(argc, argv, "--latents");   // decoder latents
    const char* ss  = arg_after(argc, argv, "--size");
    const char* op  = arg_after(argc, argv, "--out");
    if (!w || !stg || !np_ || !ss || !op) {
        std::fprintf(stderr, "terrain-sample: need --weights --stage --noise --size "
                             "--out [--cond] [--condin] [--latents]\n");
        return 2;
    }
    const std::string stage = stg;
    if (stage != "coarse" && stage != "base" && stage != "decoder") {
        std::fprintf(stderr, "terrain-sample: --stage must be coarse|base|decoder\n");
        return 2;
    }
    const int S = std::atoi(ss);
    const std::string dir = w;

    brotensor::init();
    auto cfg = td::MPUNetConfig::from_config_json(dir + "/config.json", stg);
    td::MPUNet net(cfg);
    auto f = st::File::open(dir + "/" + stg + ".safetensors");
    net.load_weights(f);

    const std::size_t plane   = static_cast<std::size_t>(S) * S;
    const std::size_t n_out   = static_cast<std::size_t>(cfg.out_channels) * plane;
    const std::size_t n_extra =
        static_cast<std::size_t>(cfg.in_channels - cfg.out_channels) * plane;

    // conditional_inputs: one scalar per 'float' entry, `dim` values per 'tensor'.
    std::vector<std::vector<float>> cond;
    if (!cfg.conditional_inputs.empty()) {
        if (!cip) throw std::runtime_error("terrain-sample: --condin is required for this stage");
        std::size_t total = 0;
        for (const auto& ci : cfg.conditional_inputs) {
            total += (ci.kind == "tensor") ? static_cast<std::size_t>(ci.dim) : 1u;
        }
        auto ch = load_raw_f32(cip, total);
        std::size_t off = 0;
        for (const auto& ci : cfg.conditional_inputs) {
            const std::size_t n = (ci.kind == "tensor") ? static_cast<std::size_t>(ci.dim) : 1u;
            cond.emplace_back(ch.begin() + static_cast<std::ptrdiff_t>(off),
                              ch.begin() + static_cast<std::ptrdiff_t>(off + n));
            off += n;
        }
    }

    std::vector<float> out;
    int steps = 0;
    if (stage == "coarse") {
        steps = 20;
        auto noise = load_raw_f32(np_, n_out);
        std::vector<float> cimg;
        if (n_extra) {
            if (!cp) throw std::runtime_error("terrain-sample: --cond is required for the coarse stage");
            cimg = load_raw_f32(cp, n_extra);
        }
        td::sample_coarse(net, noise.data(), n_extra ? cimg.data() : nullptr,
                          cond, S, steps, out);
    } else {
        const auto t_list = td::trigflow_t_list(/*two_step=*/stage == "base");
        steps = static_cast<int>(t_list.size());
        auto noise = load_raw_f32(np_, n_out * t_list.size());
        std::vector<float> lat;
        if (n_extra) {
            if (!lp) throw std::runtime_error("terrain-sample: --latents is required for this stage");
            lat = load_raw_f32(lp, n_extra);
        }
        td::sample_trigflow(net, t_list, noise.data(), n_extra ? lat.data() : nullptr,
                            cond, S, out);
    }

    std::ofstream of(op, std::ios::binary | std::ios::trunc);
    if (!of) throw std::runtime_error(std::string("cannot open --out: ") + op);
    of.write(reinterpret_cast<const char*>(out.data()),
             static_cast<std::streamsize>(out.size() * sizeof(float)));

    std::printf("terrain-sample: stage=%s size=%d steps=%d -> (1,%d,%d,%d)\n",
                stg, S, steps, cfg.out_channels, S, S);
    return 0;
}

}  // namespace brodiffusion::cli
