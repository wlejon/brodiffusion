// Terrain world determinism on the GPU.
//
// The world is a pure function of (seed, position). This test reads each
// stage of the pipeline twice from a cold cache with the SAME request and
// requires the two reads to be bit-identical: identical inputs through an
// identical launch sequence must give identical bits, so any difference here
// is a race or uninitialised read, not rounding.
//
// It then reads regions with a different cache state or as part of a LARGER
// request. That changes how many windows the latent stage batches into one
// forward, and the batch size picks the GEMM kernels (brotensor's FP16 linear
// is a split-K GEMV for B <= 4), so those reads are only required to agree to
// within FP16 rounding, and the test prints the drift.
//
// Last, the latent network directly: sample 0's output must be bit-identical
// whatever its batch-mates are (a difference would be a cross-sample read,
// i.e. a real bug), and within rounding across batch sizes.
//
// Gated on weights/terrain-diffusion-30m-bro (BRODIFFUSION_TERRAIN_DIR
// overrides); skips cleanly when absent.

#define _CRT_SECURE_NO_WARNINGS  // std::getenv

#include "brodiffusion/terrain/world_pipeline.h"
#include "brodiffusion/detail/compute.h"

#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>

#ifndef BRODIFFUSION_WEIGHTS_DIR
#define BRODIFFUSION_WEIGHTS_DIR ""
#endif

namespace tr = brodiffusion::terrain;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

namespace {

// Count of elements whose bits differ, and the largest absolute difference.
struct Diff { long long bits = 0; double max_abs = 0.0; };

Diff compare(const tr::TileBuffer& a, const tr::TileBuffer& b) {
    Diff d;
    if (a.shape != b.shape || a.data.size() != b.data.size()) {
        d.bits = -1;
        return d;
    }
    for (std::size_t i = 0; i < a.data.size(); ++i) {
        if (std::memcmp(&a.data[i], &b.data[i], sizeof(float)) != 0) {
            ++d.bits;
            d.max_abs = std::fmax(d.max_abs, std::fabs(double(a.data[i]) - double(b.data[i])));
        }
    }
    return d;
}

// The same cells of `inner` (h x w) inside `outer` (H x W), per channel.
Diff compareInside(const tr::TileBuffer& inner, const tr::TileBuffer& outer,
                   std::int64_t oy = 0, std::int64_t ox = 0) {
    Diff d;
    const std::int64_t C = inner.shape[0], h = inner.shape[1], w = inner.shape[2];
    const std::int64_t H = outer.shape[1], W = outer.shape[2];
    for (std::int64_t c = 0; c < C; ++c)
        for (std::int64_t y = 0; y < h; ++y)
            for (std::int64_t x = 0; x < w; ++x) {
                const float u = inner.data[(c * h + y) * w + x];
                const float v = outer.data[(c * H + y + oy) * W + x + ox];
                if (std::memcmp(&u, &v, sizeof(float)) != 0) {
                    ++d.bits;
                    d.max_abs = std::fmax(d.max_abs, std::fabs(double(u) - double(v)));
                }
            }
    (void)H;
    return d;
}

// The 8x8 cells at (oy, ox) of `t`, all channels.
tr::TileBuffer compareSlice(const tr::TileBuffer& t, std::int64_t oy, std::int64_t ox) {
    const std::int64_t C = t.shape[0], H = t.shape[1], W = t.shape[2];
    tr::TileBuffer s;
    s.shape = {C, 8, 8};
    s.data.resize(static_cast<std::size_t>(C * 64));
    for (std::int64_t c = 0; c < C; ++c)
        for (std::int64_t y = 0; y < 8; ++y)
            for (std::int64_t x = 0; x < 8; ++x)
                s.data[static_cast<std::size_t>((c * 8 + y) * 8 + x)] =
                    t.data[static_cast<std::size_t>((c * H + y + oy) * W + x + ox)];
    return s;
}

template <class Read>
void twice(tr::WorldPipeline& world, const char* what, Read read) {
    world.clear_cache();
    const tr::TileBuffer a = read();
    world.clear_cache();
    const tr::TileBuffer b = read();
    const Diff d = compare(a, b);
    std::printf("  %-10s cold re-read: %lld of %zu values differ (max %.3g)\n",
                what, d.bits, a.data.size(), d.max_abs);
    CHECK(d.bits == 0);
}

}  // namespace

int main() {
    std::filesystem::path dir;
    if (const char* e = std::getenv("BRODIFFUSION_TERRAIN_DIR"); e && *e) dir = e;
    else dir = std::filesystem::path(BRODIFFUSION_WEIGHTS_DIR) / "terrain-diffusion-30m-bro";
    if (!std::filesystem::exists(dir / "config.json")) {
        std::printf("terrain_determinism: skipped (no %s)\n", dir.string().c_str());
        return 0;
    }
    brotensor::init();

    try {
        tr::WorldPipeline world(dir.string(), 1234);

        twice(world, "coarse",   [&] { return world.coarse(-2, -2, 2, 2); });
        twice(world, "latent0",  [&] { return world.latent_init(0, 0, 8, 8); });
        twice(world, "latent",   [&] { return world.latent(0, 0, 8, 8); });
        twice(world, "residual", [&] { return world.residual(0, 0, 16, 16); });
        twice(world, "elevation",[&] { return world.elevation(0, 0, 16, 16); });

        // A second pipeline in the same process, same seed: bit-identical too.
        {
            world.clear_cache();
            const tr::TileBuffer a = world.elevation(0, 0, 16, 16);
            tr::WorldPipeline other(dir.string(), 1234);
            const tr::TileBuffer b = other.elevation(0, 0, 16, 16);
            const Diff d = compare(a, b);
            std::printf("  second pipeline: %lld values differ (max %.3g)\n", d.bits, d.max_abs);
            CHECK(d.bits == 0);
        }

        // Repeated cold reads: a race would show up intermittently.
        {
            world.clear_cache();
            const tr::TileBuffer ref = world.elevation(0, 0, 16, 16);
            long long worst = 0;
            for (int k = 0; k < 6; ++k) {
                world.clear_cache();
                const Diff d = compare(ref, world.elevation(0, 0, 16, 16));
                if (d.bits != 0) worst = d.bits;
            }
            std::printf("  six more cold elevation reads: worst %lld values differ\n", worst);
            CHECK(worst == 0);
        }

        // Warm vs cold: the same region with neighbours' tiles already cached
        // batches fewer windows per forward. Same bound as below.
        {
            world.clear_cache();
            const tr::TileBuffer cold = world.elevation(0, 0, 16, 16);
            world.clear_cache();
            (void)world.latent(0, 0, 8, 8);
            (void)world.residual(0, 0, 16, 16);
            const tr::TileBuffer warm = world.elevation(0, 0, 16, 16);
            const Diff d = compare(cold, warm);
            std::printf("  elevation warm vs cold: %lld of %zu values differ (max %.3g m)\n",
                        d.bits, cold.data.size(), d.max_abs);
            CHECK(d.max_abs < 0.25);

            world.clear_cache();
            const tr::TileBuffer ls = world.latent(0, 0, 8, 8);
            world.clear_cache();
            const tr::TileBuffer lb = world.latent(0, 0, 16, 16);
            const Diff dl = compareInside(ls, lb);
            std::printf("  latent inside a larger read: %lld of %zu values differ (max %.3g)\n",
                        dl.bits, ls.data.size(), dl.max_abs);
            // [0,8)^2 needs 4 latent windows, [-8,8)^2 needs 9, [-40,8)^2 16.
            world.clear_cache();
            const tr::TileBuffer l9 = world.latent(-8, -8, 8, 8);
            world.clear_cache();
            const tr::TileBuffer l16 = world.latent(-40, -40, 8, 8);
            const Diff d49 = compareInside(ls, l9, 8, 8);
            const Diff d916 = compareInside(ls, l16, 40, 40);
            const Diff d9v16 = compareInside(compareSlice(l9, 8, 8), l16, 40, 40);
            std::printf("  latent, batch 4 vs 9: %lld differ (max %.3g); 4 vs 16: %lld differ; 9 vs 16: %lld differ\n",
                        d49.bits, d49.max_abs, d916.bits, d9v16.bits);
            CHECK(d49.max_abs < 0.01 && d916.max_abs < 0.01);
        }

        // Inside a larger read: batch shapes change, so only FP16-rounding
        // agreement is required.
        {
            world.clear_cache();
            const tr::TileBuffer small = world.elevation(0, 0, 16, 16);
            world.clear_cache();
            const tr::TileBuffer big = world.elevation(0, 0, 24, 24);
            const Diff d = compareInside(small, big);
            std::printf("  elevation inside a larger read: %lld of %zu values differ (max %.3g m)\n",
                        d.bits, small.data.size(), d.max_abs);
            CHECK(d.max_abs < 0.25);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL exception: %s\n", e.what());
        ++g_failures;
    }

    // The latent network directly: a sample's output must not depend on the
    // OTHER samples in its batch (that would be a cross-sample read — a race
    // or an indexing bug), but may depend on the batch SIZE, which picks the
    // GEMM kernels and so the reduction order.
    try {
        tr::MPUNet net(tr::MPUNetConfig::from_config_json((dir / "config.json").string(), "base"));
        {
            auto f = brotensor::safetensors::File::open((dir / "base.safetensors").string());
            net.load_weights(f);
        }
        const int S = 64;
        const std::size_t n = static_cast<std::size_t>(net.config().in_channels) * S * S;
        int cdim = 0;
        for (const auto& ci : net.config().conditional_inputs) cdim = ci.dim;

        std::mt19937 rng(7);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        auto batch = [&](int B, unsigned salt) {
            std::vector<float> x(n * B), c(static_cast<std::size_t>(cdim) * B);
            std::mt19937 r2(salt);
            for (auto& v : x) v = nd(r2);
            for (auto& v : c) v = nd(r2) * 0.5f;
            return std::make_pair(x, c);
        };
        auto run = [&](std::vector<float> x, std::vector<float> c, int B) {
            brotensor::Tensor xt = brodiffusion::detail::upload_host(x.data(), B, static_cast<int>(n));
            std::vector<float> labels(static_cast<std::size_t>(B), 0.8f);
            brotensor::Tensor yt;
            net.forward(xt, B, S, labels.data(), {c}, yt);
            brotensor::sync_all();
            std::vector<float> y = yt.dtype == brotensor::Dtype::FP16
                ? [&] {
                      std::vector<std::uint16_t> bits = yt.to_host_vector_fp16();
                      std::vector<float> o(bits.size());
                      for (std::size_t i = 0; i < bits.size(); ++i) o[i] = brotensor::fp16_bits_to_fp32(bits[i]);
                      return o;
                  }()
                : yt.to_host_vector();
            y.resize(y.size() / static_cast<std::size_t>(B));   // sample 0 only
            return y;
        };
        auto [x16, c16] = batch(16, 1);
        auto [o16, oc16] = batch(16, 2);
        // Same sample 0, different samples 1..15.
        std::copy(x16.begin(), x16.begin() + static_cast<std::ptrdiff_t>(n), o16.begin());
        std::copy(c16.begin(), c16.begin() + cdim, oc16.begin());

        auto cmp = [](const std::vector<float>& a, const std::vector<float>& b) {
            Diff d;
            for (std::size_t i = 0; i < a.size(); ++i)
                if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) {
                    ++d.bits;
                    d.max_abs = std::fmax(d.max_abs, std::fabs(double(a[i]) - double(b[i])));
                }
            return d;
        };
        const std::vector<float> y16 = run(x16, c16, 16);
        const Diff other = cmp(y16, run(o16, oc16, 16));
        std::printf("  base net, sample 0 with different batch-mates (B=16): %lld differ\n", other.bits);
        CHECK(other.bits == 0);
        const Diff again = cmp(y16, run(x16, c16, 16));
        std::printf("  base net, same batch twice: %lld differ\n", again.bits);
        CHECK(again.bits == 0);
        for (int B : {1, 4, 5, 9}) {
            std::vector<float> xb(x16.begin(), x16.begin() + static_cast<std::ptrdiff_t>(n * B));
            std::vector<float> cb(c16.begin(), c16.begin() + static_cast<std::ptrdiff_t>(cdim * B));
            const Diff d = cmp(y16, run(xb, cb, B));
            std::printf("  base net, sample 0 at B=%d vs B=16: %lld of %zu differ (max %.3g)\n",
                        B, d.bits, y16.size(), d.max_abs);
            CHECK(d.max_abs < 0.05);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL exception (base net): %s\n", e.what());
        ++g_failures;
    }

    if (g_failures == 0) std::printf("terrain_determinism: OK\n");
    else std::fprintf(stderr, "terrain_determinism: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
