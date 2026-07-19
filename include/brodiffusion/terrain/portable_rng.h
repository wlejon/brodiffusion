// brodiffusion/terrain/portable_rng.h — deterministic tile-seeded Gaussian noise.
//
// This is what makes terrain-diffusion infinite rather than merely large. The
// noise value at world position p depends only on (seed, p / tile), so:
//
//   * any window can be sampled in O(window), independent of world position;
//   * two overlapping windows agree BIT-FOR-BIT on their intersection, which is
//     what lets InfiniteDiffusion blend tiles without seams;
//   * a region generates identically regardless of visit order, process, or
//     machine — caching becomes a pure optimisation rather than a correctness
//     requirement.
//
// Bit-exactness with the upstream Python is deliberate here, and cheap: this is
// all integer arithmetic, so matching it costs nothing and buys seed
// compatibility with upstream-generated worlds. (We do NOT chase bit-exactness
// in the float network path, where it would be expensive and pointless.)
// Upstream reference: terrain_diffusion/inference/portable_rng.py, which was
// written to be reimplemented in C++/Java, and world_pipeline._tile_seed.
//
// Header-only: the generators are tiny and want to inline into tile loops.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace brodiffusion::terrain {

// ─── PCG64, 64-bit LCG + XSH-RR 64/32 output ───────────────────────────────
//
// A single 64-bit seed drives the whole stream, so the same seed reproduces
// the same sequence everywhere. See https://www.pcg-random.org/.
class Pcg64 {
public:
    explicit Pcg64(std::uint64_t seed) noexcept : state_(seed) {}

    std::uint32_t next_u32() noexcept {
        state_ = state_ * kMult + kInc;              // wraps mod 2^64 by definition
        const std::uint64_t s = state_;
        const std::uint32_t x = static_cast<std::uint32_t>(((s >> 18) ^ s) >> 27);
        const unsigned rot = static_cast<unsigned>(s >> 59);
        return (x >> rot) | (x << ((32u - rot) & 31u));
    }

private:
    static constexpr std::uint64_t kMult = 6364136223846793005ULL;
    static constexpr std::uint64_t kInc  = 1442695040888963407ULL;
    std::uint64_t state_;
};

// ─── Standard normals via the Marsaglia polar method ───────────────────────
//
// Draws pairs of uniforms, rejects those outside the unit disc, and emits two
// normals per accepted pair. The rejection branch still consumes its two draws
// — the stream advances on reject, and skipping that would desync from upstream.
inline void fill_standard_normal(std::uint64_t seed, float* out, std::size_t n) noexcept {
    if (n == 0) return;
    Pcg64 rng(seed);
    constexpr double kInv2p32 = 1.0 / 4294967296.0;  // 2^-32
    std::size_t i = 0;
    while (i < n) {
        const std::uint32_t u1 = rng.next_u32();
        const std::uint32_t u2 = rng.next_u32();
        // (u + 1) * 2^-32 lands in (0, 1], so v lands in (-1, 1].
        const double v1 = 2.0 * (static_cast<double>(u1) + 1.0) * kInv2p32 - 1.0;
        const double v2 = 2.0 * (static_cast<double>(u2) + 1.0) * kInv2p32 - 1.0;
        const double s = v1 * v1 + v2 * v2;
        if (s > 0.0 && s < 1.0) {
            const double f = std::sqrt(-2.0 * std::log(s) / s);
            out[i++] = static_cast<float>(v1 * f);
            if (i < n) out[i++] = static_cast<float>(v2 * f);
        }
    }
}

// ─── Tile seeding ──────────────────────────────────────────────────────────
//
// Mixes (base_seed, tile_y, tile_x) into a 64-bit stream seed. The upstream
// Python computes the products at arbitrary precision and masks afterwards;
// uint64_t wrapping gives the identical result because (a mod 2^64 + b) mod
// 2^64 == (a + b) mod 2^64. Negative tile indices truncate to their low 32
// bits exactly as Python's `& 0xFFFFFFFF` does on a negative int.
inline std::uint64_t tile_seed(std::uint64_t base_seed,
                               std::int64_t ty, std::int64_t tx) noexcept {
    constexpr std::uint64_t kPhi = 0x9E3779B9ULL;
    std::uint64_t h = base_seed * kPhi;
    h += static_cast<std::uint64_t>(static_cast<std::uint32_t>(ty));
    h = h * kPhi + static_cast<std::uint64_t>(static_cast<std::uint32_t>(tx));
    return h;
}

// Floor division. Python's `//` floors; C++ `/` truncates toward zero, so they
// disagree for negative operands — and world coordinates ARE negative half the
// time. Getting this wrong shifts the tile lattice across the origin and puts a
// visible seam through the middle of the world.
inline std::int64_t floor_div(std::int64_t a, std::int64_t b) noexcept {
    std::int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
    return q;
}

// ─── Windowed sampling of the infinite field ───────────────────────────────
//
// Writes a (channels, h, w) patch, C-order, whose top-left corner is world
// (y0, x0). Regenerates every tile the window touches and copies the
// intersection, so overlapping windows agree exactly. Cost is O(tiles touched),
// with no dependence on how far from the origin the window sits.
//
// `scratch` must hold channels*tile_h*tile_w floats; pass one buffer across
// calls to keep this allocation-free in a tile loop.
inline void gaussian_noise_patch(std::uint64_t base_seed,
                                 std::int64_t y0, std::int64_t x0,
                                 int h, int w, int channels,
                                 int tile_h, int tile_w,
                                 float* out, float* scratch) noexcept {
    const std::int64_t ty0 = floor_div(y0, tile_h);
    const std::int64_t ty1 = floor_div(y0 + h - 1, tile_h);
    const std::int64_t tx0 = floor_div(x0, tile_w);
    const std::int64_t tx1 = floor_div(x0 + w - 1, tile_w);
    const std::size_t tile_plane = static_cast<std::size_t>(tile_h) * tile_w;
    const std::size_t out_plane  = static_cast<std::size_t>(h) * w;

    for (std::int64_t ty = ty0; ty <= ty1; ++ty) {
        const std::int64_t tile_y0 = ty * tile_h;
        for (std::int64_t tx = tx0; tx <= tx1; ++tx) {
            const std::int64_t tile_x0 = tx * tile_w;

            const std::int64_t oy0 = (y0 > tile_y0) ? y0 : tile_y0;
            const std::int64_t oy1 = (y0 + h < tile_y0 + tile_h) ? (y0 + h) : (tile_y0 + tile_h);
            const std::int64_t ox0 = (x0 > tile_x0) ? x0 : tile_x0;
            const std::int64_t ox1 = (x0 + w < tile_x0 + tile_w) ? (x0 + w) : (tile_x0 + tile_w);
            if (oy0 >= oy1 || ox0 >= ox1) continue;

            fill_standard_normal(tile_seed(base_seed, ty, tx), scratch,
                                 static_cast<std::size_t>(channels) * tile_plane);

            for (int c = 0; c < channels; ++c) {
                const float* src = scratch + static_cast<std::size_t>(c) * tile_plane;
                float*       dst = out     + static_cast<std::size_t>(c) * out_plane;
                for (std::int64_t y = oy0; y < oy1; ++y) {
                    const float* srow = src + static_cast<std::size_t>(y - tile_y0) * tile_w
                                            + (ox0 - tile_x0);
                    float*       drow = dst + static_cast<std::size_t>(y - y0) * w + (ox0 - x0);
                    for (std::int64_t x = 0; x < ox1 - ox0; ++x) drow[x] = srow[x];
                }
            }
        }
    }
}

// ─── Tile blend window ─────────────────────────────────────────────────────
//
// Separable linear tent, peak 1 at the centre, falling to eps at the edge.
// Strictly positive everywhere so the weight-channel division never hits zero.
// Writes size*size floats, row-major.
inline void linear_weight_window(int size, float* out) noexcept {
    const double mid = (size - 1) / 2.0;
    constexpr double kEps = 1e-3;
    for (int y = 0; y < size; ++y) {
        double ry = std::abs(y - mid) / mid;
        if (ry > 1.0) ry = 1.0;
        const double wy = 1.0 - (1.0 - kEps) * ry;
        for (int x = 0; x < size; ++x) {
            double rx = std::abs(x - mid) / mid;
            if (rx > 1.0) rx = 1.0;
            const double wx = 1.0 - (1.0 - kEps) * rx;
            out[static_cast<std::size_t>(y) * size + x] = static_cast<float>(wy * wx);
        }
    }
}

}  // namespace brodiffusion::terrain
