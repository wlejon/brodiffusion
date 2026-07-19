// brodiffusion/terrain/synthetic_map.h — the coarse stage's climate conditioning.
//
// terrain-diffusion does not hallucinate a planet from nothing. Its coarse model
// is conditioned on a five-channel synthetic map — elevation, mean temperature,
// temperature seasonality, precipitation, precipitation seasonality — and that map
// is what decides where the deserts, the ice caps and the rainforests land. The
// diffusion stages then supply the landforms.
//
// Each channel is Perlin FBm, but raw FBm is roughly Gaussian and Earth is not:
// real elevation is bimodal (abyssal plain + continental shelf), real precipitation
// is a hard-floored long tail. So each field is quantile-matched — remapped through
// its own empirical CDF onto the marginal distribution of the real Earth, measured
// once offline from ETOPO + WorldClim by scripts/build-terrain-synthetic-stats.py.
// That remap is the whole trick: it costs one interpolation per pixel and turns
// noise into something with a plausible planet's statistics.
//
// A few couplings then run on top (SyntheticMap::finalize), which is where the map
// stops being five independent fields and starts being a climate: temperature falls
// with elevation at a precipitation-dependent lapse rate, seasonality is rebaselined
// against temperature, and precipitation seasonality is damped in wet regions.
//
// Sampling is a pure function of world coordinates — no tiling, no accumulation, no
// state — so any window is O(area) at any distance from the origin, which is what
// keeps the infinite world infinite.
//
// Upstream reference: terrain_diffusion/inference/synthetic_map.py and
// perlin_transform.py. scripts/terrain_synthetic_parity.sh gates this against them.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace brodiffusion::terrain {

// Channel order, fixed by the trained coarse model's conditioning layout.
enum SyntheticChannel {
    kSyntheticElevation      = 0,
    kSyntheticTemperature    = 1,
    kSyntheticTemperatureStd = 2,
    kSyntheticPrecipitation  = 3,
    kSyntheticPrecipitationStd = 4,
    kSyntheticChannels       = 5,
};

// The offline-measured statistics, as written by build-terrain-synthetic-stats.py.
// `noise_quantiles[c]` and `data_quantiles[c]` are the paired CDF knots for channel
// c: noise value -> Earth value. Both are strictly increasing and the same length.
struct SyntheticMapStats {
    double a_temp_std   = 0.0;  // slope of temp_std regressed on temp
    double b_temp_std   = 0.0;  // intercept of the same fit
    double temp_std_p1  = 0.0;  // 0.1st percentile of detrended temp_std
    double temp_std_p99 = 0.0;  // 99.9th percentile of the same

    std::vector<std::vector<double>> noise_quantiles;  // [5][n_quantiles]
    std::vector<std::vector<double>> data_quantiles;   // [5][n_quantiles]

    // Reads synthetic_map_stats.json. Throws std::runtime_error if the file is
    // missing or malformed, or if the tables are not 5 paired equal-length rows —
    // a half-loaded table would still generate terrain, just a differently-shaped
    // planet, so this refuses rather than degrades.
    static SyntheticMapStats load(const std::string& path);
};

// Per-channel FBm configuration. Upstream's `frequency_mult` scales a 0.05 base
// frequency; octaves/lacunarity/gain are fixed per channel.
struct SyntheticMapConfig {
    // The shipped 30 m checkpoint's value. WorldPipeline's constructor signature
    // says {1.5, 3, 3, 3, 3}, but from_pretrained overrides it with the config
    // that ships beside the weights — so those signature defaults are what runs
    // only if you build the pipeline by hand. Set this from the checkpoint's
    // config.json rather than trusting the default if you load a different one;
    // the quantile tables are measured against whatever value is used here.
    double frequency_mult[kSyntheticChannels] = {1.0, 1.0, 1.0, 1.0, 1.0};
    // Upstream masks to 31 bits and offsets per channel: seed + c + 1.
    std::uint32_t seed = 0;
};

class SyntheticMap {
public:
    SyntheticMap(SyntheticMapStats stats, const SyntheticMapConfig& config);
    ~SyntheticMap();

    SyntheticMap(SyntheticMap&&) noexcept;
    SyntheticMap& operator=(SyntheticMap&&) noexcept;

    // Fills `out` with a (5, i2-i1, j2-j1) C-order block: the quantile-matched
    // fields with finalize() applied, and elevation passed through the signed
    // square root the coarse model was trained against.
    //
    // `out` must hold 5*(i2-i1)*(j2-j1) floats.
    void sample(std::int64_t i1, std::int64_t j1,
                std::int64_t i2, std::int64_t j2, float* out) const;

    // The raw quantile-matched fields, before finalize(). Exposed because the
    // parity gate checks the two halves separately — a sign error in the lapse
    // rate and a bad quantile table both show up as "the map looks wrong", and
    // splitting them says which.
    void sample_raw(std::int64_t i1, std::int64_t j1,
                    std::int64_t i2, std::int64_t j2, float* out) const;

    // Applies the climate couplings in place over `count` pixels per channel.
    void finalize(float* map, std::size_t count) const;

    const SyntheticMapStats& stats() const { return stats_; }

private:
    struct Impl;                       // hides FastNoiseLite from our public headers
    std::unique_ptr<Impl> impl_;
    SyntheticMapStats stats_;
};

}  // namespace brodiffusion::terrain
