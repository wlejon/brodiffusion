#include "brodiffusion/terrain/synthetic_map.h"

#include "brodiffusion/detail/json.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

// FastNoiseLite is a single header that defines its class inline; keep it out of
// our public headers so downstream translation units do not inherit 100 KB of it.
#include "FastNoiseLite.h"

namespace brodiffusion::terrain {
namespace {

namespace json = brodiffusion::detail::json;

// Per-channel FBm shape. Only the temperature field uses 2 octaves — it wants
// broad continental gradients, not ridged detail. Everything else uses 4.
struct ChannelFbm {
    int    octaves;
    double lacunarity;
    double gain;
};
constexpr ChannelFbm kFbm[kSyntheticChannels] = {
    {4, 2.0, 0.5},  // elevation
    {2, 2.0, 0.5},  // temperature
    {4, 2.0, 0.5},  // temperature seasonality
    {4, 2.0, 0.5},  // precipitation
    {4, 2.0, 0.5},  // precipitation seasonality
};
constexpr double kBaseFrequency = 0.05;

std::vector<double> read_double_array(const json::Value& v) {
    std::vector<double> out;
    for (const auto& e : v.as_array()) out.push_back(e.as_number());
    return out;
}

// np.interp with left/right clamped to the endpoints, which is what
// perlin_transform.transform_perlin asks for. `xp` is strictly increasing, so a
// binary search is exact; np.interp itself searches rather than assuming a grid.
double interp_clamped(double x,
                      const std::vector<double>& xp,
                      const std::vector<double>& fp) {
    if (x <= xp.front()) return fp.front();
    if (x >= xp.back())  return fp.back();
    const std::size_t i =
        static_cast<std::size_t>(std::upper_bound(xp.begin(), xp.end(), x) - xp.begin());
    const double x0 = xp[i - 1], x1 = xp[i];
    const double y0 = fp[i - 1], y1 = fp[i];
    const double dx = x1 - x0;
    // build_quantiles guarantees strict increase, so dx > 0; guard anyway rather
    // than emit a NaN that would propagate silently through the whole map.
    if (dx <= 0.0) return y0;
    return y0 + (y1 - y0) * ((x - x0) / dx);
}

}  // namespace

// ─── stats loading ─────────────────────────────────────────────────────────

SyntheticMapStats SyntheticMapStats::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("synthetic_map_stats: cannot open " + path);
    std::ostringstream ss;
    ss << in.rdbuf();

    const json::Value root = json::parse(ss.str());

    SyntheticMapStats s;
    s.a_temp_std   = root.at("a_temp_std").as_number();
    s.b_temp_std   = root.at("b_temp_std").as_number();
    s.temp_std_p1  = root.at("temp_std_p1").as_number();
    s.temp_std_p99 = root.at("temp_std_p99").as_number();

    for (const auto& t : root.at("noise_quantile_tables").as_array())
        s.noise_quantiles.push_back(read_double_array(t));
    for (const auto& t : root.at("data_quantile_tables").as_array())
        s.data_quantiles.push_back(read_double_array(t));

    if (s.noise_quantiles.size() != kSyntheticChannels ||
        s.data_quantiles.size()  != kSyntheticChannels) {
        throw std::runtime_error(
            "synthetic_map_stats: expected " + std::to_string(int(kSyntheticChannels)) +
            " quantile tables, got " + std::to_string(s.noise_quantiles.size()) + "/" +
            std::to_string(s.data_quantiles.size()));
    }
    for (std::size_t c = 0; c < kSyntheticChannels; ++c) {
        if (s.noise_quantiles[c].empty() ||
            s.noise_quantiles[c].size() != s.data_quantiles[c].size()) {
            throw std::runtime_error("synthetic_map_stats: channel " + std::to_string(c) +
                                     " has mismatched or empty quantile tables");
        }
    }
    return s;
}

// ─── noise generators ──────────────────────────────────────────────────────

struct SyntheticMap::Impl {
    FastNoiseLite noise[kSyntheticChannels];
};

SyntheticMap::SyntheticMap(SyntheticMapStats stats, const SyntheticMapConfig& config)
    : impl_(std::make_unique<Impl>()), stats_(std::move(stats)) {
    for (int c = 0; c < kSyntheticChannels; ++c) {
        FastNoiseLite& n = impl_->noise[c];
        // Upstream: seeds[c] = (seed + c + 1) & 0x7FFFFFFF. (It also treats a seed
        // of 0 as "pick a random one"; we do not — a C++ terrain generator that
        // silently randomised on seed 0 would be a trap, so 0 is a real seed here.)
        n.SetSeed(static_cast<int>((config.seed + static_cast<std::uint32_t>(c) + 1u)
                                   & 0x7FFFFFFFu));
        n.SetNoiseType(FastNoiseLite::NoiseType_Perlin);
        n.SetFrequency(static_cast<float>(kBaseFrequency * config.frequency_mult[c]));
        n.SetFractalType(FastNoiseLite::FractalType_FBm);
        n.SetFractalOctaves(kFbm[c].octaves);
        n.SetFractalLacunarity(static_cast<float>(kFbm[c].lacunarity));
        n.SetFractalGain(static_cast<float>(kFbm[c].gain));
    }
}

SyntheticMap::~SyntheticMap() = default;
SyntheticMap::SyntheticMap(SyntheticMap&&) noexcept = default;
SyntheticMap& SyntheticMap::operator=(SyntheticMap&&) noexcept = default;

// ─── sampling ──────────────────────────────────────────────────────────────

void SyntheticMap::sample_raw(std::int64_t i1, std::int64_t j1,
                              std::int64_t i2, std::int64_t j2, float* out) const {
    const std::int64_t nx = i2 - i1;   // length of the `i` axis
    const std::int64_t ny = j2 - j1;   // length of the `j` axis
    if (nx <= 0 || ny <= 0) return;
    const std::size_t plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);

    // Upstream builds coords with np.meshgrid(arange(i1,i2), arange(j1,j2)) and
    // flattens, so the generated order is (j outer, i inner) — then reshapes the
    // flat result to (nx, ny). For a square window that makes `i` the fast axis of
    // the output; for a non-square one the reshape genuinely reinterprets the
    // buffer. Either way the flat fill order is the same, so we reproduce that
    // order and leave the caller's view of it alone rather than "fixing" it.
    for (int c = 0; c < kSyntheticChannels; ++c) {
        // Non-const: FastNoiseLite::GetNoise is not marked const upstream, even
        // though it only reads member state (it mutates its by-value coordinate
        // args). unique_ptr does not propagate const to its pointee, so we can
        // bind a mutable reference from this const method — and because the call
        // really is read-only, one SyntheticMap is safe to share across threads
        // generating different tiles concurrently.
        FastNoiseLite&             n  = impl_->noise[c];
        const std::vector<double>& xq = stats_.noise_quantiles[c];
        const std::vector<double>& yq = stats_.data_quantiles[c];
        float* dst = out + static_cast<std::size_t>(c) * plane;

        std::size_t k = 0;
        for (std::int64_t r = 0; r < ny; ++r) {
            const float y = static_cast<float>(j1 + r);
            for (std::int64_t col = 0; col < nx; ++col) {
                const float x = static_cast<float>(i1 + col);
                // GetNoise returns float; upstream stores it into a float32 array
                // before interpolating, so widening here matches bit-for-bit.
                const double v = static_cast<double>(n.GetNoise(x, y));
                dst[k++] = static_cast<float>(interp_clamped(v, xq, yq));
            }
        }
    }
}

void SyntheticMap::finalize(float* map, std::size_t count) const {
    float* elev       = map + 0 * count;
    float* temp       = map + 1 * count;
    float* temp_std   = map + 2 * count;
    float* precip     = map + 3 * count;
    float* precip_std = map + 4 * count;

    const double a  = stats_.a_temp_std;
    const double b  = stats_.b_temp_std;
    const double p1 = stats_.temp_std_p1;
    const double p99 = stats_.temp_std_p99;
    const double span = p99 - p1;

    for (std::size_t i = 0; i < count; ++i) {
        const double e = elev[i];
        const double p = precip[i];

        // Environmental lapse rate: temperature falls with altitude, faster in dry
        // air than in moist. Clamped to the physical envelope (dry adiabatic
        // -9.8 °C/km to a damp -4.0) and applied only above sea level.
        double lapse = (-6.5 + 0.0015 * p);
        lapse = std::clamp(lapse, -9.8, -4.0) / 1000.0;
        double t = temp[i] + lapse * std::max(0.0, e);
        t = std::clamp(t, -10.0, 40.0);
        // Empirical stretch below 20 °C: expands the cold half of the range around
        // a 20 °C pivot by 1.25x, which separates desert from tropical mass in the
        // conditioning signal. Above 20 °C is untouched.
        if (t <= 20.0) t = (t - 20.0) * 1.25 + 20.0;

        // Seasonality is stored detrended against temperature; undo that here.
        // The baseline keeps the reconstructed value from going below what the
        // temperature/seasonality regression allows for this pixel.
        const double frac     = (temp_std[i] - p1) / span;
        const double baseline = std::max(p1, -(a * t + b));
        double ts = frac * (p99 - baseline) + baseline;
        ts = ts + (a * t + b);
        ts = std::max(ts, 20.0);

        // Wet places are less seasonal in their rainfall; taper toward zero as
        // annual precipitation climbs (reaching zero near 4500 mm).
        const double ps = precip_std[i] * std::max(0.0, (185.0 - 0.04111 * p) / 185.0);

        temp[i]       = static_cast<float>(t);
        temp_std[i]   = static_cast<float>(ts);
        precip_std[i] = static_cast<float>(ps);
    }
}

void SyntheticMap::sample(std::int64_t i1, std::int64_t j1,
                          std::int64_t i2, std::int64_t j2, float* out) const {
    const std::int64_t nx = i2 - i1;
    const std::int64_t ny = j2 - j1;
    if (nx <= 0 || ny <= 0) return;
    const std::size_t plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);

    sample_raw(i1, j1, i2, j2, out);
    finalize(out, plane);

    // The coarse model was trained on signed-sqrt elevation, which compresses the
    // 8 km of ocean trench and 6 km of mountain into a range the network can use
    // without the deep ocean dominating the loss.
    for (std::size_t i = 0; i < plane; ++i) {
        const float e = out[i];
        out[i] = std::copysign(std::sqrt(std::fabs(e)), e);
    }
}

}  // namespace brodiffusion::terrain
