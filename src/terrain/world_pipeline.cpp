#include "brodiffusion/terrain/world_pipeline.h"

#include "brodiffusion/detail/json.h"
#include "brodiffusion/terrain/portable_rng.h"
#include "brotensor/safetensors.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace brodiffusion::terrain {
namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("world_pipeline: " + msg);
}

// Upstream's coarse tiling. TILE_SIZE is the UNet's spatial extent for this
// stage (not its config `image_size`, which is 16 and only names the modules);
// TILE_STRIDE is 16 less, so consecutive tiles share a quarter of their width
// and the weight taper has something to blend across.
constexpr int kCoarseTileSize   = 64;
constexpr int kCoarseTileStride = kCoarseTileSize - 16;
constexpr int kCoarseSteps      = 20;
constexpr int kCoarseChannels   = 7;   // 6 model outputs + the weight channel

// Read a fixed-length array of doubles. Absent, null, or the wrong length is an
// error rather than a default: every one of these changes the generated world.
void read_doubles(const detail::json::Value& obj, const std::string& key,
                  double* out, std::size_t n) {
    const detail::json::Value* v = obj.find(key);
    if (v == nullptr || v->is_null()) fail("config.json pipeline has no '" + key + "'");
    const auto& a = v->as_array();
    if (a.size() != n) {
        fail("config.json pipeline." + key + " has " + std::to_string(a.size()) +
             " entries, expected " + std::to_string(n));
    }
    for (std::size_t i = 0; i < n; ++i) out[i] = a[i].as_number();
}

double read_double(const detail::json::Value& obj, const std::string& key) {
    const detail::json::Value* v = obj.find(key);
    if (v == nullptr || v->is_null()) fail("config.json pipeline has no '" + key + "'");
    return v->as_number();
}

}  // namespace

// ─── config ────────────────────────────────────────────────────────────────

WorldPipelineConfig WorldPipelineConfig::from_config_json(const std::string& config_path) {
    std::ifstream in(config_path, std::ios::binary);
    if (!in) fail("cannot open " + config_path);
    std::ostringstream ss;
    ss << in.rdbuf();
    const detail::json::Value root = detail::json::parse(ss.str());

    const detail::json::Value* p = root.find("pipeline");
    if (p == nullptr) fail("config.json has no 'pipeline' object");

    WorldPipelineConfig c;
    read_doubles(*p, "coarse_means",   c.coarse_means,   6);
    read_doubles(*p, "coarse_stds",    c.coarse_stds,    6);
    read_doubles(*p, "cond_snr",       c.cond_snr,       kSyntheticChannels);
    read_doubles(*p, "frequency_mult", c.frequency_mult, kSyntheticChannels);
    c.drop_water_pct     = read_double(*p, "drop_water_pct");
    c.coarse_pooling     = p->get_int("coarse_pooling", 1);
    c.latent_compression = p->get_int("latent_compression", 8);
    c.native_resolution  = read_double(*p, "native_resolution");
    c.residual_mean      = p->get_float("residual_mean", 0.0f);
    c.residual_std       = read_double(*p, "residual_std");

    for (int i = 0; i < 6; ++i) {
        if (c.coarse_stds[i] == 0.0) fail("config.json pipeline.coarse_stds[" +
                                          std::to_string(i) + "] is zero");
    }
    return c;
}

// ─── weight window ─────────────────────────────────────────────────────────

std::vector<float> linear_weight_window(int size) {
    if (size < 2) fail("linear_weight_window needs size >= 2");
    const double mid = (size - 1) / 2.0;
    const double eps = 1e-3;

    std::vector<double> axis(static_cast<std::size_t>(size));
    for (int i = 0; i < size; ++i) {
        const double d = std::min(1.0, std::abs(i - mid) / mid);
        axis[static_cast<std::size_t>(i)] = 1.0 - (1.0 - eps) * d;
    }

    std::vector<float> w(static_cast<std::size_t>(size) * size);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            w[static_cast<std::size_t>(y) * size + x] =
                static_cast<float>(axis[static_cast<std::size_t>(y)] *
                                   axis[static_cast<std::size_t>(x)]);
        }
    }
    return w;
}

// ─── pipeline ──────────────────────────────────────────────────────────────

WorldPipeline::WorldPipeline(const std::string& weights_dir, std::uint64_t seed)
    : seed_(seed) {
    const std::string config_path = weights_dir + "/config.json";
    cfg_ = WorldPipelineConfig::from_config_json(config_path);

    // Pooling downsamples each coarse tile before it is stored. The shipped
    // checkpoints all use 1 (no pooling); rather than carry an untested
    // avg/max/min path, refuse a value that would silently take it.
    if (cfg_.coarse_pooling != 1) {
        fail("coarse_pooling " + std::to_string(cfg_.coarse_pooling) +
             " is not supported (only 1 is ported)");
    }

    coarse_net_ = std::make_unique<MPUNet>(
        MPUNetConfig::from_config_json(config_path, "coarse"));
    {
        auto f = brotensor::safetensors::File::open(weights_dir + "/coarse.safetensors");
        coarse_net_->load_weights(f);
    }

    SyntheticMapConfig smc;
    for (int c = 0; c < kSyntheticChannels; ++c) {
        smc.frequency_mult[c] = cfg_.frequency_mult[c];
    }
    // The synthetic map's Perlin seeding is 32-bit; the world seed is 64. Upstream
    // hands the same value to both, and Python's FastNoiseLite binding truncates
    // it the same way, so take the low bits rather than mixing — mixing would
    // give a different (and equally valid, but incompatible) planet per seed.
    smc.seed = static_cast<std::uint32_t>(seed_);
    synthetic_ = std::make_unique<SyntheticMap>(
        SyntheticMapStats::load(weights_dir + "/synthetic_map_stats.json"), smc);

    weight_window_ = linear_weight_window(kCoarseTileSize);

    // t_cond = atan(snr) per channel; the network is told log(tan(t)/8), which is
    // log(snr/8) — one scalar conditioner per synthetic channel.
    t_cond_.resize(kSyntheticChannels);
    cond_inputs_.resize(kSyntheticChannels);
    for (int c = 0; c < kSyntheticChannels; ++c) {
        t_cond_[static_cast<std::size_t>(c)] = std::atan(cfg_.cond_snr[c]);
        const double v = std::log(std::tan(t_cond_[static_cast<std::size_t>(c)]) / 8.0);
        cond_inputs_[static_cast<std::size_t>(c)] = {static_cast<float>(v)};
    }

    store_  = std::make_unique<MemoryTileStore>();
    coarse_ = std::make_unique<InfiniteTensor>(
        std::vector<std::int64_t>{kCoarseChannels, -1, -1},
        [this](const std::vector<std::vector<std::int64_t>>& windows,
               const std::vector<std::vector<TileBuffer>>&) {
            std::vector<TileBuffer> out;
            out.reserve(windows.size());
            for (const auto& w : windows) out.push_back(coarse_tile_(w[1], w[2]));
            return out;
        },
        TensorWindow({kCoarseChannels, kCoarseTileSize, kCoarseTileSize},
                     {kCoarseChannels, kCoarseTileStride, kCoarseTileStride}),
        // The coarse stage is a leaf: its only inputs are the synthetic map and
        // the noise field, both of which are pure functions of world position.
        std::vector<InfiniteTensor*>{}, std::vector<TensorWindow>{},
        /*batch_size=*/1, store_.get(), "base_coarse_map");
}

WorldPipeline::~WorldPipeline() = default;

TileBuffer WorldPipeline::coarse_tile_(std::int64_t wi, std::int64_t wj) {
    constexpr int S     = kCoarseTileSize;
    const std::size_t plane = static_cast<std::size_t>(S) * S;

    const std::int64_t i1 = wi * kCoarseTileStride;
    const std::int64_t j1 = wj * kCoarseTileStride;

    // Conditioning image: the synthetic climate map, normalised by the coarse
    // stats. The five synthetic channels line up with coarse_means indices
    // {0, 2, 3, 4, 5} — index 1 is the derived channel the model produces, not
    // one it is given.
    static constexpr int kStatIndex[kSyntheticChannels] = {0, 2, 3, 4, 5};

    // COLUMN bounds first. SyntheticMap::sample mirrors upstream's factory, whose
    // leading pair drives the noise X axis — and WorldPipeline calls it as
    // `synthetic_map_factory(cj0, ci0, cj1, ci1)`, columns before rows. Passing
    // (i1, j1) here instead transposes the conditioning map, which on the square
    // tiles this stage uses is shape-preserving and therefore silent: it yields
    // terrain that looks entirely plausible and is simply the wrong world.
    std::vector<float> cond_img(static_cast<std::size_t>(kSyntheticChannels) * plane);
    synthetic_->sample(j1, i1, j1 + S, i1 + S, cond_img.data());

    // Then blend toward noise along the TrigFlow arc, per channel:
    //   cond = cos(t)*map + sin(t)*noise
    // The network is separately told each channel's t, so it knows how much of
    // what it is looking at is real.
    std::vector<float> cond_noise(static_cast<std::size_t>(kSyntheticChannels) * plane);
    std::vector<float> scratch(static_cast<std::size_t>(kSyntheticChannels) * plane);
    gaussian_noise_patch(seed_, i1, j1, S, S, kSyntheticChannels, S, S,
                         cond_noise.data(), scratch.data());

    for (int c = 0; c < kSyntheticChannels; ++c) {
        const float mean = static_cast<float>(cfg_.coarse_means[kStatIndex[c]]);
        const float sd   = static_cast<float>(cfg_.coarse_stds[kStatIndex[c]]);
        const float ct   = static_cast<float>(std::cos(t_cond_[static_cast<std::size_t>(c)]));
        const float st   = static_cast<float>(std::sin(t_cond_[static_cast<std::size_t>(c)]));
        float* ch = cond_img.data() + static_cast<std::size_t>(c) * plane;
        const float* nz = cond_noise.data() + static_cast<std::size_t>(c) * plane;
        for (std::size_t k = 0; k < plane; ++k) {
            ch[k] = ct * ((ch[k] - mean) / sd) + st * nz[k];
        }
    }

    // The sample noise uses seed + 1 — a distinct field from the conditioning
    // noise, at the same world position.
    const int C_out = coarse_net_->config().out_channels;
    std::vector<float> sample_noise(static_cast<std::size_t>(C_out) * plane);
    scratch.resize(static_cast<std::size_t>(C_out) * plane);
    gaussian_noise_patch(seed_ + 1, i1, j1, S, S, C_out, S, S,
                         sample_noise.data(), scratch.data());

    std::vector<float> sample;
    sample_coarse(*coarse_net_, sample_noise.data(), cond_img.data(),
                  cond_inputs_, S, kCoarseSteps, sample);

    // Denormalise, then rebuild channel 1. The model predicts channel 1 as an
    // offset from channel 0, so the stored value is the difference — this is
    // upstream's `sample[0, 1] = sample[0, 0] - sample[0, 1]`, not a swap.
    for (int c = 0; c < C_out; ++c) {
        const float mean = static_cast<float>(cfg_.coarse_means[c]);
        const float sd   = static_cast<float>(cfg_.coarse_stds[c]);
        float* ch = sample.data() + static_cast<std::size_t>(c) * plane;
        for (std::size_t k = 0; k < plane; ++k) ch[k] = ch[k] * sd + mean;
    }
    {
        float* c0 = sample.data();
        float* c1 = sample.data() + plane;
        for (std::size_t k = 0; k < plane; ++k) c1[k] = c0[k] - c1[k];
    }

    // Emit [value * w, w] so the evaluator's additive blend produces a weighted
    // mean once the consumer divides by the last channel.
    TileBuffer out;
    out.shape = {kCoarseChannels, S, S};
    out.data.resize(static_cast<std::size_t>(kCoarseChannels) * plane);
    for (int c = 0; c < C_out; ++c) {
        const float* src = sample.data() + static_cast<std::size_t>(c) * plane;
        float* dst = out.data.data() + static_cast<std::size_t>(c) * plane;
        for (std::size_t k = 0; k < plane; ++k) dst[k] = src[k] * weight_window_[k];
    }
    std::copy(weight_window_.begin(), weight_window_.end(),
              out.data.begin() + static_cast<std::ptrdiff_t>(
                  static_cast<std::size_t>(C_out) * plane));
    return out;
}

TileBuffer WorldPipeline::coarse(std::int64_t i1, std::int64_t j1,
                                 std::int64_t i2, std::int64_t j2) {
    return (*coarse_)({Slice{0, kCoarseChannels}, Slice{i1, i2}, Slice{j1, j2}});
}

TileBuffer WorldPipeline::coarse_normalized(std::int64_t i1, std::int64_t j1,
                                            std::int64_t i2, std::int64_t j2) {
    TileBuffer raw = coarse(i1, j1, i2, j2);
    const std::size_t plane =
        static_cast<std::size_t>(raw.shape[1]) * static_cast<std::size_t>(raw.shape[2]);

    TileBuffer out;
    out.shape = {kCoarseChannels - 1, raw.shape[1], raw.shape[2]};
    out.data.resize(static_cast<std::size_t>(kCoarseChannels - 1) * plane);

    const float* w = raw.data.data() + static_cast<std::size_t>(kCoarseChannels - 1) * plane;
    for (int c = 0; c < kCoarseChannels - 1; ++c) {
        const float* src = raw.data.data() + static_cast<std::size_t>(c) * plane;
        float* dst = out.data.data() + static_cast<std::size_t>(c) * plane;
        for (std::size_t k = 0; k < plane; ++k) dst[k] = src[k] / w[k];
    }
    return out;
}

void WorldPipeline::clear_cache() { coarse_->clear_cache(); }

}  // namespace brodiffusion::terrain
