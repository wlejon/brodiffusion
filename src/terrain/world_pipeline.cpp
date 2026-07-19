#include "brodiffusion/terrain/world_pipeline.h"

#include "brodiffusion/detail/compute.h"
#include "brodiffusion/detail/json.h"
#include "brodiffusion/terrain/laplacian.h"
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

// Pull a device tensor back to host FP32, upconverting the CUDA build's FP16.
// Same helper as sampler.cpp's; both are file-local because it is four lines and
// sharing it would mean a header for one function.
void download_f32(const brotensor::Tensor& t, std::vector<float>& out) {
    const std::size_t n = static_cast<std::size_t>(t.rows) * t.cols;
    if (t.dtype == brotensor::Dtype::FP16) {
        std::vector<std::uint16_t> bits(n);
        t.copy_to_host_fp16(bits.data());
        brotensor::sync_all();
        out.resize(n);
        for (std::size_t i = 0; i < n; ++i) out[i] = brotensor::fp16_bits_to_fp32(bits[i]);
        return;
    }
    brotensor::sync_all();
    out = t.to_host_vector();
}

// Upstream's coarse tiling. TILE_SIZE is the UNet's spatial extent for this
// stage (not its config `image_size`, which is 16 and only names the modules);
// TILE_STRIDE is 16 less, so consecutive tiles share a quarter of their width
// and the weight taper has something to blend across.
constexpr int kCoarseTileSize   = 64;
constexpr int kCoarseTileStride = kCoarseTileSize - 16;
constexpr int kCoarseSteps      = 20;
constexpr int kCoarseChannels   = 7;   // 6 model outputs + the weight channel

// Latent tiling. Note the stride is half the tile, a heavier overlap than the
// coarse stage's — every latent cell is covered by four windows.
constexpr int kLatentTileSize   = 64;
constexpr int kLatentTileStride = kLatentTileSize / 2;
constexpr int kLatentChannels   = 6;   // 5 model outputs + the weight channel
constexpr int kLatentBatch      = 16;

// The coarse patch one latent window reads: 4x4 coarse cells at unit stride,
// offset by -1 so the window is centred on the cell the tile sits over rather
// than hanging off its corner.
constexpr int kCoarsePatch      = 4;

// Normalisation for the latent stage's 7-channel conditioning patch. Baked into
// the checkpoint's training, not read from config.json — upstream carries them
// as literals in _build_latent_stage and no shipped config overrides them.
constexpr float kCondInputMean[7] = {14.99f, 11.65f, 15.87f, 619.26f, 833.12f, 69.40f, 0.66f};
constexpr float kCondInputStd[7]  = {21.72f, 21.78f, 10.40f, 452.29f, 738.09f, 34.59f, 0.47f};

// Decoder tiling. Much larger tiles than the other stages, and the only stage
// whose tile size is a pipeline setting rather than a constant — upstream
// exposes decoder_tile_size/stride as constructor kwargs. The shipped default is
// 512/384, and no shipped config overrides it.
constexpr int kDecoderTileSize   = 512;
constexpr int kDecoderTileStride = 384;
constexpr int kDecoderChannels   = 2;   // 1 model output + the weight channel

// The conditioning vector's six segments, in order: the 4x4 elevation patch, the
// 4x4 p5 patch, the four climate means, the 4x4 validity mask, the histogram,
// and the noise level. 58 values, which is the base net's conditioner width.
constexpr int kCondSegments[6] = {16, 16, 4, 16, 5, 1};
constexpr int kCondDim         = 58;

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

// mp_concat's per-segment scale. EDM2 concatenates magnitude-preserving pieces
// but a later layer sees each piece in proportion to its channel count, so each
// segment is rescaled to contribute equally:
//   C = sqrt(sum_N / sum(w^2)),  scale_i = C / sqrt(N_i) * w_i
// with equal weights w_i = 1/k. Computed here rather than hard-coded so the
// relationship to the segment sizes stays visible.
std::vector<float> mp_concat_scales(const int* sizes, int k) {
    const double w = 1.0 / k;
    double sum_n = 0.0;
    for (int i = 0; i < k; ++i) sum_n += sizes[i];
    const double C = std::sqrt(sum_n / (k * w * w));
    std::vector<float> s(static_cast<std::size_t>(k));
    for (int i = 0; i < k; ++i) {
        s[static_cast<std::size_t>(i)] =
            static_cast<float>(C / std::sqrt(static_cast<double>(sizes[i])) * w);
    }
    return s;
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

    // ── latent stage ───────────────────────────────────────────────────────
    base_net_ = std::make_unique<MPUNet>(
        MPUNetConfig::from_config_json(config_path, "base"));
    {
        auto f = brotensor::safetensors::File::open(weights_dir + "/base.safetensors");
        base_net_->load_weights(f);
    }
    latent_weight_window_ = linear_weight_window(kLatentTileSize);

    const TensorWindow latent_out({kLatentChannels, kLatentTileSize, kLatentTileSize},
                                  {kLatentChannels, kLatentTileStride, kLatentTileStride});
    const TensorWindow coarse_arg({kCoarseChannels, kCoarsePatch, kCoarsePatch},
                                  {kCoarseChannels, 1, 1},
                                  {0, -1, -1});

    const auto t_list = trigflow_t_list(/*two_step=*/true);

    // Two TrigFlow steps, two tensors. The seed offsets are upstream's literals:
    // each step draws its own noise field, and reusing one offset would make the
    // second step re-add the first's noise instead of fresh noise.
    latent_init_ = std::make_unique<InfiniteTensor>(
        std::vector<std::int64_t>{kLatentChannels, -1, -1},
        [this, t_list](const std::vector<std::vector<std::int64_t>>& windows,
                       const std::vector<std::vector<TileBuffer>>& args) {
            return latent_step_(windows, args.at(0), /*prev=*/nullptr,
                                t_list[0], 5819);
        },
        latent_out,
        std::vector<InfiniteTensor*>{coarse_.get()},
        std::vector<TensorWindow>{coarse_arg},
        kLatentBatch, store_.get(), "init_latent_map");

    latent_step0_ = std::make_unique<InfiniteTensor>(
        std::vector<std::int64_t>{kLatentChannels, -1, -1},
        [this, t_list](const std::vector<std::vector<std::int64_t>>& windows,
                       const std::vector<std::vector<TileBuffer>>& args) {
            // args[0] is the previous step at the SAME window, args[1] the
            // coarse patch — the order the tensor was constructed with.
            return latent_step_(windows, args.at(1), &args.at(0),
                                t_list[1], 5820);
        },
        latent_out,
        std::vector<InfiniteTensor*>{latent_init_.get(), coarse_.get()},
        std::vector<TensorWindow>{latent_out, coarse_arg},
        kLatentBatch, store_.get(), "step_latent_map_0");

    // ── decoder stage ──────────────────────────────────────────────────────
    decoder_net_ = std::make_unique<MPUNet>(
        MPUNetConfig::from_config_json(config_path, "decoder"));
    {
        auto f = brotensor::safetensors::File::open(weights_dir + "/decoder.safetensors");
        decoder_net_->load_weights(f);
    }
    decoder_weight_window_ = linear_weight_window(kDecoderTileSize);

    const int lc = cfg_.latent_compression;
    if (kDecoderTileSize % lc != 0 || kDecoderTileStride % lc != 0) {
        fail("latent_compression " + std::to_string(lc) +
             " does not divide the decoder tile size/stride");
    }

    residual_ = std::make_unique<InfiniteTensor>(
        std::vector<std::int64_t>{kDecoderChannels, -1, -1},
        [this](const std::vector<std::vector<std::int64_t>>& windows,
               const std::vector<std::vector<TileBuffer>>& args) {
            std::vector<TileBuffer> out;
            out.reserve(windows.size());
            for (std::size_t b = 0; b < windows.size(); ++b) {
                out.push_back(residual_tile_(windows[b][1], windows[b][2],
                                             args.at(0)[b]));
            }
            return out;
        },
        TensorWindow({kDecoderChannels, kDecoderTileSize, kDecoderTileSize},
                     {kDecoderChannels, kDecoderTileStride, kDecoderTileStride}),
        std::vector<InfiniteTensor*>{latent_step0_.get()},
        // The latent window is the output window divided by the compression, so
        // one decoder tile reads exactly the latents that sit under it.
        std::vector<TensorWindow>{
            TensorWindow({kLatentChannels, kDecoderTileSize / lc, kDecoderTileSize / lc},
                         {kLatentChannels, kDecoderTileStride / lc, kDecoderTileStride / lc})},
        /*batch_size=*/1, store_.get(), "init_residual_map");
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

std::vector<TileBuffer> WorldPipeline::latent_step_(
    const std::vector<std::vector<std::int64_t>>& windows,
    const std::vector<TileBuffer>&                coarse_tiles,
    const std::vector<TileBuffer>*                prev,
    float t, std::uint64_t seed_offset) {

    constexpr int S = kLatentTileSize;
    const std::size_t plane = static_cast<std::size_t>(S) * S;
    const int C = base_net_->config().out_channels;       // 5
    const std::size_t n = static_cast<std::size_t>(C) * plane;
    const std::size_t B = windows.size();

    const float cos_t = std::cos(t);
    const float sin_t = std::sin(t);

    static const std::vector<float> kScales = mp_concat_scales(kCondSegments, 6);

    std::vector<float> x_in(B * n);
    std::vector<float> x_t(B * n);
    std::vector<float> cond(B * kCondDim);
    std::vector<float> noise(n), scratch(n);

    constexpr std::size_t patch = kCoarsePatch * kCoarsePatch;   // 16

    for (std::size_t b = 0; b < B; ++b) {
        const std::int64_t wi = windows[b][1], wj = windows[b][2];

        // Prior sample. The initial step starts from zero; a later step reads
        // the previous tensor's weighted output, normalises it, and rescales by
        // sigma_data to undo the division applied when it was stored.
        std::vector<float> sample(n, 0.0f);
        if (prev != nullptr) {
            const TileBuffer& p = (*prev)[b];
            const float* w = p.data.data() + static_cast<std::size_t>(C) * plane;
            for (int c = 0; c < C; ++c) {
                const float* src = p.data.data() + static_cast<std::size_t>(c) * plane;
                float* dst = sample.data() + static_cast<std::size_t>(c) * plane;
                for (std::size_t k = 0; k < plane; ++k) dst[k] = src[k] / w[k] * kSigmaData;
            }
        }

        // Conditioning: the 4x4 coarse patch, weight-normalised, with an
        // all-ones validity mask appended as a seventh channel. The mask exists
        // because upstream can import real-world tiles that leave holes; a
        // purely generated world has none, so it is constant here — but the
        // network still expects the channel.
        const TileBuffer& ct = coarse_tiles[b];
        float cp[7 * patch];
        {
            const float* wsum = ct.data.data() + static_cast<std::size_t>(kCoarseChannels - 1) * patch;
            for (int c = 0; c < kCoarseChannels - 1; ++c) {
                const float* src = ct.data.data() + static_cast<std::size_t>(c) * patch;
                for (std::size_t k = 0; k < patch; ++k) {
                    cp[static_cast<std::size_t>(c) * patch + k] = src[k] / wsum[k];
                }
            }
            for (std::size_t k = 0; k < patch; ++k) cp[6 * patch + k] = 1.0f;
        }
        for (int c = 0; c < 7; ++c) {
            for (std::size_t k = 0; k < patch; ++k) {
                cp[static_cast<std::size_t>(c) * patch + k] =
                    (cp[static_cast<std::size_t>(c) * patch + k] - kCondInputMean[c]) / kCondInputStd[c];
            }
        }

        // Assemble the 58-vector: elevation patch, p5 patch, the four climate
        // channels averaged over the patch's centre 2x2, the mask, the
        // histogram (zeros for a generated world), and the noise level.
        float* cv = cond.data() + b * kCondDim;
        std::size_t off = 0;
        for (std::size_t k = 0; k < patch; ++k) cv[off++] = cp[k] * kScales[0];
        for (std::size_t k = 0; k < patch; ++k) cv[off++] = cp[patch + k] * kScales[1];
        for (int c = 2; c < 6; ++c) {
            float acc = 0.0f;
            for (int y = 1; y < 3; ++y) {
                for (int x = 1; x < 3; ++x) {
                    acc += cp[static_cast<std::size_t>(c) * patch +
                              static_cast<std::size_t>(y) * kCoarsePatch + x];
                }
            }
            cv[off++] = acc / 4.0f * kScales[2];
        }
        for (std::size_t k = 0; k < patch; ++k) cv[off++] = cp[6 * patch + k] * kScales[3];
        for (int k = 0; k < 5; ++k) cv[off++] = 0.0f;   // histogram_raw
        // noise_level is 0, standardised over a uniform [0,1] prior:
        // (level - 0.5) * sqrt(12).
        cv[off++] = static_cast<float>((0.0 - 0.5) * std::sqrt(12.0)) * kScales[5];

        // x_t = cos(t)*sample + sin(t)*z, with z the tile-seeded noise scaled by
        // sigma_data. The model sees x_t / sigma_data.
        gaussian_noise_patch(seed_ + seed_offset, wi * kLatentTileStride,
                             wj * kLatentTileStride, S, S, C, S, S,
                             noise.data(), scratch.data());
        for (std::size_t k = 0; k < n; ++k) {
            const float z = noise[k] * kSigmaData;
            const float v = cos_t * sample[k] + sin_t * z;
            x_t[b * n + k]  = v;
            x_in[b * n + k] = v / kSigmaData;
        }
    }

    // One batched forward for the whole group. `f` is pure, so batching is a
    // scheduling concern only — the infinite-tensor gate proved the graph is
    // batch-invariant, and any residual batch dependence would come from cuDNN
    // algorithm selection inside the network rather than from here.
    brotensor::Tensor xt = brodiffusion::detail::upload_host(
        x_in.data(), static_cast<int>(B), static_cast<int>(n));
    std::vector<float> labels(B, t);
    brotensor::Tensor yt;
    base_net_->forward(xt, static_cast<int>(B), S, labels.data(), {cond}, yt);

    std::vector<float> pred;
    download_f32(yt, pred);

    std::vector<TileBuffer> out(B);
    for (std::size_t b = 0; b < B; ++b) {
        TileBuffer& o = out[b];
        o.shape = {kLatentChannels, S, S};
        o.data.resize(static_cast<std::size_t>(kLatentChannels) * plane);
        for (int c = 0; c < C; ++c) {
            for (std::size_t k = 0; k < plane; ++k) {
                const std::size_t i = b * n + static_cast<std::size_t>(c) * plane + k;
                // The TrigFlow update negates the model output: upstream's
                // `pred = -model(...)`, so this is
                //   cos(t)*x_t - sin(t)*sigma_data*(-model)
                // folded into a plus. Then divide by sigma_data for storage.
                const float s = (cos_t * x_t[i] + sin_t * kSigmaData * pred[i]) / kSigmaData;
                o.data[static_cast<std::size_t>(c) * plane + k] =
                    s * latent_weight_window_[k];
            }
        }
        std::copy(latent_weight_window_.begin(), latent_weight_window_.end(),
                  o.data.begin() + static_cast<std::ptrdiff_t>(
                      static_cast<std::size_t>(C) * plane));
    }
    return out;
}

TileBuffer WorldPipeline::latent(std::int64_t i1, std::int64_t j1,
                                 std::int64_t i2, std::int64_t j2) {
    return (*latent_step0_)({Slice{0, kLatentChannels}, Slice{i1, i2}, Slice{j1, j2}});
}

TileBuffer WorldPipeline::residual_tile_(std::int64_t wi, std::int64_t wj,
                                         const TileBuffer& latent_tile) {
    constexpr int S = kDecoderTileSize;
    const std::size_t plane = static_cast<std::size_t>(S) * S;
    const int lc = cfg_.latent_compression;
    const int LS = S / lc;                       // latent tile side
    const std::size_t lplane = static_cast<std::size_t>(LS) * LS;

    // The decoder conditions on FOUR of the latent map's five channels. The
    // fifth is the low-frequency elevation band, which the Laplacian
    // reconstruction consumes directly rather than the network — feeding it here
    // would both mis-shape the input and hand the decoder information it was
    // never trained to see.
    constexpr int kLatentCondCh = 4;

    // Weight-normalise, then upsample by nearest neighbour. Nearest, not
    // bilinear: the latents are a learned code, and interpolating between codes
    // is not meaningful — upstream is explicit about this.
    std::vector<float> up(static_cast<std::size_t>(kLatentCondCh) * plane);
    {
        const float* w = latent_tile.data.data() +
                         static_cast<std::size_t>(kLatentChannels - 1) * lplane;
        for (int c = 0; c < kLatentCondCh; ++c) {
            const float* src = latent_tile.data.data() + static_cast<std::size_t>(c) * lplane;
            float* dst = up.data() + static_cast<std::size_t>(c) * plane;
            for (int y = 0; y < S; ++y) {
                const int ly = y / lc;
                for (int x = 0; x < S; ++x) {
                    const std::size_t li = static_cast<std::size_t>(ly) * LS + (x / lc);
                    dst[static_cast<std::size_t>(y) * S + x] = src[li] / w[li];
                }
            }
        }
    }

    // A single TrigFlow step from a zero sample, so x_t is pure noise on the arc.
    const float t = trigflow_t_init();
    const float cos_t = std::cos(t), sin_t = std::sin(t);

    std::vector<float> noise(plane), scratch(plane);
    gaussian_noise_patch(seed_ + 5819, wi * kDecoderTileStride, wj * kDecoderTileStride,
                         S, S, 1, S, S, noise.data(), scratch.data());

    const int C_in = decoder_net_->config().in_channels;   // 1 + 4
    std::vector<float> x_in(static_cast<std::size_t>(C_in) * plane);
    std::vector<float> x_t(plane);
    for (std::size_t k = 0; k < plane; ++k) {
        x_t[k]  = sin_t * (noise[k] * kSigmaData);
        x_in[k] = x_t[k] / kSigmaData;
    }
    std::copy(up.begin(), up.end(), x_in.begin() + static_cast<std::ptrdiff_t>(plane));

    brotensor::Tensor xt = brodiffusion::detail::upload_host(
        x_in.data(), 1, static_cast<int>(x_in.size()));
    brotensor::Tensor yt;
    decoder_net_->forward(xt, /*N=*/1, S, &t, {}, yt);

    std::vector<float> pred;
    download_f32(yt, pred);

    TileBuffer out;
    out.shape = {kDecoderChannels, S, S};
    out.data.resize(static_cast<std::size_t>(kDecoderChannels) * plane);
    for (std::size_t k = 0; k < plane; ++k) {
        // Same negated-model-output convention as the latent stage.
        const float s = (cos_t * x_t[k] + sin_t * kSigmaData * pred[k]) / kSigmaData;
        out.data[k] = s * decoder_weight_window_[k];
    }
    std::copy(decoder_weight_window_.begin(), decoder_weight_window_.end(),
              out.data.begin() + static_cast<std::ptrdiff_t>(plane));
    return out;
}

TileBuffer WorldPipeline::residual(std::int64_t i1, std::int64_t j1,
                                   std::int64_t i2, std::int64_t j2) {
    return (*residual_)({Slice{0, kDecoderChannels}, Slice{i1, i2}, Slice{j1, j2}});
}

TileBuffer WorldPipeline::residual_normalized(std::int64_t i1, std::int64_t j1,
                                              std::int64_t i2, std::int64_t j2) {
    TileBuffer raw = residual(i1, j1, i2, j2);
    const std::size_t plane =
        static_cast<std::size_t>(raw.shape[1]) * static_cast<std::size_t>(raw.shape[2]);

    TileBuffer out;
    out.shape = {1, raw.shape[1], raw.shape[2]};
    out.data.resize(plane);
    const float* w = raw.data.data() + plane;
    for (std::size_t k = 0; k < plane; ++k) out.data[k] = raw.data[k] / w[k];
    return out;
}

TileBuffer WorldPipeline::elevation(std::int64_t i1, std::int64_t j1,
                                    std::int64_t i2, std::int64_t j2) {
    // Upstream's constants. The low band is stored standardised, so it comes
    // back as metres through these; the residual through the config's pair.
    constexpr double kLowfreqMean = -31.4;
    constexpr double kLowfreqStd  = 38.6;
    constexpr double kSigma       = 5.0;

    const std::int64_t scale = cfg_.latent_compression;

    // Pad by the blur's reach, then round OUTWARD to whole latent cells so the
    // padded region maps to an exact latent slice. floor/ceil rather than
    // truncating division: these coordinates go negative and truncation would
    // pull the low end toward the origin, silently shifting the low band by a
    // cell relative to the residual.
    const int kernel_size = (static_cast<int>(kSigma * 2) / 2) * 2 + 1;
    const std::int64_t pad_lr = kernel_size / 2 + 1;
    const std::int64_t pad_hr = pad_lr * scale;

    const std::int64_t pi1 = floor_div(i1 - pad_hr, scale) * scale;
    const std::int64_t pj1 = floor_div(j1 - pad_hr, scale) * scale;
    const std::int64_t pi2 = ceil_div(i2 + pad_hr, scale) * scale;
    const std::int64_t pj2 = ceil_div(j2 + pad_hr, scale) * scale;

    const int ph = static_cast<int>(pi2 - pi1);
    const int pw = static_cast<int>(pj2 - pj1);
    const int lh = static_cast<int>((pi2 - pi1) / scale);
    const int lw = static_cast<int>((pj2 - pj1) / scale);

    // High band: the decoder residual, weight-normalised and rescaled.
    TileBuffer res = residual(pi1, pj1, pi2, pj2);
    const std::size_t rplane = static_cast<std::size_t>(ph) * pw;
    std::vector<double> residual_p(rplane);
    {
        const float* w = res.data.data() + rplane;
        for (std::size_t k = 0; k < rplane; ++k) {
            residual_p[k] = static_cast<double>(res.data[k] / w[k]) * cfg_.residual_std +
                            cfg_.residual_mean;
        }
    }

    // Low band: latent channel 4, the one the decoder was NOT given.
    TileBuffer lat = latent(pi1 / scale, pj1 / scale, pi2 / scale, pj2 / scale);
    const std::size_t lplane = static_cast<std::size_t>(lh) * lw;
    std::vector<double> lowfreq_p(lplane);
    {
        const float* w = lat.data.data() +
                         static_cast<std::size_t>(kLatentChannels - 1) * lplane;
        const float* src = lat.data.data() + 4 * lplane;
        for (std::size_t k = 0; k < lplane; ++k) {
            lowfreq_p[k] = static_cast<double>(src[k] / w[k]) * kLowfreqStd + kLowfreqMean;
        }
    }

    std::vector<double> new_low, elev_p;
    laplacian_denoise(residual_p.data(), ph, pw, lowfreq_p.data(), lh, lw,
                      kSigma, new_low);
    laplacian_decode(residual_p.data(), ph, pw, new_low.data(), lh, lw,
                     /*extrapolate=*/false, elev_p);

    // Crop back to the request and undo the signed square root the whole
    // pipeline works in. That transform is why the models can represent both
    // ocean trenches and mountains without the deep end dominating the loss.
    const int oi = static_cast<int>(i1 - pi1), oj = static_cast<int>(j1 - pj1);
    const int h  = static_cast<int>(i2 - i1),  w  = static_cast<int>(j2 - j1);

    TileBuffer out;
    out.shape = {1, h, w};
    out.data.resize(static_cast<std::size_t>(h) * w);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const double v = elev_p[static_cast<std::size_t>(y + oi) * pw + (x + oj)];
            out.data[static_cast<std::size_t>(y) * w + x] =
                static_cast<float>(std::copysign(v * v, v));
        }
    }
    return out;
}

TileBuffer WorldPipeline::latent_init(std::int64_t i1, std::int64_t j1,
                                      std::int64_t i2, std::int64_t j2) {
    return (*latent_init_)({Slice{0, kLatentChannels}, Slice{i1, i2}, Slice{j1, j2}});
}

TileBuffer WorldPipeline::latent_normalized(std::int64_t i1, std::int64_t j1,
                                            std::int64_t i2, std::int64_t j2) {
    TileBuffer raw = latent(i1, j1, i2, j2);
    const std::size_t plane =
        static_cast<std::size_t>(raw.shape[1]) * static_cast<std::size_t>(raw.shape[2]);

    TileBuffer out;
    out.shape = {kLatentChannels - 1, raw.shape[1], raw.shape[2]};
    out.data.resize(static_cast<std::size_t>(kLatentChannels - 1) * plane);

    const float* w = raw.data.data() + static_cast<std::size_t>(kLatentChannels - 1) * plane;
    for (int c = 0; c < kLatentChannels - 1; ++c) {
        const float* src = raw.data.data() + static_cast<std::size_t>(c) * plane;
        float* dst = out.data.data() + static_cast<std::size_t>(c) * plane;
        for (std::size_t k = 0; k < plane; ++k) dst[k] = src[k] / w[k];
    }
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

void WorldPipeline::clear_cache() {
    coarse_->clear_cache();
    latent_init_->clear_cache();
    latent_step0_->clear_cache();
    residual_->clear_cache();
}

}  // namespace brodiffusion::terrain
