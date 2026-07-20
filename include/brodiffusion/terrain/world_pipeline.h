// brodiffusion/terrain/world_pipeline.h — the terrain-diffusion world generator.
//
// Port of xandergos/terrain-diffusion (MIT) —
// terrain_diffusion/inference/world_pipeline.py (WorldPipeline).
//
// This is the layer that turns the individually-ported pieces (MPUNet, the
// samplers, the tile-seeded RNG, the synthetic climate map, the infinite-tensor
// evaluator) into an actual infinite world. It owns the DAG:
//
//   base_coarse_map      7 ch  64x64 tiles, stride 48   <- synthetic map + noise
//     -> init_latent_map        6 ch  64x64 tiles       TrigFlow step 1
//     -> step_latent_map_0      6 ch  64x64 tiles       TrigFlow step 2
//     -> init_residual_map      2 ch  512x512 tiles     elevation Laplacian
//
// All three stages are wired. `elevation()` is the product — metres at
// native_resolution — but every intermediate is exposed too, because each is
// meaningful on its own: the coarse map is the world's elevation and climate at
// 7.7 km/cell, which is what every finer stage is conditioned on.
//
// ── The seventh channel ────────────────────────────────────────────────────
//
// The coarse UNet produces 6 channels, but the tensor carries 7. Tiles overlap
// (64 wide, 48 apart), and the evaluator's only blend operation is addition, so
// each window emits `[value * w, w]` for a linear taper `w` and the consumer
// divides the value sum by the weight sum at read time. That is why `w` must be
// strictly positive everywhere (upstream's `eps = 1e-3` floor) — a zero would
// divide 0/0 at the tile corners. `normalize()` below does the division; the
// raw 7-channel form is exposed because it is what the next stage consumes.
//
// ── Coordinates ────────────────────────────────────────────────────────────
//
// Coarse-map cells, not metres and not pixels of the final terrain. One coarse
// cell is `native_resolution * latent_compression * 32` metres on a side. World
// coordinates are signed and unbounded in both axes; nothing here has an origin
// bias, and a region generates identically regardless of what was generated
// before it.

#pragma once

#include "brodiffusion/terrain/infinite_tensor.h"
#include "brodiffusion/terrain/mp_unet.h"
#include "brodiffusion/terrain/sampler.h"
#include "brodiffusion/terrain/synthetic_map.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace brodiffusion::terrain {

// The `pipeline` block of a converted checkpoint's config.json. These are the
// checkpoint's constants, not tunables — the coarse means/stds are the
// normalisation the model was trained under, and cond_snr is the conditioning
// noise level it expects. Changing any of them changes what the network sees.
struct WorldPipelineConfig {
    // Per-output-channel normalisation for the coarse stage. Six entries:
    // elevation, the derived channel, then the four climate fields.
    double coarse_means[6] = {};
    double coarse_stds[6]  = {};

    // Signal-to-noise ratio per synthetic-map channel. The conditioning image is
    // cos(atan(snr)) * map + sin(atan(snr)) * noise, and log(tan(atan(snr))/8)
    // is fed to the network as a scalar conditioner so it knows how noisy the
    // conditioning it was handed actually is.
    double cond_snr[kSyntheticChannels] = {};

    double frequency_mult[kSyntheticChannels] = {};
    double drop_water_pct   = 0.5;
    int    coarse_pooling   = 1;
    int    latent_compression = 8;
    double native_resolution  = 30.0;
    double residual_mean = 0.0;
    double residual_std  = 0.7;

    // Parse the `pipeline` block. Throws if the file is missing or a required
    // key is absent — every value here changes the output, so a defaulted one
    // would generate a plausible but wrong world.
    static WorldPipelineConfig from_config_json(const std::string& config_path);
};

// The linear taper applied to each coarse tile before accumulation:
//   w(y, x) = (1 - (1-eps)*|y-mid|/mid) * (1 - (1-eps)*|x-mid|/mid)
// with mid = (size-1)/2 and eps = 1e-3. Strictly positive everywhere, so the
// weight-channel division never sees a zero denominator. Row-major, size*size.
std::vector<float> linear_weight_window(int size);

class WorldPipeline {
public:
    // `weights_dir` is a converted checkpoint directory (config.json,
    // coarse.safetensors, base.safetensors, decoder.safetensors,
    // synthetic_map_stats.json) — see scripts/convert-terrain-diffusion.py.
    WorldPipeline(const std::string& weights_dir, std::uint64_t seed);
    ~WorldPipeline();

    WorldPipeline(const WorldPipeline&)            = delete;
    WorldPipeline& operator=(const WorldPipeline&) = delete;

    const WorldPipelineConfig& config() const { return cfg_; }
    std::uint64_t              seed() const { return seed_; }

    // The coarse map over [i1, i2) x [j1, j2), as the raw 7-channel weighted
    // form: channels 0..5 are value*weight sums, channel 6 is the weight sum.
    // Shape (7, i2-i1, j2-j1).
    TileBuffer coarse(std::int64_t i1, std::int64_t j1,
                      std::int64_t i2, std::int64_t j2);

    // The same region with the weight division applied: shape (6, h, w).
    // This is the form to look at or render; `coarse()` is what the next stage
    // consumes.
    TileBuffer coarse_normalized(std::int64_t i1, std::int64_t j1,
                                 std::int64_t i2, std::int64_t j2);

    // The latent map over [i1, i2) x [j1, j2), in the same weighted form:
    // channels 0..4 are value*weight, channel 5 is the weight sum. Shape
    // (6, i2-i1, j2-j1). Latent cells are `latent_compression` times finer than
    // coarse cells; one latent cell covers native_resolution*8 metres.
    TileBuffer latent(std::int64_t i1, std::int64_t j1,
                      std::int64_t i2, std::int64_t j2);

    // The latent map with the weight division applied: shape (5, h, w).
    TileBuffer latent_normalized(std::int64_t i1, std::int64_t j1,
                                 std::int64_t i2, std::int64_t j2);

    // The elevation residual over [i1, i2) x [j1, j2), weighted form: channel 0
    // is value*weight, channel 1 the weight sum. Shape (2, i2-i1, j2-j1).
    // These are the FINEST cells the pipeline produces, one per
    // native_resolution metres, but they are a Laplacian residual and not
    // elevation. Turning them into metres needs the low-frequency band from the
    // latent map — that reconstruction is `elevation()` below.
    TileBuffer residual(std::int64_t i1, std::int64_t j1,
                        std::int64_t i2, std::int64_t j2);

    // The residual with the weight division applied: shape (1, h, w).
    TileBuffer residual_normalized(std::int64_t i1, std::int64_t j1,
                                   std::int64_t i2, std::int64_t j2);

    // ELEVATION IN METRES over [i1, i2) x [j1, j2), at native_resolution metres
    // per cell. Shape (1, i2-i1, j2-j1). This is the pipeline's actual product.
    //
    // Reconstructed from two bands: the decoder's high-pass residual at this
    // resolution and the latent map's channel 4 as the low-frequency band. The
    // request is padded outward before reconstruction and cropped afterwards,
    // because the blur and the resampling both reach beyond the requested edge —
    // without the pad, the boundary cells would be reconstructed from truncated
    // support and a region would not match the same region read as part of a
    // larger one.
    TileBuffer elevation(std::int64_t i1, std::int64_t j1,
                         std::int64_t i2, std::int64_t j2);

    // The FIRST of the latent stage's two TrigFlow steps, weighted form.
    // Exposed for the parity gate: reading it alone halves the composition
    // depth, which is what distinguishes error accumulating per step from
    // error arriving already-formed out of the coarse stage.
    TileBuffer latent_init(std::int64_t i1, std::int64_t j1,
                           std::int64_t i2, std::int64_t j2);

    // Drop every cached tile. Purely an optimisation — the world is a pure
    // function of (seed, position), so a cleared cache changes nothing but time.
    void clear_cache();

private:
    // Runs one coarse tile: builds its conditioning, denoises, and applies the
    // weight taper. `wi`/`wj` are window indices, not pixels.
    TileBuffer coarse_tile_(std::int64_t wi, std::int64_t wj);

    // One TrigFlow step of the latent stage, over a whole batch of windows.
    //
    // This is a single step rather than a loop because the DAG splits the two
    // steps across two InfiniteTensors: the first materializes from zero, the
    // second reads the first's output as an argument. Splitting them is what
    // lets a tile's first step be cached and reused by neighbours whose second
    // step needs it, so it is a structural choice and not an artifact.
    //
    // `prev` is null for the initial step (the sample starts at zero) and
    // otherwise carries the previous step's weighted tiles, one per window.
    std::vector<TileBuffer> latent_step_(
        const std::vector<std::vector<std::int64_t>>& windows,
        const std::vector<TileBuffer>&                coarse_tiles,
        const std::vector<TileBuffer>*                prev,
        float t, std::uint64_t seed_offset);

    // One decoder tile: upsample four latent channels 8x by nearest neighbour,
    // take a single TrigFlow step from zero, and apply the weight taper.
    TileBuffer residual_tile_(std::int64_t wi, std::int64_t wj,
                              const TileBuffer& latent_tile);

    WorldPipelineConfig cfg_;
    std::uint64_t       seed_ = 0;

    std::unique_ptr<MPUNet>          coarse_net_;
    std::unique_ptr<MPUNet>          base_net_;
    std::unique_ptr<MPUNet>          decoder_net_;
    std::unique_ptr<SyntheticMap>    synthetic_;
    std::unique_ptr<MemoryTileStore> store_;
    std::unique_ptr<InfiniteTensor>  coarse_;
    std::unique_ptr<InfiniteTensor>  latent_init_;
    std::unique_ptr<InfiniteTensor>  latent_step0_;
    std::unique_ptr<InfiniteTensor>  residual_;

    std::vector<float> weight_window_;        // 64x64, coarse stage
    std::vector<float> latent_weight_window_;  // 64x64, latent stage
    std::vector<float> decoder_weight_window_; // 512x512, decoder stage
    // log(tan(atan(snr))/8) per channel, one single-element vector each — the
    // shape MPUNet::forward expects for a batch-1 'float' conditioner.
    std::vector<std::vector<float>> cond_inputs_;
    std::vector<double>             t_cond_;   // atan(cond_snr), per channel
};

}  // namespace brodiffusion::terrain
