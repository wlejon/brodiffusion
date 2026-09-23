// bro.diffusion.loadTerrain — the terrain-diffusion world generator
// (brodiffusion/terrain/world_pipeline.h) as a TerrainWorld handle.
//
// A world is a pure function of (seed, position): every read below is a
// region [i1, i2) x [j1, j2) — i rows (north-south), j columns — at one of the
// pipeline's four resolutions, answered from a tile cache the handle owns.
// Coordinates are signed and unbounded. Each read is one synchronous native
// call that may run the three UNets over every tile the region touches, so a
// large cold read belongs in a Worker, like Pipeline.generate().
//
// Every read returns { channels, height, width, data: Float32Array } with data
// channel-major (channels x height x width). The coarse / latent / residual
// reads return the normalised form unless opts.weighted, which hands back the
// raw weighted sums with the weight channel last — the form the next stage
// consumes.

#include "host_diffusion_internal.h"

#include <brodiffusion/terrain/world_pipeline.h>
#include <brotensor/runtime.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace brodiffusion::api {

namespace {

namespace td = brodiffusion::terrain;

inline constexpr uint32_t kHostTerrainTag = 0x54455252u;  // 'TERR'

struct TerrainWrapper {
    uint32_t tag = kHostTerrainTag;
    std::unique_ptr<td::WorldPipeline> world;
};

HostClass g_terrainClass;

TerrainWrapper* unwrapTerrain(Value v) {
    auto* w = static_cast<TerrainWrapper*>(g_terrainClass.unwrap(v));
    return (w && w->tag == kHostTerrainTag) ? w : nullptr;
}

// One read may not ask for more than this many cells per channel: the
// elevation stage runs 512x512 decoder tiles, and a mistyped coordinate would
// otherwise try to materialize a continent.
constexpr double kMaxCells = 64.0 * 1024.0 * 1024.0;

Value tileToJs(const td::TileBuffer& t) {
    int64_t c = 1, h = 1, w = 1;
    if (t.shape.size() == 3) {
        c = t.shape[0]; h = t.shape[1]; w = t.shape[2];
    } else if (t.shape.size() == 2) {
        h = t.shape[0]; w = t.shape[1];
    } else if (t.shape.size() == 1) {
        w = t.shape[0];
    }
    ObjectBuilder o;
    o.set("channels", static_cast<double>(c));
    o.set("height", static_cast<double>(h));
    o.set("width", static_cast<double>(w));
    ev::Persistent d(makeFloat32Array(t.data.data(), t.data.size()));
    o.set("data", d.get());
    return o.build();
}

// Read (i1, j1, i2, j2) from args[0..3]; false with `err` set when they are
// not integers bounding a non-empty region of sane size.
bool readRegion(std::span<const Value> args, int64_t r[4], std::string& err) {
    if (args.size() < 4) {
        err = "(i1, j1, i2, j2) required";
        return false;
    }
    for (int k = 0; k < 4; ++k) {
        if (!ev::isNumber(args[static_cast<size_t>(k)])) {
            err = "(i1, j1, i2, j2) must be integers";
            return false;
        }
        const double d = ev::toDouble(args[static_cast<size_t>(k)]);
        if (!std::isfinite(d) || d != std::floor(d) || std::fabs(d) > 9.0e15) {
            err = "(i1, j1, i2, j2) must be integers";
            return false;
        }
        r[k] = static_cast<int64_t>(d);
    }
    if (r[2] <= r[0] || r[3] <= r[1]) {
        err = "the region is empty (need i2 > i1 and j2 > j1)";
        return false;
    }
    const double cells = static_cast<double>(r[2] - r[0]) * static_cast<double>(r[3] - r[1]);
    if (cells > kMaxCells) {
        err = "the region is " + std::to_string(static_cast<long long>(cells)) +
              " cells; one read is limited to " +
              std::to_string(static_cast<long long>(kMaxCells));
        return false;
    }
    return true;
}

enum class Stage { Coarse, Latent, Residual, Elevation };

Value readStage(Value thisVal, std::span<const Value> args, Stage stage, const char* who) {
    auto* w = unwrapTerrain(thisVal);
    if (!w || !w->world) {
        return ev::throwTypeError(std::string("TerrainWorld.") + who + ": not a loaded TerrainWorld");
    }
    int64_t r[4];
    std::string err;
    if (!readRegion(args, r, err)) {
        return ev::throwTypeError(std::string("TerrainWorld.") + who + ": " + err);
    }
    const bool weighted = args.size() > 4 && propBool(args[4], "weighted");
    try {
        td::TileBuffer t;
        switch (stage) {
            case Stage::Coarse:
                t = weighted ? w->world->coarse(r[0], r[1], r[2], r[3])
                             : w->world->coarse_normalized(r[0], r[1], r[2], r[3]);
                break;
            case Stage::Latent:
                t = weighted ? w->world->latent(r[0], r[1], r[2], r[3])
                             : w->world->latent_normalized(r[0], r[1], r[2], r[3]);
                break;
            case Stage::Residual:
                t = weighted ? w->world->residual(r[0], r[1], r[2], r[3])
                             : w->world->residual_normalized(r[0], r[1], r[2], r[3]);
                break;
            case Stage::Elevation:
                t = w->world->elevation(r[0], r[1], r[2], r[3]);
                break;
        }
        return tileToJs(t);
    } catch (const std::exception& e) {
        return ev::throwError(std::string("TerrainWorld.") + who + " failed: " + e.what());
    }
}

void decorateTerrainProto(ObjectBuilder& proto) {
    proto.def("elevation", 4, [](Value self, std::span<const Value> a) {
        return readStage(self, a, Stage::Elevation, "elevation");
    });
    proto.def("coarse", 5, [](Value self, std::span<const Value> a) {
        return readStage(self, a, Stage::Coarse, "coarse");
    });
    proto.def("latent", 5, [](Value self, std::span<const Value> a) {
        return readStage(self, a, Stage::Latent, "latent");
    });
    proto.def("residual", 5, [](Value self, std::span<const Value> a) {
        return readStage(self, a, Stage::Residual, "residual");
    });
    proto.def("clearCache", 0, [](Value self, std::span<const Value>) -> Value {
        auto* w = unwrapTerrain(self);
        if (!w || !w->world) return ev::throwTypeError("TerrainWorld.clearCache: not a loaded TerrainWorld");
        w->world->clear_cache();
        return ev::undefined();
    });
    proto.def("dispose", 0, [](Value self, std::span<const Value>) -> Value {
        auto* w = unwrapTerrain(self);
        if (!w) return ev::throwTypeError("TerrainWorld.dispose: not a TerrainWorld");
        w->world.reset();
        return ev::undefined();
    });
    // The seed as a number, or as a decimal string when it does not fit a
    // double exactly (the embed API cannot mint a BigInt).
    proto.accessor("seed", [](Value self, std::span<const Value>) -> Value {
        auto* w = unwrapTerrain(self);
        if (!w || !w->world) return ev::undefined();
        const uint64_t s = w->world->seed();
        if (s <= (uint64_t{1} << 53)) return ev::fromDouble(static_cast<double>(s));
        return ev::fromUtf8(std::to_string(s));
    });
    // The checkpoint's constants that say what one cell of each read is.
    proto.def("config", 0, [](Value self, std::span<const Value>) -> Value {
        auto* w = unwrapTerrain(self);
        if (!w || !w->world) return ev::throwTypeError("TerrainWorld.config: not a loaded TerrainWorld");
        const td::WorldPipelineConfig& c = w->world->config();
        const double native = c.native_resolution;
        ObjectBuilder o;
        o.set("nativeResolution", native);
        o.set("latentCompression", static_cast<double>(c.latent_compression));
        o.set("elevationCellMetres", native);
        o.set("residualCellMetres", native);
        o.set("latentCellMetres", native * c.latent_compression);
        o.set("coarseCellMetres", native * c.latent_compression * 32.0);
        o.set("residualMean", c.residual_mean);
        o.set("residualStd", c.residual_std);
        o.set("dropWaterPct", c.drop_water_pct);
        return o.build();
    });
}

}  // namespace

void ensureTerrainClassInstalled() {
    // Per thread, like ensureDiffusionClassesInstalled.
    static thread_local bool installed = false;
    if (installed) return;
    installed = true;
    g_terrainClass.install("TerrainWorld", 0, nullptr, decorateTerrainProto);
}

Value terrainConstructor() { return g_terrainClass.constructor(); }

// loadTerrain(weightsDir, { seed? }) -> TerrainWorld
Value loadTerrain(Value, std::span<const Value> args) {
    if (args.empty() || !ev::isString(args[0])) {
        return ev::throwTypeError("bro.diffusion.loadTerrain(weightsDir, opts?): weightsDir string required");
    }
    const std::string dir = resolveDiffusionPath(ev::toUtf8(args[0]));
    uint64_t seed = 0;
    if (args.size() > 1 && ev::isObject(args[1])) {
        Value sv = ev::getProperty(args[1], "seed");
        if (ev::isBigInt(sv)) {
            seed = ev::toUint64(sv);
        } else if (ev::isNumber(sv)) {
            const double d = ev::toDouble(sv);
            if (!std::isfinite(d) || d < 0 || d != std::floor(d) || d > 9007199254740992.0) {
                return ev::throwTypeError(
                    "bro.diffusion.loadTerrain: opts.seed must be a non-negative integer (or a BigInt)");
            }
            seed = static_cast<uint64_t>(d);
        } else if (!ev::isUndefined(sv)) {
            return ev::throwTypeError(
                "bro.diffusion.loadTerrain: opts.seed must be a non-negative integer (or a BigInt)");
        }
    }
    if (!std::filesystem::exists(std::filesystem::path(dir) / "config.json")) {
        return ev::throwError("bro.diffusion.loadTerrain: no config.json in " + dir +
                              " (a converted terrain-diffusion checkpoint dir)");
    }
    try {
        brotensor::init();
        auto w = std::make_unique<TerrainWrapper>();
        w->world = std::make_unique<td::WorldPipeline>(dir, seed);
        return g_terrainClass.createInstance(std::move(w));
    } catch (const std::exception& e) {
        return ev::throwError(std::string("bro.diffusion.loadTerrain failed: ") + e.what());
    }
}

}  // namespace brodiffusion::api
