// brodiffusion/terrain/infinite_tensor.h — lazy sliding-window tensor graph.
//
// A port of the Python `infinite-tensor` library (MIT), reduced to the graph
// machinery terrain actually uses: window arithmetic, an in-memory tile store,
// and a demand-driven evaluator. No model code lives here — `f` is an opaque
// callback, so this file knows nothing about diffusion, UNets, or latents.
//
// Two coordinate systems, and confusing them is the classic bug:
//
//   pixel space  — raw tensor coordinates, what a caller asks for.
//   window space — one point per window; the index handed to `f`.
//
// The whole point is that a world can be unbounded: shape entries of -1 mark
// infinite dimensions (Python spells this `None`), window indices are signed and
// routinely negative, and cost scales with the region you ask for rather than
// with how far from the origin it sits.
//
// Deliberately NOT ported (terrain never exercises them): the HDF5/persistent
// store, `dimension_map`, custom `blend`/`blend_init` (terrain is additive), and
// `.to()`/`migrate`/serialization.

#pragma once

#include "brodiffusion/terrain/portable_rng.h"  // floor_div — see ceil_div below

#include <cstdint>
#include <functional>
#include <list>
#include <map>
#include <string>
#include <vector>

namespace brodiffusion::terrain {

// A half-open pixel-space range [start, stop). Upstream slices carry a `step`;
// nothing in the terrain graph ever uses a step other than 1, so this port drops
// it rather than carrying a field that is always 1 through every index formula.
struct Slice {
    std::int64_t start = 0;
    std::int64_t stop  = 0;

    std::int64_t extent() const { return stop > start ? stop - start : 0; }
};

// A dense N-D CPU float buffer in C order. Upstream window outputs are already
// `sample.cpu().float()`, so nothing richer is needed — and brotensor is 2-D
// only, so it cannot represent these shapes at all.
struct TileBuffer {
    std::vector<std::int64_t> shape;
    std::vector<float>        data;

    std::int64_t numel() const {
        std::int64_t n = 1;
        for (std::int64_t d : shape) n *= d;
        return shape.empty() ? 0 : n;
    }
};

// Ceiling division, the companion to portable_rng.h's floor_div. Both exist
// because C++ `/` truncates toward zero while Python's `//` floors, and window
// indices go negative the moment a caller reads west or north of the origin.
// Truncating division would round those toward zero instead of down, shifting
// the entire window lattice by one across the origin — a seam straight through
// the middle of the world. `b` is a stride and must be positive.
inline std::int64_t ceil_div(std::int64_t a, std::int64_t b) noexcept {
    return -floor_div(-a, b);
}

// ─── TensorWindow ──────────────────────────────────────────────────────────
//
// A sliding-window specification owning exactly two pieces of math: window
// index -> pixel bounds, and pixel bounds -> the set of window indices that
// touch them. Nothing else in this file does index arithmetic.
class TensorWindow {
public:
    TensorWindow() = default;

    // `stride` empty defaults to `size` (non-overlapping tiling); `offset` empty
    // defaults to zeros. Mismatched lengths throw.
    explicit TensorWindow(std::vector<std::int64_t> size,
                          std::vector<std::int64_t> stride = {},
                          std::vector<std::int64_t> offset = {});

    const std::vector<std::int64_t>& size() const { return size_; }
    const std::vector<std::int64_t>& stride() const { return stride_; }
    const std::vector<std::int64_t>& offset() const { return offset_; }
    std::size_t ndim() const { return size_.size(); }

    // Pixel-space extent of one window: start = w*stride + offset,
    // stop = start + size, per dimension.
    std::vector<Slice> get_bounds(const std::vector<std::int64_t>& window_index) const;

    // Every window index whose pixel extent intersects `pixel_slices`.
    //
    // `tensor_shape` filters windows that would hang off a bounded dimension:
    // -1 marks an infinite dim (no filtering on that axis), a positive value is
    // a hard extent. Pass an empty vector for no bounds filtering at all — that
    // is the port's spelling of upstream's `tensor_shape=None`.
    std::vector<std::vector<std::int64_t>> intersecting_windows(
        const std::vector<Slice>&        pixel_slices,
        const std::vector<std::int64_t>& tensor_shape = {}) const;

    bool operator==(const TensorWindow& o) const {
        return size_ == o.size_ && stride_ == o.stride_ && offset_ == o.offset_;
    }

private:
    std::vector<std::int64_t> size_;
    std::vector<std::int64_t> stride_;
    std::vector<std::int64_t> offset_;
};

// ─── MemoryTileStore ───────────────────────────────────────────────────────
//
// One LRU cache shared by every registered tensor, keyed by
// (tensor_id, window_index).
class MemoryTileStore {
public:
    // 100 MB matches upstream's default `cache_limit`.
    static constexpr std::size_t kDefaultCacheBytes = 100u * 1024u * 1024u;

    explicit MemoryTileStore(std::size_t cache_size_bytes = kDefaultCacheBytes)
        : cache_size_bytes_(cache_size_bytes) {}

    // Idempotent for a matching (shape, output_window); throws on a mismatch,
    // because reusing an id with different geometry would quietly reinterpret
    // whatever windows are already cached under it.
    void register_tensor(const std::string&               tensor_id,
                         const std::vector<std::int64_t>& shape,
                         const TensorWindow&              output_window);

    bool is_window_processed(const std::string&               tensor_id,
                             const std::vector<std::int64_t>& window_index) const;

    // Stores this window's output verbatim, replacing any previous entry for the
    // same key. It does NOT accumulate — see read_pixels for why.
    void notify_window_processed(const std::string&               tensor_id,
                                 const std::vector<std::int64_t>& window_index,
                                 TileBuffer                       output);

    // Assembles a pixel region by SUMMING every intersecting window's overlap
    // into a zeroed buffer.
    //
    // The write-stores / read-sums split is load-bearing, not an accident of the
    // port: it makes recomputing an evicted window idempotent. If windows were
    // accumulated into a shared canvas at write time, evicting one and later
    // regenerating it would add its contribution twice — silently, and only
    // under memory pressure, which is the worst possible way to find a bug. Do
    // not "optimize" this into a write-time accumulator.
    //
    // Plain `+=` is the correct combiner because the values being summed are
    // [value*weight, weight] pairs; consumers divide the value sum by the weight
    // sum to get a seamless blend across overlapping windows.
    TileBuffer read_pixels(const std::string&        tensor_id,
                           const std::vector<Slice>& pixel_slices);

    // Eviction is deferred while any access is open (the counter nests), so a
    // window cannot be evicted between being computed and being read by the
    // read_pixels call that demanded it.
    void begin_access();
    void end_access();

    void clear_cache(const std::string& tensor_id);
    void clear_tensor(const std::string& tensor_id);

    std::size_t cached_bytes() const { return bytes_; }
    std::size_t cached_windows() const { return entries_.size(); }

private:
    struct Registration {
        std::vector<std::int64_t> shape;
        TensorWindow              output_window;
    };
    using Key = std::pair<std::string, std::vector<std::int64_t>>;
    struct Entry {
        TileBuffer                          tile;
        std::list<Key>::iterator            lru;  // position in lru_, front = oldest
    };

    void evict();
    void drop_tensor_windows(const std::string& tensor_id);

    std::map<std::string, Registration> registry_;
    std::map<Key, Entry>                entries_;
    std::list<Key>                      lru_;
    std::size_t                         bytes_            = 0;
    std::size_t                         cache_size_bytes_ = kDefaultCacheBytes;
    int                                 access_depth_     = 0;
};

// ─── InfiniteTensor ────────────────────────────────────────────────────────

// The window generator. `window_indices` holds one window index per batch
// element (a batch of one when batch_size == 1), and `arg_tiles` is indexed
// [arg][batch]. Returns one TileBuffer per batch element, each shaped exactly
// like the output window's size.
using ComputeFn = std::function<std::vector<TileBuffer>(
    const std::vector<std::vector<std::int64_t>>& window_indices,
    const std::vector<std::vector<TileBuffer>>&   arg_tiles)>;

class InfiniteTensor {
public:
    // `shape` uses -1 for infinite dims. `args` are non-owning upstream pointers
    // and must outlive this tensor; `args_windows` must be the same length and
    // says how much of each upstream one output window needs. `store` is also
    // non-owning and must outlive every tensor registered in it.
    InfiniteTensor(std::vector<std::int64_t>    shape,
                   ComputeFn                    f,
                   TensorWindow                 output_window,
                   std::vector<InfiniteTensor*> args,
                   std::vector<TensorWindow>    args_windows,
                   std::int64_t                 batch_size,
                   MemoryTileStore*             store,
                   std::string                  tensor_id);

    const std::vector<std::int64_t>& shape() const { return shape_; }
    const TensorWindow&              output_window() const { return output_window_; }
    const std::string&               tensor_id() const { return tensor_id_; }
    std::int64_t                     batch_size() const { return batch_size_; }

    // The read entry point: materialize whatever windows this request needs
    // (recursing into args first), then assemble and return the region.
    TileBuffer operator()(const std::vector<Slice>& pixel_slices);

    void clear_cache() { store_->clear_cache(tensor_id_); }

private:
    void ensure_processed_range(const std::vector<std::vector<Slice>>& pixel_ranges);
    void ensure_processed(const std::vector<std::vector<std::int64_t>>& window_indices);
    void process_windows(const std::vector<std::vector<std::int64_t>>& windows);

    std::vector<std::int64_t>    shape_;
    ComputeFn                    f_;
    TensorWindow                 output_window_;
    std::vector<InfiniteTensor*> args_;
    std::vector<TensorWindow>    args_windows_;
    std::int64_t                 batch_size_ = 1;
    MemoryTileStore*             store_      = nullptr;
    std::string                  tensor_id_;
};

}  // namespace brodiffusion::terrain
