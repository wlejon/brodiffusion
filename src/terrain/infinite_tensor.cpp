#include "brodiffusion/terrain/infinite_tensor.h"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <string>

namespace brodiffusion::terrain {
namespace {

std::string vec_to_string(const std::vector<std::int64_t>& v) {
    std::string s = "(";
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) s += ", ";
        s += std::to_string(v[i]);
    }
    return s + ")";
}

std::size_t tile_bytes(const TileBuffer& t) { return t.data.size() * sizeof(float); }

}  // namespace

// ─── TensorWindow ──────────────────────────────────────────────────────────

TensorWindow::TensorWindow(std::vector<std::int64_t> size,
                           std::vector<std::int64_t> stride,
                           std::vector<std::int64_t> offset)
    : size_(std::move(size)), stride_(std::move(stride)), offset_(std::move(offset)) {
    if (size_.empty()) throw std::invalid_argument("TensorWindow: size must be non-empty");
    if (stride_.empty()) stride_ = size_;                    // non-overlapping tiling
    if (offset_.empty()) offset_.assign(size_.size(), 0);
    if (stride_.size() != size_.size())
        throw std::invalid_argument("TensorWindow: stride length must match size length");
    if (offset_.size() != size_.size())
        throw std::invalid_argument("TensorWindow: offset length must match size length");
    for (std::int64_t s : stride_)
        if (s <= 0) throw std::invalid_argument("TensorWindow: stride must be positive");
}

std::vector<Slice> TensorWindow::get_bounds(
    const std::vector<std::int64_t>& window_index) const {
    if (window_index.size() != size_.size())
        throw std::invalid_argument("TensorWindow::get_bounds: window index rank mismatch");
    std::vector<Slice> bounds(size_.size());
    for (std::size_t d = 0; d < size_.size(); ++d) {
        const std::int64_t base = window_index[d] * stride_[d] + offset_[d];
        bounds[d] = Slice{base, base + size_[d]};
    }
    return bounds;
}

std::vector<std::vector<std::int64_t>> TensorWindow::intersecting_windows(
    const std::vector<Slice>&        pixel_slices,
    const std::vector<std::int64_t>& tensor_shape) const {
    const std::size_t n = size_.size();
    if (pixel_slices.size() != n)
        throw std::invalid_argument("TensorWindow::intersecting_windows: slice rank mismatch");

    // Per dim: window w covers [w*stride + offset, w*stride + offset + size), so
    // it touches [start, stop) iff
    //     w*stride + offset + size > start   and   w*stride + offset < stop
    // which solves to low = ceil((start - offset - size + 1)/stride) and
    // high = floor((stop - 1 - offset)/stride).
    //
    // These MUST be floor/ceil, not C truncating division. Every term here goes
    // negative for regions west or north of the origin, and truncation would
    // round those toward zero — shifting the whole window lattice by one on the
    // negative side of each axis. This is the single easiest thing to get wrong
    // in this file, and the symptom (a seam through the origin) shows up far
    // from the cause.
    std::vector<std::int64_t> low(n), high(n);
    for (std::size_t d = 0; d < n; ++d) {
        const std::int64_t numerator = pixel_slices[d].start - offset_[d] - size_[d] + 1;
        low[d]  = ceil_div(numerator, stride_[d]);
        high[d] = floor_div(pixel_slices[d].stop - 1 - offset_[d], stride_[d]);
        if (low[d] > high[d]) return {};  // empty in any dim => empty overall
    }

    // Cartesian product of the per-dim ranges, walked as an odometer with the
    // last dim fastest (matching itertools.product's order).
    std::vector<std::vector<std::int64_t>> out;
    std::vector<std::int64_t>              cur = low;
    const bool                             filter = !tensor_shape.empty();
    for (;;) {
        bool keep = true;
        if (filter) {
            const std::vector<Slice> bounds = get_bounds(cur);
            for (std::size_t d = 0; d < bounds.size() && d < tensor_shape.size(); ++d) {
                if (tensor_shape[d] < 0) continue;  // -1 == infinite dim, never clipped
                if (bounds[d].start < 0 || bounds[d].stop > tensor_shape[d]) {
                    keep = false;
                    break;
                }
            }
        }
        if (keep) out.push_back(cur);

        std::size_t d = n;
        while (d > 0) {
            --d;
            if (++cur[d] <= high[d]) break;
            cur[d] = low[d];
            if (d == 0) return out;  // odometer wrapped past the slowest dim
        }
    }
}

// ─── MemoryTileStore ───────────────────────────────────────────────────────

void MemoryTileStore::register_tensor(const std::string&               tensor_id,
                                      const std::vector<std::int64_t>& shape,
                                      const TensorWindow&              output_window) {
    auto it = registry_.find(tensor_id);
    if (it != registry_.end()) {
        if (it->second.shape != shape || !(it->second.output_window == output_window)) {
            throw std::invalid_argument("MemoryTileStore: tensor '" + tensor_id +
                                        "' re-registered with mismatched metadata (existing shape " +
                                        vec_to_string(it->second.shape) + ", new shape " +
                                        vec_to_string(shape) + ")");
        }
        return;
    }
    registry_.emplace(tensor_id, Registration{shape, output_window});
}

bool MemoryTileStore::is_window_processed(
    const std::string& tensor_id, const std::vector<std::int64_t>& window_index) const {
    return entries_.find(Key{tensor_id, window_index}) != entries_.end();
}

void MemoryTileStore::notify_window_processed(const std::string&               tensor_id,
                                              const std::vector<std::int64_t>& window_index,
                                              TileBuffer                       output) {
    const Key key{tensor_id, window_index};
    auto      it = entries_.find(key);
    if (it != entries_.end()) {
        bytes_ -= tile_bytes(it->second.tile);
        lru_.erase(it->second.lru);
        entries_.erase(it);
    }
    const std::size_t added = tile_bytes(output);
    lru_.push_back(key);
    Entry entry;
    entry.tile = std::move(output);
    entry.lru  = std::prev(lru_.end());
    entries_.emplace(key, std::move(entry));
    bytes_ += added;

    // Only trim when no access is open; otherwise we could drop a window the
    // in-flight read_pixels is about to sum.
    if (access_depth_ == 0) evict();
}

void MemoryTileStore::begin_access() { ++access_depth_; }

void MemoryTileStore::end_access() {
    --access_depth_;
    if (access_depth_ < 0)
        throw std::logic_error("MemoryTileStore: end_access without matching begin_access");
    if (access_depth_ == 0) evict();
}

void MemoryTileStore::evict() {
    while (!lru_.empty() && bytes_ > cache_size_bytes_) {
        const Key key = lru_.front();
        auto      it  = entries_.find(key);
        if (it != entries_.end()) {
            bytes_ -= tile_bytes(it->second.tile);
            entries_.erase(it);
        }
        lru_.pop_front();
    }
}

void MemoryTileStore::drop_tensor_windows(const std::string& tensor_id) {
    for (auto it = entries_.begin(); it != entries_.end();) {
        if (it->first.first == tensor_id) {
            bytes_ -= tile_bytes(it->second.tile);
            lru_.erase(it->second.lru);
            it = entries_.erase(it);
        } else {
            ++it;
        }
    }
}

void MemoryTileStore::clear_cache(const std::string& tensor_id) {
    if (registry_.find(tensor_id) != registry_.end()) drop_tensor_windows(tensor_id);
}

void MemoryTileStore::clear_tensor(const std::string& tensor_id) {
    registry_.erase(tensor_id);
    drop_tensor_windows(tensor_id);
}

TileBuffer MemoryTileStore::read_pixels(const std::string&        tensor_id,
                                        const std::vector<Slice>& pixel_slices) {
    auto reg_it = registry_.find(tensor_id);
    if (reg_it == registry_.end())
        throw std::invalid_argument("MemoryTileStore::read_pixels: unregistered tensor '" +
                                    tensor_id + "'");
    const Registration& reg = reg_it->second;
    const std::size_t   n   = pixel_slices.size();

    TileBuffer out;
    out.shape.resize(n);
    std::int64_t total = 1;
    for (std::size_t d = 0; d < n; ++d) {
        out.shape[d] = pixel_slices[d].extent();
        total *= out.shape[d];
    }
    out.data.assign(static_cast<std::size_t>(total), 0.0f);  // sum accumulator starts at zero
    if (total == 0) return out;

    // Row strides for the request buffer; the window buffers all share the
    // output window's size, so their strides are computed once too.
    const std::vector<std::int64_t>& win_shape = reg.output_window.size();
    std::vector<std::int64_t>        dst_stride(n), src_stride(n);
    std::int64_t                     ds = 1, ss = 1;
    for (std::size_t i = n; i > 0; --i) {
        dst_stride[i - 1] = ds;
        ds *= out.shape[i - 1];
        src_stride[i - 1] = ss;
        ss *= win_shape[i - 1];
    }

    for (const auto& window_index :
         reg.output_window.intersecting_windows(pixel_slices, reg.shape)) {
        auto ent = entries_.find(Key{tensor_id, window_index});
        if (ent == entries_.end()) {
            throw std::logic_error("MemoryTileStore::read_pixels: window " +
                                   vec_to_string(window_index) + " of tensor '" + tensor_id +
                                   "' was not processed");
        }
        // Reading bumps the window to most-recently-used.
        lru_.splice(lru_.end(), lru_, ent->second.lru);
        ent->second.lru        = std::prev(lru_.end());
        const TileBuffer& tile = ent->second.tile;

        const std::vector<Slice> bounds = reg.output_window.get_bounds(window_index);
        std::vector<std::int64_t> len(n);
        std::int64_t              src_base = 0, dst_base = 0;
        bool                      empty = false;
        for (std::size_t d = 0; d < n; ++d) {
            const std::int64_t start = std::max(pixel_slices[d].start, bounds[d].start);
            const std::int64_t stop  = std::min(pixel_slices[d].stop, bounds[d].stop);
            if (start >= stop) {
                empty = true;
                break;
            }
            len[d] = stop - start;
            src_base += (start - bounds[d].start) * src_stride[d];
            dst_base += (start - pixel_slices[d].start) * dst_stride[d];
        }
        if (empty) continue;  // bounds-filtered lattice can still touch only a corner

        // Walk every row of the intersection; the innermost dim is contiguous in
        // both buffers, so it becomes a straight `+=` run.
        const std::int64_t          run = len[n - 1];
        std::int64_t                rows = 1;
        for (std::size_t d = 0; d + 1 < n; ++d) rows *= len[d];
        std::vector<std::int64_t> counter(n > 0 ? n - 1 : 0, 0);
        for (std::int64_t r = 0; r < rows; ++r) {
            std::int64_t so = src_base, dof = dst_base;
            for (std::size_t d = 0; d + 1 < n; ++d) {
                so += counter[d] * src_stride[d];
                dof += counter[d] * dst_stride[d];
            }
            const float* src = tile.data.data() + so;
            float*       dst = out.data.data() + dof;
            for (std::int64_t k = 0; k < run; ++k) dst[k] += src[k];

            for (std::size_t d = counter.size(); d > 0; --d) {
                if (++counter[d - 1] < len[d - 1]) break;
                counter[d - 1] = 0;
            }
        }
    }
    return out;
}

// ─── InfiniteTensor ────────────────────────────────────────────────────────

InfiniteTensor::InfiniteTensor(std::vector<std::int64_t>    shape,
                               ComputeFn                    f,
                               TensorWindow                 output_window,
                               std::vector<InfiniteTensor*> args,
                               std::vector<TensorWindow>    args_windows,
                               std::int64_t                 batch_size,
                               MemoryTileStore*             store,
                               std::string                  tensor_id)
    : shape_(std::move(shape)),
      f_(std::move(f)),
      output_window_(std::move(output_window)),
      args_(std::move(args)),
      args_windows_(std::move(args_windows)),
      batch_size_(batch_size),
      store_(store),
      tensor_id_(std::move(tensor_id)) {
    if (!f_) throw std::invalid_argument("InfiniteTensor: compute function must be callable");
    if (!store_) throw std::invalid_argument("InfiniteTensor: tile store must not be null");
    if (shape_.empty()) throw std::invalid_argument("InfiniteTensor: shape cannot be empty");
    for (std::int64_t d : shape_)
        if (d != -1 && d <= 0)
            throw std::invalid_argument("InfiniteTensor: dims must be -1 (infinite) or positive");
    if (output_window_.ndim() != shape_.size())
        throw std::invalid_argument("InfiniteTensor: output_window rank must match shape rank");
    if (args_windows_.size() != args_.size())
        throw std::invalid_argument("InfiniteTensor: args_windows length must match args length");
    for (std::size_t i = 0; i < args_.size(); ++i) {
        if (!args_[i]) throw std::invalid_argument("InfiniteTensor: null arg tensor");
        if (args_windows_[i].ndim() != args_[i]->shape().size())
            throw std::invalid_argument("InfiniteTensor: args_windows[" + std::to_string(i) +
                                        "] rank must match that arg tensor's rank");
    }
    if (batch_size_ < 1) throw std::invalid_argument("InfiniteTensor: batch_size must be >= 1");

    store_->register_tensor(tensor_id_, shape_, output_window_);
}

TileBuffer InfiniteTensor::operator()(const std::vector<Slice>& pixel_slices) {
    TileStoreAccess guard(store_);
    ensure_processed_range({pixel_slices});
    return store_->read_pixels(tensor_id_, pixel_slices);
}

void InfiniteTensor::ensure_processed_range(
    const std::vector<std::vector<Slice>>& pixel_ranges) {
    std::vector<std::vector<std::int64_t>> all;
    for (const auto& range : pixel_ranges) {
        auto windows = output_window_.intersecting_windows(range, shape_);
        all.insert(all.end(), windows.begin(), windows.end());
    }
    ensure_processed(all);
}

void InfiniteTensor::ensure_processed(
    const std::vector<std::vector<std::int64_t>>& window_indices) {
    // Dedup twice over: skip anything the store already has, and skip repeats
    // within this request. A window is never computed twice in one run.
    std::vector<std::vector<std::int64_t>> pending;
    std::set<std::vector<std::int64_t>>    seen;
    for (const auto& w : window_indices) {
        if (store_->is_window_processed(tensor_id_, w)) continue;
        if (seen.insert(w).second) pending.push_back(w);
    }

    // Hold an access open across the upstream materialization so nothing an arg
    // produces here can be evicted before process_windows reads it back.
    TileStoreAccess guard(store_);
    for (std::size_t i = 0; i < args_.size(); ++i) {
        std::vector<std::vector<Slice>> ranges;
        ranges.reserve(pending.size());
        for (const auto& w : pending) ranges.push_back(args_windows_[i].get_bounds(w));
        args_[i]->ensure_processed_range(ranges);
    }
    process_windows(pending);
}

void InfiniteTensor::process_windows(
    const std::vector<std::vector<std::int64_t>>& windows) {
    const std::size_t batch = static_cast<std::size_t>(batch_size_);
    for (std::size_t begin = 0; begin < windows.size(); begin += batch) {
        const std::size_t end = std::min(begin + batch, windows.size());
        const std::vector<std::vector<std::int64_t>> batch_windows(windows.begin() + static_cast<std::ptrdiff_t>(begin),
                                                                  windows.begin() + static_cast<std::ptrdiff_t>(end));

        // [arg][batch] — the upstream reads go through operator(), which is a
        // no-op re-check after ensure_processed_range above, then a read.
        std::vector<std::vector<TileBuffer>> arg_tiles(args_.size());
        for (std::size_t i = 0; i < args_.size(); ++i) {
            arg_tiles[i].reserve(batch_windows.size());
            for (const auto& w : batch_windows)
                arg_tiles[i].push_back((*args_[i])(args_windows_[i].get_bounds(w)));
        }

        std::vector<TileBuffer> outputs = f_(batch_windows, arg_tiles);
        if (outputs.size() != batch_windows.size())
            throw std::runtime_error("InfiniteTensor: compute function returned " +
                                     std::to_string(outputs.size()) + " tiles for a batch of " +
                                     std::to_string(batch_windows.size()));

        for (std::size_t b = 0; b < batch_windows.size(); ++b) {
            if (outputs[b].shape != output_window_.size()) {
                throw std::runtime_error("InfiniteTensor: output shape " +
                                         vec_to_string(outputs[b].shape) +
                                         " does not match window shape " +
                                         vec_to_string(output_window_.size()));
            }
            store_->notify_window_processed(tensor_id_, batch_windows[b], std::move(outputs[b]));
        }
    }
}

}  // namespace brodiffusion::terrain
