// Conditioning-space control axes. See cond_control.h.

#include "brodiffusion/cond_control.h"

#include "brotensor/ops/elementwise.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace brodiffusion {

namespace {

template <typename T>
T read_pod(std::ifstream& f, const char* what) {
    T v{};
    f.read(reinterpret_cast<char*>(&v), sizeof(T));
    if (!f) throw std::runtime_error(std::string("CondControl::load: truncated reading ") + what);
    return v;
}

}  // namespace

void CondControl::load(const std::string& path, bool merge) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("CondControl::load: cannot open " + path);

    char magic[4];
    f.read(magic, 4);
    if (!f || std::memcmp(magic, "BCD1", 4) != 0) {
        throw std::runtime_error("CondControl::load: bad magic (not a BCD1 dictionary): " + path);
    }
    const int n_axes = read_pod<std::int32_t>(f, "n_axes");
    const int dim    = read_pod<std::int32_t>(f, "dim");
    if (n_axes <= 0 || dim <= 0) {
        throw std::runtime_error("CondControl::load: nonpositive n_axes/dim in " + path);
    }

    std::vector<std::string> names;
    std::vector<float> scale, dirs;
    std::unordered_map<std::string, int> index;
    names.reserve(n_axes);
    scale.reserve(n_axes);
    dirs.resize(static_cast<std::size_t>(n_axes) * dim);

    for (int k = 0; k < n_axes; ++k) {
        const int len = read_pod<std::int32_t>(f, "name_len");
        if (len < 0 || len > (1 << 20)) {
            throw std::runtime_error("CondControl::load: implausible name length in " + path);
        }
        std::string name(static_cast<std::size_t>(len), '\0');
        f.read(name.data(), len);
        if (!f) throw std::runtime_error("CondControl::load: truncated reading axis name in " + path);
        const float s = read_pod<float>(f, "scale");
        f.read(reinterpret_cast<char*>(&dirs[static_cast<std::size_t>(k) * dim]),
               static_cast<std::streamsize>(sizeof(float) * dim));
        if (!f) throw std::runtime_error("CondControl::load: truncated reading axis direction in " + path);
        index[name] = k;
        names.push_back(std::move(name));
        scale.push_back(s);
    }

    if (!merge || !loaded()) {
        dim_    = dim;
        names_  = std::move(names);
        scale_  = std::move(scale);
        dirs_   = std::move(dirs);
        index_  = std::move(index);
        weight_.assign(names_.size(), 0.0f);
        return;
    }

    if (dim != dim_) {
        throw std::runtime_error(
            "CondControl::load: cannot merge " + path + " — dim " + std::to_string(dim) +
            " does not match the loaded dim " + std::to_string(dim_));
    }

    // Merge: overwrite same-named axes in place (weight reset — the direction it
    // referred to is gone), append the rest.
    for (int k = 0; k < n_axes; ++k) {
        const float* src = &dirs[static_cast<std::size_t>(k) * dim];
        auto it = index_.find(names[static_cast<std::size_t>(k)]);
        if (it != index_.end()) {
            const std::size_t j = static_cast<std::size_t>(it->second);
            std::copy(src, src + dim, dirs_.begin() + static_cast<std::ptrdiff_t>(j * dim));
            scale_[j]  = scale[static_cast<std::size_t>(k)];
            weight_[j] = 0.0f;
            continue;
        }
        index_[names[static_cast<std::size_t>(k)]] = static_cast<int>(names_.size());
        names_.push_back(names[static_cast<std::size_t>(k)]);
        scale_.push_back(scale[static_cast<std::size_t>(k)]);
        dirs_.insert(dirs_.end(), src, src + dim);
        weight_.push_back(0.0f);
    }
}

void CondControl::set(const std::string& name, float alpha) {
    auto it = index_.find(name);
    if (it == index_.end()) {
        throw std::runtime_error("CondControl::set: no such axis '" + name + "'");
    }
    weight_[static_cast<std::size_t>(it->second)] = alpha;
}

std::vector<float> CondControl::direction(const std::string& name) const {
    auto it = index_.find(name);
    if (it == index_.end()) {
        throw std::runtime_error("CondControl::direction: no such axis '" + name + "'");
    }
    const std::size_t k = static_cast<std::size_t>(it->second);
    return std::vector<float>(dirs_.begin() + k * dim_,
                              dirs_.begin() + (k + 1) * dim_);
}

float CondControl::axis_scale(const std::string& name) const {
    auto it = index_.find(name);
    if (it == index_.end()) {
        throw std::runtime_error("CondControl::axis_scale: no such axis '" + name + "'");
    }
    return scale_[static_cast<std::size_t>(it->second)];
}

void CondControl::set_vector(const std::string& name, float alpha,
                             const std::vector<float>& dir, float scale) {
    if (dir.empty()) {
        throw std::runtime_error("CondControl::set_vector: empty direction");
    }
    if (dim_ == 0) {
        dim_ = static_cast<int>(dir.size());
    } else if (static_cast<int>(dir.size()) != dim_) {
        throw std::runtime_error(
            "CondControl::set_vector: direction width " + std::to_string(dir.size()) +
            " != dim " + std::to_string(dim_));
    }
    auto it = index_.find(name);
    if (it == index_.end()) {
        const int k = static_cast<int>(names_.size());
        index_[name] = k;
        names_.push_back(name);
        scale_.push_back(scale);
        weight_.push_back(alpha);
        dirs_.insert(dirs_.end(), dir.begin(), dir.end());
    } else {
        const int k = it->second;
        scale_[static_cast<std::size_t>(k)]  = scale;
        weight_[static_cast<std::size_t>(k)] = alpha;
        std::copy(dir.begin(), dir.end(),
                  dirs_.begin() + static_cast<std::size_t>(k) * dim_);
    }
}

void CondControl::remove(const std::string& name) {
    auto it = index_.find(name);
    if (it == index_.end()) return;
    const std::size_t k = static_cast<std::size_t>(it->second);
    names_.erase(names_.begin() + k);
    scale_.erase(scale_.begin() + k);
    weight_.erase(weight_.begin() + k);
    dirs_.erase(dirs_.begin() + k * dim_, dirs_.begin() + (k + 1) * dim_);
    index_.clear();
    for (int i = 0; i < static_cast<int>(names_.size()); ++i) index_[names_[i]] = i;
    if (names_.empty()) dim_ = 0;
}

void CondControl::clear() {
    std::fill(weight_.begin(), weight_.end(), 0.0f);
}

bool CondControl::active() const {
    for (float w : weight_) {
        if (w != 0.0f) return true;
    }
    return false;
}

std::vector<float> CondControl::combined(float& alpha_norm) const {
    alpha_norm = 0.0f;
    if (dim_ == 0) return {};

    std::vector<float> v(static_cast<std::size_t>(dim_), 0.0f);
    double scale_sum = 0.0;
    int n_active = 0;
    for (std::size_t k = 0; k < names_.size(); ++k) {
        const float w = weight_[k];
        if (w == 0.0f) continue;
        const float s = w * scale_[k];
        const float* d = &dirs_[k * static_cast<std::size_t>(dim_)];
        for (int j = 0; j < dim_; ++j) v[j] += s * d[j];
        scale_sum += scale_[k];
        ++n_active;
    }
    if (n_active == 0) return {};

    double sq = 0.0;
    for (float x : v) sq += static_cast<double>(x) * x;
    // Alpha units: the axes' own natural injection scale. With one dictionary
    // they share a scale, so this is exactly "how many sliders' worth".
    const double unit = scale_sum / n_active;
    alpha_norm = unit > 0.0 ? static_cast<float>(std::sqrt(sq) / unit) : 0.0f;
    return v;
}

float CondControl::active_norm() const {
    float alpha = 0.0f;
    combined(alpha);
    return alpha;
}

void CondControl::apply(brotensor::Tensor& emb, int row_end,
                        int row_start) const {
    if (!active()) return;
    const int rows = emb.rows;
    const int D    = emb.cols;
    if (D != dim_) {
        throw std::runtime_error(
            "CondControl::apply: embedding width " + std::to_string(D) +
            " != dictionary dim " + std::to_string(dim_) +
            " (dictionary built for a different text encoder)");
    }
    // Steer rows [row_start, end): clamp the caller's row_end (CLIP EOS index)
    // into range; <0 means "all rows" (Sana, whose conditioning has no
    // padding tail).
    const int end = (row_end >= 0 && row_end <= rows) ? row_end : rows;
    if (end - row_start < 1) return;  // nothing to steer

    // Combined injection vector v = Σ weight_k * scale_k * dir_k, held to the
    // stack budget: a stack that overspends is scaled down by one common factor,
    // which keeps the caller's mix and only sheds the overdrive.
    float alpha = 0.0f;
    std::vector<float> v = combined(alpha);
    if (budget_ > 0.0f && alpha > budget_) {
        const float f = budget_ / alpha;
        for (float& x : v) x *= f;
    }

    // Injection matrix: rows [row_start, end) = v, all others (BOS + EOS/
    // padding tail when row_end clips it) zero. Built FP32, cast to the
    // embedding dtype, added on the embedding's own device.
    std::vector<float> M(static_cast<std::size_t>(rows) * D, 0.0f);
    for (int r = row_start; r < end; ++r) {
        std::memcpy(&M[static_cast<std::size_t>(r) * D], v.data(), sizeof(float) * D);
    }
    brotensor::Tensor inj =
        brotensor::Tensor::from_host_on(emb.device, M.data(), rows, D);
    if (emb.dtype == brotensor::Dtype::FP32) {
        brotensor::add_inplace(emb, inj);
    } else {
        brotensor::Tensor inj_cast;
        brotensor::cast(inj, inj_cast, emb.dtype);
        brotensor::add_inplace(emb, inj_cast);
    }
}

}  // namespace brodiffusion
