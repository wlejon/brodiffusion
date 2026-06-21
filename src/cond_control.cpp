// Conditioning-space control axes. See cond_control.h.

#include "brodiffusion/cond_control.h"

#include "brotensor/ops/elementwise.h"

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

void CondControl::load(const std::string& path) {
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

    dim_    = dim;
    names_  = std::move(names);
    scale_  = std::move(scale);
    dirs_   = std::move(dirs);
    index_  = std::move(index);
    weight_.assign(names_.size(), 0.0f);
}

void CondControl::set(const std::string& name, float alpha) {
    auto it = index_.find(name);
    if (it == index_.end()) {
        throw std::runtime_error("CondControl::set: no such axis '" + name + "'");
    }
    weight_[static_cast<std::size_t>(it->second)] = alpha;
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

void CondControl::apply(brotensor::Tensor& emb) const {
    if (!active()) return;
    const int rows = emb.rows;
    const int D    = emb.cols;
    if (D != dim_) {
        throw std::runtime_error(
            "CondControl::apply: embedding width " + std::to_string(D) +
            " != dictionary dim " + std::to_string(dim_) +
            " (dictionary built for a different text encoder)");
    }
    if (rows < 2) return;  // only BOS (or empty): nothing to steer

    // Combined injection vector v = Σ weight_k * scale_k * dir_k.
    std::vector<float> v(static_cast<std::size_t>(D), 0.0f);
    for (std::size_t k = 0; k < names_.size(); ++k) {
        const float w = weight_[k];
        if (w == 0.0f) continue;
        const float s = w * scale_[k];
        const float* d = &dirs_[k * static_cast<std::size_t>(D)];
        for (int j = 0; j < D; ++j) v[j] += s * d[j];
    }

    // Injection matrix: row 0 (BOS) zero, rows 1.. = v. Built FP32, cast to the
    // embedding dtype, added on the embedding's own device.
    std::vector<float> M(static_cast<std::size_t>(rows) * D, 0.0f);
    for (int r = 1; r < rows; ++r) {
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
