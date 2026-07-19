#include "brodiffusion/terrain/laplacian.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace brodiffusion::terrain {
namespace {

// One output sample's taps: the first input index, and the normalised weights.
struct Taps {
    int                 first = 0;
    std::vector<double> w;
};

// PIL's resampling geometry. `support` is the filter radius in INPUT samples: 1
// for the bilinear triangle, stretched by the scale factor when minifying so the
// filter averages everything it is discarding (that stretch is the antialiasing).
// When magnifying, scale < 1, support clamps to 1, and this reduces exactly to
// align-corners-false bilinear.
std::vector<Taps> build_taps(int in_size, int out_size) {
    const double scale   = static_cast<double>(in_size) / out_size;
    const double support = std::max(1.0, scale);

    std::vector<Taps> taps(static_cast<std::size_t>(out_size));
    for (int i = 0; i < out_size; ++i) {
        const double center = (i + 0.5) * scale;
        const int lo = static_cast<int>(std::max(
            0.0, std::floor(center - support + 0.5)));
        const int hi = static_cast<int>(std::min(
            static_cast<double>(in_size), std::ceil(center + support + 0.5)));

        Taps& t = taps[static_cast<std::size_t>(i)];
        t.first = lo;
        t.w.reserve(static_cast<std::size_t>(hi - lo));
        double sum = 0.0;
        for (int k = lo; k < hi; ++k) {
            double v = 1.0 - std::abs((k + 0.5 - center) / support);
            if (v < 0.0) v = 0.0;
            t.w.push_back(v);
            sum += v;
        }
        // A degenerate window would mean the geometry above is wrong, not that
        // the input was unusual; fail rather than emit silent zeros.
        if (sum <= 0.0) throw std::runtime_error("laplacian: empty resize kernel");
        for (double& v : t.w) v /= sum;
    }
    return taps;
}

}  // namespace

void resize_bilinear(const double* src, int ih, int iw,
                     int oh, int ow, std::vector<double>& out) {
    if (ih <= 0 || iw <= 0 || oh <= 0 || ow <= 0) {
        throw std::runtime_error("laplacian: resize with a non-positive extent");
    }
    const std::vector<Taps> tx = build_taps(iw, ow);
    const std::vector<Taps> ty = build_taps(ih, oh);

    // Horizontal first into a (ih x ow) intermediate, then vertical. Separable,
    // so this is O(n*taps) rather than O(n*taps^2).
    std::vector<double> tmp(static_cast<std::size_t>(ih) * ow);
    for (int y = 0; y < ih; ++y) {
        const double* row = src + static_cast<std::size_t>(y) * iw;
        double* dst = tmp.data() + static_cast<std::size_t>(y) * ow;
        for (int x = 0; x < ow; ++x) {
            const Taps& t = tx[static_cast<std::size_t>(x)];
            double acc = 0.0;
            for (std::size_t k = 0; k < t.w.size(); ++k) {
                acc += row[static_cast<std::size_t>(t.first) + k] * t.w[k];
            }
            dst[x] = acc;
        }
    }

    out.assign(static_cast<std::size_t>(oh) * ow, 0.0);
    for (int y = 0; y < oh; ++y) {
        const Taps& t = ty[static_cast<std::size_t>(y)];
        double* dst = out.data() + static_cast<std::size_t>(y) * ow;
        for (std::size_t k = 0; k < t.w.size(); ++k) {
            const double* row = tmp.data() +
                (static_cast<std::size_t>(t.first) + k) * ow;
            const double wk = t.w[k];
            for (int x = 0; x < ow; ++x) dst[x] += row[x] * wk;
        }
    }
}

void gaussian_blur(const double* src, int h, int w,
                   int kernel_size, double sigma, std::vector<double>& out) {
    if (kernel_size <= 0 || (kernel_size % 2) == 0) {
        throw std::runtime_error("laplacian: gaussian_blur needs an odd kernel size");
    }
    const int p = kernel_size / 2;
    if (p >= h || p >= w) {
        // Reflect padding cannot produce more than (extent - 1) samples per side.
        throw std::runtime_error("laplacian: gaussian_blur kernel wider than the input");
    }

    std::vector<double> k(static_cast<std::size_t>(kernel_size));
    double sum = 0.0;
    for (int i = 0; i < kernel_size; ++i) {
        const double x = i - (kernel_size - 1) / 2.0;
        k[static_cast<std::size_t>(i)] = std::exp(-(x / sigma) * (x / sigma) / 2.0);
        sum += k[static_cast<std::size_t>(i)];
    }
    for (double& v : k) v /= sum;

    // Reflect WITHOUT repeating the edge sample, which is torch's 'reflect'
    // (numpy calls the same thing 'reflect'; torch's 'replicate' is the other).
    auto reflect = [](int i, int n) {
        if (n == 1) return 0;
        const int period = 2 * (n - 1);
        i = ((i % period) + period) % period;
        return i < n ? i : period - i;
    };

    std::vector<double> tmp(static_cast<std::size_t>(h) * w);
    for (int y = 0; y < h; ++y) {
        const double* row = src + static_cast<std::size_t>(y) * w;
        for (int x = 0; x < w; ++x) {
            double acc = 0.0;
            for (int t = 0; t < kernel_size; ++t) {
                acc += row[reflect(x + t - p, w)] * k[static_cast<std::size_t>(t)];
            }
            tmp[static_cast<std::size_t>(y) * w + x] = acc;
        }
    }
    out.assign(static_cast<std::size_t>(h) * w, 0.0);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            double acc = 0.0;
            for (int t = 0; t < kernel_size; ++t) {
                acc += tmp[static_cast<std::size_t>(reflect(y + t - p, h)) * w + x] *
                       k[static_cast<std::size_t>(t)];
            }
            out[static_cast<std::size_t>(y) * w + x] = acc;
        }
    }
}

void pad_linear_extrapolation(const double* src, int h, int w,
                              std::vector<double>& out) {
    const int ph = h + 2, pw = w + 2;
    out.assign(static_cast<std::size_t>(ph) * pw, 0.0);

    auto at = [&](int y, int x) -> double& {
        return out[static_cast<std::size_t>(y + 1) * pw + (x + 1)];
    };
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) at(y, x) = src[static_cast<std::size_t>(y) * w + x];
    }
    // Rows first, then columns over the already-padded rows — the order matters
    // for the corners, which upstream fills from the row-padded array.
    for (int x = 0; x < w; ++x) {
        at(-1, x) = h > 1 ? 2 * at(0, x) - at(1, x) : at(0, x);
        at(h, x)  = h > 1 ? 2 * at(h - 1, x) - at(h - 2, x) : at(h - 1, x);
    }
    for (int y = -1; y <= h; ++y) {
        at(y, -1) = w > 1 ? 2 * at(y, 0) - at(y, 1) : at(y, 0);
        at(y, w)  = w > 1 ? 2 * at(y, w - 1) - at(y, w - 2) : at(y, w - 1);
    }
}

void resize_extrapolated(const double* src, int h, int w,
                         int oh, int ow, std::vector<double>& out) {
    const double scale_h = static_cast<double>(oh) / h;
    const double scale_w = static_cast<double>(ow) / w;

    std::vector<double> padded;
    pad_linear_extrapolation(src, h, w, padded);

    const int nh = static_cast<int>(std::lround(oh + 2 * scale_h));
    const int nw = static_cast<int>(std::lround(ow + 2 * scale_w));

    std::vector<double> big;
    resize_bilinear(padded.data(), h + 2, w + 2, nh, nw, big);

    const int ph = static_cast<int>(std::lround(scale_h));
    const int pw = static_cast<int>(std::lround(scale_w));

    out.assign(static_cast<std::size_t>(oh) * ow, 0.0);
    for (int y = 0; y < oh; ++y) {
        const double* row = big.data() + static_cast<std::size_t>(y + ph) * nw + pw;
        std::copy(row, row + ow, out.begin() + static_cast<std::ptrdiff_t>(y) * ow);
    }
}

void laplacian_decode(const double* residual, int rh, int rw,
                      const double* lowres, int lh, int lw,
                      bool extrapolate, std::vector<double>& out) {
    std::vector<double> up;
    if (extrapolate) {
        resize_extrapolated(lowres, lh, lw, rh, rw, up);
    } else {
        resize_bilinear(lowres, lh, lw, rh, rw, up);
    }
    out.assign(static_cast<std::size_t>(rh) * rw, 0.0);
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = residual[i] + up[i];
}

void laplacian_denoise(const double* residual, int rh, int rw,
                       const double* lowres, int lh, int lw,
                       double sigma, std::vector<double>& new_lowres) {
    // Upstream decodes with extrapolation here even though the caller then
    // decodes WITHOUT it. Both are deliberate: the extrapolated decode is only
    // an intermediate used to re-derive the low band, and matching upstream
    // means matching that asymmetry rather than tidying it away.
    std::vector<double> decoded;
    laplacian_decode(residual, rh, rw, lowres, lh, lw, /*extrapolate=*/true, decoded);

    // laplacian_encode's low branch: downsample to the low resolution, blur.
    // `downsample_size` is passed as a single int upstream, which torchvision
    // reads as "shorter side", so a non-square low band would not round-trip.
    // Every call here is square.
    std::vector<double> small;
    resize_bilinear(decoded.data(), rh, rw, lh, lw, small);

    const int kernel_size = (static_cast<int>(sigma * 2) / 2) * 2 + 1;
    gaussian_blur(small.data(), lh, lw, kernel_size, sigma, new_lowres);
}

}  // namespace brodiffusion::terrain
