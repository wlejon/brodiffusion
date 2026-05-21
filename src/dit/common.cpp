#include "brodiffusion/dit/common.h"

#include "brodiffusion/detail/compute.h"
#include "brodiffusion/detail/device.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace brodiffusion::dit {

namespace bt = ::brotensor;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("dit::common: " + msg);
}

// Download a compute-dtype tensor into a host FP32 vector, handling both the
// FP16 (GPU) and FP32 (CPU) cases — same pattern as pipeline.cpp's decode().
std::vector<float> download_fp32(const bt::Tensor& t) {
    const std::size_t n = static_cast<std::size_t>(t.size());
    if (t.dtype == bt::Dtype::FP16) {
        std::vector<std::uint16_t> bits(n);
        t.copy_to_host_fp16(bits.data());
        std::vector<float> out(n);
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = bt::fp16_bits_to_fp32(bits[i]);
        }
        return out;
    }
    return t.to_host_vector();
}

}  // namespace

// ─── 2x2 latent patch packing ──────────────────────────────────────────────

bt::Tensor pack_latents(const bt::Tensor& latent, int latent_channels,
                        int H_lat, int W_lat) {
    if (H_lat % 2 != 0 || W_lat % 2 != 0) {
        fail("pack_latents: H_lat and W_lat must be even");
    }
    const int hp = H_lat / 2;
    const int wp = W_lat / 2;
    const int img_len = hp * wp;
    const int patch = latent_channels * 4;
    const std::size_t expect =
        static_cast<std::size_t>(latent_channels) * H_lat * W_lat;
    if (static_cast<std::size_t>(latent.size()) != expect) {
        fail("pack_latents: latent has unexpected element count");
    }

    bt::sync_all();
    std::vector<float> src = download_fp32(latent);

    std::vector<float> packed(static_cast<std::size_t>(img_len) * patch);
    const std::size_t plane = static_cast<std::size_t>(H_lat) * W_lat;
    for (int i = 0; i < hp; ++i) {
        for (int j = 0; j < wp; ++j) {
            const std::size_t tok =
                static_cast<std::size_t>(i) * wp + j;
            for (int c = 0; c < latent_channels; ++c) {
                for (int dy = 0; dy < 2; ++dy) {
                    for (int dx = 0; dx < 2; ++dx) {
                        const std::size_t si =
                            static_cast<std::size_t>(c) * plane +
                            static_cast<std::size_t>(2 * i + dy) * W_lat +
                            static_cast<std::size_t>(2 * j + dx);
                        const std::size_t di =
                            tok * patch +
                            static_cast<std::size_t>(c) * 4 + dy * 2 + dx;
                        packed[di] = src[si];
                    }
                }
            }
        }
    }
    return detail::upload_host(packed.data(), img_len, patch);
}

void unpack_latents(const bt::Tensor& packed, int latent_channels,
                    int H_lat, int W_lat, bt::Tensor& out) {
    if (H_lat % 2 != 0 || W_lat % 2 != 0) {
        fail("unpack_latents: H_lat and W_lat must be even");
    }
    const int hp = H_lat / 2;
    const int wp = W_lat / 2;
    const int img_len = hp * wp;
    const int patch = latent_channels * 4;
    if (packed.rows != img_len || packed.cols != patch) {
        fail("unpack_latents: packed has unexpected shape");
    }

    bt::sync_all();
    std::vector<float> src = download_fp32(packed);

    const std::size_t plane = static_cast<std::size_t>(H_lat) * W_lat;
    std::vector<float> latent(
        static_cast<std::size_t>(latent_channels) * plane);
    for (int i = 0; i < hp; ++i) {
        for (int j = 0; j < wp; ++j) {
            const std::size_t tok =
                static_cast<std::size_t>(i) * wp + j;
            for (int c = 0; c < latent_channels; ++c) {
                for (int dy = 0; dy < 2; ++dy) {
                    for (int dx = 0; dx < 2; ++dx) {
                        const std::size_t si =
                            tok * patch +
                            static_cast<std::size_t>(c) * 4 + dy * 2 + dx;
                        const std::size_t di =
                            static_cast<std::size_t>(c) * plane +
                            static_cast<std::size_t>(2 * i + dy) * W_lat +
                            static_cast<std::size_t>(2 * j + dx);
                        latent[di] = src[si];
                    }
                }
            }
        }
    }
    bt::Tensor t = detail::upload_host(
        latent.data(), 1,
        static_cast<int>(latent.size()));
    out = t;
}

// ─── 2D axial RoPE tables ──────────────────────────────────────────────────

RopeTables build_axial_rope_tables(int txt_len, int hp, int wp, int head_dim,
                                   const std::vector<int>& axes_dims_rope) {
    if (head_dim % 2 != 0) fail("build_axial_rope_tables: head_dim must be even");
    int sum = 0;
    for (int d : axes_dims_rope) sum += d;
    if (sum != head_dim) {
        fail("build_axial_rope_tables: axes_dims_rope must sum to head_dim");
    }
    const int half = head_dim / 2;          // total pair count
    const int img_len = hp * wp;
    const int L = txt_len + img_len;

    // Per-axis pair offsets / counts. axis a owns pairs
    // [pair_off[a], pair_off[a] + axes_dims_rope[a]/2).
    const int n_axes = static_cast<int>(axes_dims_rope.size());
    std::vector<int> pair_off(static_cast<std::size_t>(n_axes));
    std::vector<int> pair_cnt(static_cast<std::size_t>(n_axes));
    {
        int off = 0;
        for (int a = 0; a < n_axes; ++a) {
            pair_off[static_cast<std::size_t>(a)] = off;
            pair_cnt[static_cast<std::size_t>(a)] = axes_dims_rope[a] / 2;
            off += pair_cnt[static_cast<std::size_t>(a)];
        }
    }

    std::vector<float> cos(static_cast<std::size_t>(L) * half, 1.0f);
    std::vector<float> sin(static_cast<std::size_t>(L) * half, 0.0f);

    // Fill one (row, pair) cell for axis `a`, axis position `pos`.
    auto fill_axis = [&](std::size_t row, int a, int pos) {
        const int D_a = axes_dims_rope[a];
        const int off = pair_off[static_cast<std::size_t>(a)];
        const int cnt = pair_cnt[static_cast<std::size_t>(a)];
        for (int p = 0; p < cnt; ++p) {
            const double omega =
                1.0 / std::pow(10000.0, (2.0 * p) / static_cast<double>(D_a));
            const double angle = static_cast<double>(pos) * omega;
            const std::size_t idx =
                row * half + static_cast<std::size_t>(off + p);
            cos[idx] = static_cast<float>(std::cos(angle));
            sin[idx] = static_cast<float>(std::sin(angle));
        }
    };

    // Text tokens: position (0,0,0) → identity (already initialized).
    // Image tokens: grid (i,j) → axial position (0, i, j).
    for (int i = 0; i < hp; ++i) {
        for (int j = 0; j < wp; ++j) {
            const std::size_t row =
                static_cast<std::size_t>(txt_len) +
                static_cast<std::size_t>(i) * wp + j;
            const int pos[3] = {0, i, j};
            for (int a = 0; a < n_axes && a < 3; ++a) {
                fill_axis(row, a, pos[a]);
            }
        }
    }

    RopeTables out;
    // rope_apply requires FP32 cos/sin tables on every backend.
    out.cos = bt::Tensor::from_host(cos.data(), L, half)
                  .to(bt::default_device());
    out.sin = bt::Tensor::from_host(sin.data(), L, half)
                  .to(bt::default_device());
    return out;
}

// ─── joint attention ───────────────────────────────────────────────────────

void joint_attention(const bt::Tensor& Q, const bt::Tensor& K,
                     const bt::Tensor& V, const bt::Tensor& cos,
                     const bt::Tensor& sin, int head_dim, int num_heads,
                     bt::Tensor& out, bt::Tensor& Qr, bt::Tensor& Kr) {
    bt::rope_apply(Q, cos, sin, head_dim, num_heads, Qr);
    bt::rope_apply(K, cos, sin, head_dim, num_heads, Kr);
    bt::flash_attention_forward(Qr, Kr, V, /*d_mask=*/nullptr, num_heads,
                                /*causal=*/false, out);
}

// ─── AdaLN modulation chunks ───────────────────────────────────────────────

void slice_modulation_chunks(const bt::Tensor& row, int inner_dim,
                             int n_chunks,
                             std::vector<bt::Tensor>& chunks) {
    if (row.cols != inner_dim * n_chunks) {
        fail("slice_modulation_chunks: row width mismatch");
    }
    chunks.resize(static_cast<std::size_t>(n_chunks));
    for (int k = 0; k < n_chunks; ++k) {
        bt::Tensor& dst = chunks[static_cast<std::size_t>(k)];
        detail::resize_like(dst, 1, inner_dim, row.dtype, row.device);
        bt::copy_d2d(row, k * inner_dim, dst, 0, inner_dim);
    }
}

}  // namespace brodiffusion::dit
