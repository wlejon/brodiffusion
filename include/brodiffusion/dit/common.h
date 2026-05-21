#pragma once

// Shared DiT building blocks.
//
// Bricks common to diffusion-transformer denoisers (Flux today; SD3 / Wan
// later): 2x2 latent patch packing, the 2D-axial RoPE cos/sin table builder,
// the joint full-attention helper (RoPE on Q & K then flash attention), and
// the AdaLN modulation-chunk slicer. Free functions / small structs — the
// concrete denoiser (dit::flux) composes them.
//
// Everything is at the pipeline compute dtype (FP32 on CPU, FP16 on a GPU
// backend) except the RoPE cos/sin tables, which are FP32 on every backend
// (brotensor::rope_apply requires FP32 tables).

#include "brotensor/tensor.h"

#include <vector>

namespace brodiffusion::dit {

// ─── 2x2 latent patch packing ──────────────────────────────────────────────
//
// Flux operates on 2x2-patch-packed tokens. The latent is a flat NCHW row
// (1, latent_channels*H_lat*W_lat): element [c, y, x] at index
// c*(H_lat*W_lat) + y*W_lat + x. With hp = H_lat/2, wp = W_lat/2 there are
// hp*wp image tokens, each a (latent_channels*4)-vector.
//
//   packed[i*wp + j][c*4 + dy*2 + dx] = latent[c, 2i+dy, 2j+dx]
//
// pack/unpack download to host, rearrange, and re-upload — correct, not the
// fastest path; a device kernel can come later. Both handle the FP16 and FP32
// compute-dtype cases.

// Pack a (1, latent_channels*H_lat*W_lat) latent into a
// (hp*wp, latent_channels*4) token tensor at the compute dtype. H_lat and
// W_lat must be even.
brotensor::Tensor pack_latents(const brotensor::Tensor& latent,
                               int latent_channels, int H_lat, int W_lat);

// Inverse of pack_latents. `packed` is (hp*wp, latent_channels*4); writes a
// (1, latent_channels*H_lat*W_lat) NCHW row into `out` at the compute dtype.
void unpack_latents(const brotensor::Tensor& packed,
                    int latent_channels, int H_lat, int W_lat,
                    brotensor::Tensor& out);

// ─── 2D axial RoPE tables ──────────────────────────────────────────────────

// cos / sin rotation tables for the joint [text ; image] sequence, shape
// (txt_len + img_len, head_dim/2), FP32. Text tokens (rows [0,txt_len)) get
// the identity rotation (cos=1, sin=0); image token at grid (i,j) — row
// txt_len + i*wp + j — gets axial position (0, i, j). axes_dims_rope assigns
// frequency pairs to the three position axes and must sum to head_dim.
struct RopeTables {
    brotensor::Tensor cos;  // (L, head_dim/2) FP32
    brotensor::Tensor sin;  // (L, head_dim/2) FP32
};
RopeTables build_axial_rope_tables(int txt_len, int hp, int wp, int head_dim,
                                   const std::vector<int>& axes_dims_rope);

// ─── joint attention ───────────────────────────────────────────────────────

// Full-bidirectional joint attention. Q/K/V are (L, D); applies the supplied
// cos/sin RoPE tables to Q and K, then runs flash attention (no mask, not
// causal). Writes (L, D) into `out`. Caller has already applied the per-head
// RMSNorm to Q and K.
void joint_attention(const brotensor::Tensor& Q,
                     const brotensor::Tensor& K,
                     const brotensor::Tensor& V,
                     const brotensor::Tensor& cos,
                     const brotensor::Tensor& sin,
                     int head_dim, int num_heads,
                     brotensor::Tensor& out,
                     // scratch reused across calls (rotated Q / K)
                     brotensor::Tensor& Qr, brotensor::Tensor& Kr);

// ─── AdaLN modulation chunks ───────────────────────────────────────────────

// Slice a (1, n_chunks*inner_dim) modulation row into `n_chunks` separate
// (1, inner_dim) tensors via copy_d2d. `chunks` is resized to n_chunks.
void slice_modulation_chunks(const brotensor::Tensor& row, int inner_dim,
                             int n_chunks,
                             std::vector<brotensor::Tensor>& chunks);

}  // namespace brodiffusion::dit
