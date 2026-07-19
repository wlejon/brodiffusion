// brodiffusion/terrain/laplacian.h — the Laplacian pyramid the decoder writes into.
//
// Port of xandergos/terrain-diffusion (MIT) —
// terrain_diffusion/data/laplacian_encoder.py.
//
// terrain-diffusion never predicts elevation directly. It predicts a HIGH-PASS
// RESIDUAL at native resolution (the decoder stage) and a LOW-FREQUENCY BAND at
// latent resolution (channel 4 of the latent map), and elevation is their sum
// once the low band is upsampled back. Splitting the bands is what lets a 28M
// decoder produce 512x512 tiles that still agree on continent-scale shape: the
// shape lives in the low band, which came from a much wider context.
//
// ── Why these functions are so particular ──────────────────────────────────
//
// Everything here has to match torchvision, not merely resemble it, because the
// checkpoint was trained against torchvision's exact resampling. Two places
// where the obvious implementation is the wrong one:
//
//   * `TF.resize(..., BILINEAR)` DEFAULTS TO antialias=True (torchvision >= 0.17,
//     and 0.24 is what upstream runs). When minifying, that is not bilinear at
//     all — it is a triangle filter whose support stretches with the scale
//     factor, i.e. PIL's resampling. Plain `F.interpolate(mode='bilinear')`
//     disagrees materially: on an 8x8 -> 4x4 test the first output value is
//     6.43 antialiased against 4.50 not. Verified against torchvision 0.24.1.
//
//   * The same triangle filter with support = max(1, scale) reduces EXACTLY to
//     align-corners-false bilinear when magnifying, so one implementation covers
//     both directions. That is checked, not assumed: `resize_bilinear` matches
//     torchvision bit-for-bit in double precision in both directions.
//
// `gaussian_blur` likewise matches torchvision's: a normalised exp(-(x/sigma)^2/2)
// kernel applied separably with REFLECT padding (which does not repeat the edge
// sample).
//
// All of this is host-side double precision. It runs once per requested region
// rather than per tile, and the arrays are small next to a UNet evaluation, so
// there is nothing to gain from the GPU and something to lose in reproducibility.

#pragma once

#include <cstddef>
#include <vector>

namespace brodiffusion::terrain {

// Separable triangle-filter resize, matching torchvision's
// `TF.resize(..., BILINEAR, antialias=True)` in both directions. Row-major,
// single channel. `out` is resized to oh*ow.
void resize_bilinear(const double* src, int ih, int iw,
                     int oh, int ow, std::vector<double>& out);

// Separable Gaussian blur with reflect padding, matching
// `TF.gaussian_blur(kernel_size, sigma)`. Row-major, single channel.
void gaussian_blur(const double* src, int h, int w,
                   int kernel_size, double sigma, std::vector<double>& out);

// Pad by one sample on every side by LINEAR EXTRAPOLATION (2*edge - next),
// not by clamping or reflecting. Upstream uses this before an upsample so the
// interpolated band keeps its slope through the boundary instead of flattening
// against it. Output is (h+2) x (w+2).
void pad_linear_extrapolation(const double* src, int h, int w,
                              std::vector<double>& out);

// Upsample with the boundary slope preserved: extrapolate by one sample, resize
// to a correspondingly larger target, then crop the padding back off.
void resize_extrapolated(const double* src, int h, int w,
                         int oh, int ow, std::vector<double>& out);

// residual + upsample(lowres). `extrapolate` selects resize_extrapolated over
// plain resize for the upsample. Output is rh*rw, the residual's own size.
void laplacian_decode(const double* residual, int rh, int rw,
                      const double* lowres, int lh, int lw,
                      bool extrapolate, std::vector<double>& out);

// Re-derive the low band so it is consistent with the residual it will be added
// to: decode, downsample back to the low resolution, blur. Returns the new low
// band (lh*lw); the residual is unchanged, which is why it is not returned.
//
// Upstream calls this "denoise" — the point is that the decoder's residual and
// the latent map's low band were produced by different networks and need not
// agree about the low frequencies. Rebuilding the low band from their sum makes
// the split consistent, so the seam does not show up as banding.
void laplacian_denoise(const double* residual, int rh, int rw,
                       const double* lowres, int lh, int lw,
                       double sigma, std::vector<double>& new_lowres);

}  // namespace brodiffusion::terrain
