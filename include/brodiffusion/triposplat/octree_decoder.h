#pragma once
//
// TripoSplat — OctreeGaussianDecoder (latent -> 3D Gaussian splats).
//
// The decoder half of TripoSplat's generative core (VAST-AI/TripoSplat). It
// turns the flow DiT's clean latent token set into an explicit 3D Gaussian
// cloud through two point-conditioned transformers that both cross-attend to
// the latent:
//
//   octree  OctreeProbabilityFixedlenDecoder — an autoregressive octree
//           structure predictor. Starting from the root voxel it descends
//           `max_level` levels; at each level it predicts, per active parent
//           voxel, an 8-way occupancy distribution over child octants, and a
//           systematic resampler distributes the parent's point budget among
//           the children. After the final level every surviving voxel is
//           expanded (with per-point jitter) into the requested point count.
//   gs      ElasticGaussianFixedlenDecoder — for each sampled point it emits
//           `num_gaussians` Gaussians (an "elastic" cluster): per-Gaussian
//           position offset, SH DC color, scale, rotation and opacity.
//
// Both transformers are composed from the same brotensor primitives the flow
// DiT uses (batched linear, per-head Q/K RMSNorm = l2_norm + broadcast_mul,
// flash attention, adaLN modulate, gelu-tanh FFN). Neither uses rotary — the
// point positions enter through an absolute sin/cos position embedding
// (PcdAbsolutePositionEmbedderV2), built host-side in FP32 (its frequencies
// reach 2^10 and the points originate host-side in the octree loop anyway).
//
// The octree forward and the GS forward are deterministic and golden-matched
// against the upstream reference; only the resampler is stochastic (seeded
// here for reproducibility).

#include "brotensor/tensor.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace brotensor::safetensors { class File; }

namespace brodiffusion::triposplat {

// Render-ready Gaussian-splat cloud (SoA), the decoder's output. The field
// layout mirrors bromesh::GaussianSplatCloud exactly — positions/scales (linear
// std-dev)/rotations (unit quaternion xyzw)/opacities ([0,1])/SH coefficients —
// so the bro JS-binding layer (which links both brodiffusion and bromesh)
// adapts it with a field-for-field copy. brodiffusion deliberately does not link
// bromesh (a heavy mesh library) just to share this POD; the renderer and the
// container live in their place, the decoder lives in its, bro composites.
//
// SH memory layout (`sh`): interleaved by coefficient, RGB together —
//   [r0 g0 b0  r1 g1 b1  ...]  per splat, coefficient 0 the DC term.
// For TripoSplat shDegree is always 0, so stride is 3 (one DC RGB triple).
struct GaussianSplats {
    std::vector<float> positions;  // xyz,  stride 3
    std::vector<float> scales;     // xyz,  stride 3 (linear std-dev)
    std::vector<float> rotations;  // xyzw, stride 4 (unit quaternion)
    std::vector<float> opacities;  // a,    stride 1 ([0,1])
    std::vector<float> sh;         // stride 3*(shDegree+1)^2
    int shDegree = 0;

    std::size_t count() const { return positions.size() / 3; }
    int shStride() const { return 3 * (shDegree + 1) * (shDegree + 1); }
    bool empty() const { return positions.empty(); }
};

// Fixed configuration (OCTREE_DECODER_ARGS + GS_DECODER_ARGS). All TripoSplat
// checkpoints share it.
struct OctreeDecoderConfig {
    int model_channels   = 1024;
    int cond_channels    = 16;    // latent token dim (ctx of every cross-attn)
    int octree_num_blocks = 4;
    int gs_num_blocks    = 16;
    int num_heads        = 16;
    int head_dim         = 64;    // num_head_channels
    int mlp_ratio        = 4;
    int num_gaussians    = 32;    // Gaussians emitted per sampled point
    int max_level        = 8;     // _MAX_VOXEL_LEVEL
    int gs_out_channels  = 480;   // num_gaussians * (3+3+3+4+1+1)
};

class OctreeGaussianDecoder {
public:
    OctreeGaussianDecoder();
    explicit OctreeGaussianDecoder(const OctreeDecoderConfig& cfg);
    ~OctreeGaussianDecoder();

    OctreeGaussianDecoder(const OctreeGaussianDecoder&) = delete;
    OctreeGaussianDecoder& operator=(const OctreeGaussianDecoder&) = delete;
    OctreeGaussianDecoder(OctreeGaussianDecoder&&) noexcept = default;
    OctreeGaussianDecoder& operator=(OctreeGaussianDecoder&&) noexcept = default;

    // Load all weights from a TripoSplat decoder safetensors file (strict —
    // octree.* and gs.* prefixes). Throws on a missing tensor / shape mismatch.
    void load_weights(const brotensor::safetensors::File& f);

    int gaussians_per_point() const { return cfg_.num_gaussians; }

    // ── deterministic forwards (exposed for parity + composition) ───────────

    // Octree occupancy logits. `coords_host` is the row-major (N,3) FP32 parent
    // voxel centers in [0,1]; `res` the integer child resolution (1<<level);
    // `cond` the latent (Kc, cond_channels) at the compute dtype. Writes the
    // (N, 8) logits (compute dtype) into `logits_out`. Caller syncs before read.
    void octree_logits(const std::vector<float>& coords_host, int n_coords,
                       int res, const brotensor::Tensor& cond,
                       brotensor::Tensor& logits_out);

    // Elastic-Gaussian head features. `points_host` is the row-major (M,3) FP32
    // sampled points in [0,1]; `cond` the latent (Kc, cond_channels). Writes the
    // (M, gs_out_channels) raw features (compute dtype) into `feats_out`.
    void gs_features(const std::vector<float>& points_host, int n_points,
                     const brotensor::Tensor& cond, brotensor::Tensor& feats_out);

    // Assemble the render-ready Gaussian cloud from the GS head features and the
    // points they were predicted for. `feats` is (M, gs_out_channels); produces
    // M * num_gaussians splats (positions, scales, rotations xyzw, opacities,
    // SH-DC colors), all activated exactly as the reference Gaussian wrapper.
    GaussianSplats build_gaussians(
        const std::vector<float>& points_host, int n_points,
        const brotensor::Tensor& feats);

    // ── full pipeline ────────────────────────────────────────────────────────

    // latent -> Gaussian cloud. `latent` is the flow DiT's clean latent
    // (Kc, cond_channels) at the compute dtype. `num_gaussians` is the target
    // splat count (rounded down to a multiple of num_gaussians per point; at
    // least one decoder token). `seed` drives the stochastic resampler.
    GaussianSplats decode(const brotensor::Tensor& latent,
                          int num_gaussians, std::uint64_t seed);

    const OctreeDecoderConfig& config() const { return cfg_; }

private:
    struct Linear { brotensor::Tensor W, b; };

    // Cross-attention to the latent: Q from x (D->D), K/V from the latent
    // context (cond_channels->D, the fused to_kv split into k/v at load),
    // per-head Q/K RMSNorm (gamma folded with sqrt(head_dim)), output proj.
    struct CrossAttn {
        Linear to_q;
        Linear to_k, to_v;                    // split from to_kv (2D, cond_ch)
        brotensor::Tensor q_gamma, k_gamma;   // (1, D), *sqrt(head_dim)
        Linear to_out;
    };

    // Self-attention: fused to_qkv split into q/k/v (D->D each), per-head Q/K
    // RMSNorm, output proj. No rotary.
    struct SelfAttn {
        Linear q, k, v;
        brotensor::Tensor q_gamma, k_gamma;
        Linear to_out;
    };

    // ModulatedTransformerCrossOnlyBlock (octree): affine-free norms, shared
    // adaLN modulation, cross-attn + gelu FFN.
    struct OctBlock {
        CrossAttn cross;
        Linear ff0, ff2;
    };

    // TransformerCrossBlock (gs): self-attn (norm1 affine-free), cross-attn
    // (norm2 affine), gelu FFN (norm3 affine-free).
    struct GsBlock {
        SelfAttn self_attn;
        CrossAttn cross;
        Linear ff0, ff2;
        brotensor::Tensor norm2_g, norm2_b;   // the one affine norm
    };

    void load_linear(const brotensor::safetensors::File& f, const std::string& key,
                     int rows, int cols, Linear& lin);
    void load_cross(const brotensor::safetensors::File& f, const std::string& prefix,
                    CrossAttn& a);
    void load_self(const brotensor::safetensors::File& f, const std::string& prefix,
                   SelfAttn& a);
    brotensor::Tensor load_gamma(const brotensor::safetensors::File& f,
                                 const std::string& key);

    // Absolute sin/cos position embedding (PcdAbsolutePositionEmbedderV2), built
    // host-side in FP32 from the (N,3) points, returned at the compute dtype.
    brotensor::Tensor pos_embed_v2(const std::vector<float>& pts, int n) const;

    // in_proj(points) + pos_embed_v2(points), then input_layer. Shared head of
    // both transformers. Returns (N, D) at the compute dtype.
    brotensor::Tensor embed_points(const std::vector<float>& pts, int n,
                                   const Linear& in_proj,
                                   const Linear& input_layer) const;

    // Cross-attention over the latent context. x (Lq,D), cond (Lk,cond_ch).
    void cross_attention(const brotensor::Tensor& x, const CrossAttn& a,
                         const brotensor::Tensor& cond, brotensor::Tensor& out);
    // Self-attention. x (L,D).
    void self_attention(const brotensor::Tensor& x, const SelfAttn& a,
                        brotensor::Tensor& out);

    OctreeDecoderConfig cfg_;

    // octree
    Linear oct_in_proj_, oct_input_layer_, oct_out_proj_;
    Linear oct_l_emb0_, oct_l_emb1_;   // LevelEmbedder.mlp.{0,2}
    Linear oct_adaLN_;                 // share_mod adaLN_modulation.1
    std::vector<OctBlock> oct_blocks_;

    // gs
    Linear gs_in_proj_, gs_input_layer_, gs_out_proj_;
    std::vector<GsBlock> gs_blocks_;

    // gs gaussian-assembly constants (host)
    float base_offset_scale_ = 0.0f;
    std::vector<float> perturb_;       // (num_gaussians, 3), atanh-transformed

    brotensor::Tensor ada_gamma_, ada_beta_;  // ones / zeros for affine-free LN

    bool loaded_ = false;
};

}  // namespace brodiffusion::triposplat
