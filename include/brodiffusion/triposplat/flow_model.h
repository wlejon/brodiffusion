#pragma once
//
// TripoSplat — flow-matching DiT (LatentSeqMMFlowModel).
//
// The generative core of TripoSplat (VAST-AI/TripoSplat). A multi-modal
// flow-matching transformer that, conditioned on two image features
// (feature1 = DINOv3 ViT-H, feature2 = Flux.2 VAE; brovisionml + the encoder in
// this repo), predicts the flow velocity that transports a noised latent token
// set toward the clean latent the octree Gaussian decoder consumes.
//
// Architecture (reference: model.py `LatentSeqMMFlowModel`):
//   input_layer(z) + pos_embedder(sobol_pe)
//   t_emb = t_embedder(t); t_mod = adaLN_modulation(t_emb)          [share_mod]
//   cond = cond_embedder(feature1) + cond_embedder2(feature2)
//   2x noise_refiner   (modulated, content-rotary noise_repo[i])     on latents
//   2x context_refiner (un-modulated, content-rotary context_repo[i]) on cond
//   cam = cam_refiner(camera)                                        [MLP 5->1024]
//   h = concat([latents, cond, cam]);   24x block (modulated, repo[i])
//   final LN; shift_table + t_emb gates h;  out_layer / cam_out_layer
//
// Every transformer block uses per-head content-dependent 3D rotary
// (RePo3DRotaryEmbedding -> brotensor::rope_apply_perhead) and per-head
// RMSNorm on Q/K (MultiHeadRMSNorm = l2_norm + per-head gamma). Runs at the
// pipeline compute dtype (FP16 on GPU); the rotary cos/sin tables are built in
// FP32 (matching the reference, which forces FP32 for the rotary).

#include "brotensor/tensor.h"

#include <string>
#include <vector>

namespace brotensor::safetensors { class File; }

namespace brodiffusion::triposplat {

// The baked Sobol positional table: SobolEngine(3, scramble=True, seed=123)
// .draw(8192), row-major (8192, 3). Defined in flow_sobol_pe.cpp. torch's
// scrambled-Sobol RNG is not reproducible in C++, so the table is frozen as a
// constant; it is effectively a model weight.
extern const float kFlowSobolPosPE[24576];

// Fixed configuration (FLOW_MODEL_ARGS). All TripoSplat checkpoints share it.
struct FlowModelConfig {
    int q_token_length   = 8192;
    int in_channels      = 16;
    int out_channels     = 16;
    int cam_channels     = 5;
    int model_channels   = 1024;
    int cond_channels    = 1280;
    int cond2_channels   = 128;
    int num_refiner_blocks = 2;
    int num_blocks       = 24;
    int num_heads        = 16;
    int head_dim         = 64;   // num_head_channels
    int mlp_ratio        = 4;
};

class FlowDiT {
public:
    FlowDiT();
    explicit FlowDiT(const FlowModelConfig& cfg);
    ~FlowDiT();

    FlowDiT(const FlowDiT&) = delete;
    FlowDiT& operator=(const FlowDiT&) = delete;
    FlowDiT(FlowDiT&&) noexcept = default;
    FlowDiT& operator=(FlowDiT&&) noexcept = default;

    // Load all weights from a TripoSplat flow-model safetensors file (strict —
    // every tensor in the reference state_dict must be present). Throws on a
    // missing tensor or shape mismatch.
    void load_weights(const brotensor::safetensors::File& f);

    // Predict the flow velocity for one sampler step.
    //   latent:  (L, in_channels)  noised latent tokens (L = q_token_length).
    //   camera:  (1, cam_channels) camera token.
    //   feature1:(K, cond_channels)   DINOv3 conditioning.
    //   feature2:(K, cond2_channels)  Flux.2 VAE conditioning (same K as feat1).
    //   t:       scalar flow time.
    //   out_latent: (L, out_channels) predicted latent velocity.
    //   out_camera: (1, cam_channels) predicted camera velocity.
    // All tensors at the pipeline compute dtype. Caller must sync_all() before
    // reading the outputs.
    void forward(const brotensor::Tensor& latent,
                 const brotensor::Tensor& camera,
                 const brotensor::Tensor& feature1,
                 const brotensor::Tensor& feature2,
                 float t,
                 brotensor::Tensor& out_latent,
                 brotensor::Tensor& out_camera);

    // ── CUDA-graph step-capture seam ───────────────────────────────────────
    //
    // forward(...) == prepare_step(t) + forward_body(...). The split lets the
    // sampler record forward_body with brotensor::CudaGraphCapture and replay
    // it every step after refreshing the inputs in place and re-running
    // prepare_step.
    //
    // prepare_step(t) — the t-dependent head: builds the host timestep
    // sinusoid, uploads it, and runs the t_embedder MLP + adaLN chain into the
    // persistent t_emb_/t_mod_ members the captured body reads. Host-dependent,
    // so it always runs eagerly, OUTSIDE any graph capture.
    void prepare_step(float t);

    // forward_body — everything after the time embedding (input_layer through
    // the out layers), reading t_emb_/t_mod_. Capture contract: after warm-up
    // calls at fixed shapes, a call performs no Tensor (re)allocation (every
    // intermediate lives in a member scratch buffer settled at its high-water
    // capacity), no host reads/writes, and launches every kernel on
    // brotensor's current stream — so a sampler may record it with
    // brotensor::CudaGraphCapture and replay it each step after refreshing the
    // input buffers in place and calling prepare_step.
    void forward_body(const brotensor::Tensor& latent,
                      const brotensor::Tensor& camera,
                      const brotensor::Tensor& feature1,
                      const brotensor::Tensor& feature2,
                      brotensor::Tensor& out_latent,
                      brotensor::Tensor& out_camera);

    const FlowModelConfig& config() const { return cfg_; }

private:
    struct Linear { brotensor::Tensor W, b; };

    // RePo3DRotaryEmbedding: content-dependent per-head 3D rotary. Produces the
    // FP32 cos/sin tables fed to rope_apply_perhead, recomputed each block from
    // the block's own hidden state.
    struct Repo {
        brotensor::Tensor norm_g, norm_b;       // LayerNorm32(model_channels)
        Linear gate_map, content_map;           // bias-free (1024->128)
        Linear final_map;                        // bias-free (128->3*num_heads)
        std::vector<float> freqs0, freqs1, freqs2;  // host fp32, per-axis (CPU path)
        brotensor::Tensor  freqs_pi;                // (1, half) device, [f0|f1|f2]×π
    };

    // UnifiedTransformerBlock. Modulated blocks (noise_refiner / blocks) use a
    // shared t_mod + per-block shift_table and affine-free norms; the
    // context_refiner is un-modulated with affine norm1/norm2. The fused qkv
    // weight is split into separate q/k/v linears at load (contiguous row
    // ranges) so attention is three plain linears, no column slicing.
    struct Block {
        bool modulated = true;
        Linear q, k, v, out;                     // attention
        brotensor::Tensor q_gamma, k_gamma;      // MultiHeadRMSNorm, (1,D), *sqrt(hd)
        Linear ff0, ff2;                         // mlp
        brotensor::Tensor shift_table;           // modulated-only, (1, 6*D)
        brotensor::Tensor norm1_g, norm1_b, norm2_g, norm2_b;  // un-modulated affine
    };

    void load_block(const brotensor::safetensors::File& f,
                    const std::string& prefix, bool modulated, Block& blk);
    void load_repo(const brotensor::safetensors::File& f,
                   const std::string& prefix, Repo& repo);

    // Build the (L*num_heads, head_dim/2) FP32 cos/sin tables from `hidden`.
    void build_rope(const brotensor::Tensor& hidden, const Repo& repo,
                    brotensor::Tensor& cos_out, brotensor::Tensor& sin_out);

    // Self-attention with per-head rotary + per-head Q/K RMSNorm. Reads x (L,D),
    // writes (L,D) into out.
    void attention(const brotensor::Tensor& x, const Block& blk,
                   const brotensor::Tensor& cos, const brotensor::Tensor& sin,
                   brotensor::Tensor& out);

    // One transformer block in place on x (L,D). t_mod is the shared modulation
    // row (1, 6*D) for modulated blocks (ignored otherwise).
    void run_block(brotensor::Tensor& x, const Block& blk,
                   const brotensor::Tensor* t_mod,
                   const brotensor::Tensor& cos, const brotensor::Tensor& sin);

    FlowModelConfig cfg_;

    Linear t_emb0_, t_emb1_;          // TimestepEmbedder.mlp.{0,2}
    Linear adaLN_;                    // share_mod adaLN_modulation.1
    Linear input_layer_, cond_embedder_, cond_embedder2_;
    Linear cam0_, cam2_;              // cam_refiner MLP.{0,2}
    Linear out_layer_, cam_out_layer_;
    brotensor::Tensor shift_table_;   // model-level (1, 2*D)

    std::vector<Repo> noise_repo_, context_repo_, repo_;
    std::vector<Block> noise_refiner_, context_refiner_, blocks_;

    brotensor::Tensor pos_emb_;       // precomputed (L, D) absolute position emb
    brotensor::Tensor ada_gamma_, ada_beta_;  // ones / zeros for affine-free LN

    // ── step-capture scratch ────────────────────────────────────────────────
    // Every intermediate in the prepare_step/forward_body path lives in a
    // member buffer so the body is allocation-stable after warm-up (capacity-
    // aware Tensor::resize keeps the device pointer once a buffer has reached
    // its high-water size). Shapes vary across call sites (L latent rows, K
    // cond rows, L+K+1 joint rows); each buffer settles at its high-water
    // capacity during the eager warm-up steps.
    brotensor::Tensor t_emb_, t_mod_;         // (1,D) / (1,6D), read by the body
    brotensor::Tensor t_mlp_, t_silu_;        // t_embedder hidden / silu(t_emb)
    brotensor::Tensor h_x_, h_cond_, c2_;     // input/cond embeddings
    brotensor::Tensor h_cam_, cam_mid_;       // cam_refiner MLP
    brotensor::Tensor h_cat_;                 // concat([h_x, h_cond, h_cam])
    brotensor::Tensor cos_, sin_;             // rope tables, reused per block
    brotensor::Tensor hx_, hcam_;             // final LN outputs
    brotensor::Tensor hx_slice_, cam_slice_;  // final LN input slices
    brotensor::Tensor shift_, scale_;         // shift_table + t_emb gates
    brotensor::Tensor hx_mod_, hcam_mod_;     // modulated final activations
    brotensor::Tensor q_, k_, v_, qr_, kr_, fa_;  // attention()
    brotensor::Tensor blk_ln_, blk_h_, blk_attn_; // run_block()
    brotensor::Tensor blk_ff_mid_, blk_ff_out_, blk_gated_, blk_mod_;
    std::vector<brotensor::Tensor> blk_ch_;   // six (1,D) modulation chunks
    brotensor::Tensor rope_h_, rope_g_, rope_c_, rope_dp_;  // build_rope()

    bool loaded_ = false;
};

}  // namespace brodiffusion::triposplat
