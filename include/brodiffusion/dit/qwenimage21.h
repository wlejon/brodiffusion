#pragma once

// QwenImage21Transformer2DModel — Qwen-Image 2.1's single-stream image DiT.
//
// A forward-only port of diffusers' QwenImage21Transformer2DModel (7.1B).
// Runs FP32 on CPU and BF16 on CUDA (dit::flux_compute_dtype).
//
// Shape of the model, and how it differs from Krea 2 / Flux:
//
//   - ONE joint sequence. Text tokens come from a Qwen3-VL vision-language
//     encoder; each `<|image_pad|>` slot in that text stands for a 2x2 group
//     of latent tokens, so a condition image's tokens are SUBSTITUTED into
//     the text stream at those positions. The target image's tokens are
//     appended after the text. For plain text-to-image the sequence is just
//     `[text ; target]`.
//   - Latents are NOT 2x2-packed (patch_size == 1): the 64-channel latent is
//     flattened spatially, one token per latent pixel, `img_in` projecting
//     64 -> hidden.
//   - ONE shared modulation vector feeds every block — there is no per-block
//     modulation parameter. `modulation` = Linear(hidden -> 4*hidden) on
//     SiLU(temb), chunked as [mod1.scale, mod1.gate, mod2.scale, mod2.gate].
//     Each block is
//         x += tanh(gate1) * attn(LN(x) * (1 + scale1))
//         x += tanh(gate2) * mlp (LN(x) * (1 + scale2))
//     with a non-affine LayerNorm (eps 1e-6) and a SwiGLU MLP
//     `out(silu(gate_layer(x)) * proj(x))`.
//   - `causal_condition`: text and condition-image tokens are modulated from
//     t = 0, only the target image's tokens see the sampled timestep. That
//     makes every prefix activation timestep-INDEPENDENT, which is what the
//     prefix KV cache below exploits.
//   - Block-causal attention: key j is visible to query i iff `i >= j` OR
//     both sit in the same image block. Text is strictly causal; each image
//     block is internally bidirectional and sees everything before it.
//   - 3-axis (frame, height, width) RoPE, axes_dims_rope {16, 56, 56} over a
//     128-wide head. A shared running position advances one per text token;
//     an image block freezes the frame axis at the position the preceding
//     text reached and lays its tokens on a zero-centred h/w grid, then
//     advances the shared position by max(H, W).
//
// Prefix KV cache. Because the prefix is timestep-independent, the first
// denoising step ("extract") runs the whole joint sequence and stores every
// layer's post-RoPE prefix K/V; later steps ("cached") push ONLY the target
// image's rows through the network and attend against
// `[cached prefix K/V ; fresh target K/V]`. Target queries see the entire
// sequence under the block-causal rule, so the cached step is plain
// non-causal attention. At 1024x1024 that turns a 4200-token forward into a
// 4096-token one and drops the whole text half of every block's projections.
//
// Pairs with the FlowMatch (rectified-flow Euler) scheduler; the forward
// predicts the velocity for the target image's tokens
// (prediction_type Velocity).

#include "brodiffusion/denoiser.h"
#include "brotensor/tensor.h"

#include <functional>
#include <string>
#include <vector>

namespace brotensor::safetensors { class File; }

namespace brodiffusion::dit {

struct QwenImage21Config {
    int patch_size          = 1;     // 2.1 consumes latents unpatched
    int in_channels         = 64;
    int out_channels        = 64;
    int num_layers          = 32;
    int attention_head_dim  = 128;
    int num_attention_heads = 32;    // hidden = heads * head_dim = 4096
    int context_in_dim      = 4096;  // Qwen3-VL hidden width
    int mlp_ratio           = 3;     // mlp hidden = 3 * hidden
    std::vector<int> axes_dims_rope = {16, 56, 56};  // sums to head_dim
    float eps               = 1e-6f;
    bool  causal_condition  = true;
    // Not in config.json (the reference hardcodes both).
    float rope_theta        = 10000.0f;
    int   timestep_embed_dim = 256;

    // When true (and the default device is CUDA), load_weights() quantizes the
    // per-block linears — attn to_q/to_k/to_v/to_out and img_mlp
    // gate_layer/proj/out across all 32 blocks — to INT8 weight-only (W8A16,
    // per-output-row symmetric FP32 scales) as they stream in, so the BF16
    // copy never lands on the device. It matters here: the BF16 transformer is
    // ~14.2 GB and the Qwen3-VL 8B text encoder has to share a 24 GB card
    // (WDDM starts demoting allocations above ~21 GiB). INT8 is ~7.2 GB.
    // The small in/out layers (img_in, txt_in, the time embedder, modulation,
    // norm_out, proj_out) keep the compute dtype, as do the q/k RMSNorm gains.
    // Ignored with a warning on a non-CUDA backend (the fused dequant matmuls
    // are GPU-only).
    bool quantize_weights = false;

    int hidden_size() const { return num_attention_heads * attention_head_dim; }
    int mlp_hidden_size() const { return hidden_size() * mlp_ratio; }
    // patch_size == 1, so a latent pixel is a token: the DiT's in_channels IS
    // the VAE's latent channel count.
    int latent_channels() const { return in_channels; }
};

// One run of the joint sequence's PREFIX — everything before the target image.
// A text run consumes `n_tokens` rows from encode_text()'s output; an image run
// consumes `h*w` rows from the condition-latent tensor and carries its latent
// token grid so RoPE can lay it out. For plain text-to-image the prefix is a
// single Text segment; edit / reference-image modes interleave Text and Image
// runs in the order the vision-language encoder emitted them.
struct QwenImage21Segment {
    enum class Kind { Text, Image };
    Kind kind = Kind::Text;
    int n_tokens = 0;   // Text: token count. Image: must equal h*w.
    int h = 0;          // Image only: latent token rows
    int w = 0;          // Image only: latent token columns
};

// Per-layer post-RoPE K/V of the joint sequence's prefix, plus the layout it
// was extracted for. Owned by the caller (the Denoiser keeps one per CFG
// branch); reset() drops it so the next forward re-extracts.
class QwenImage21PrefixCache {
public:
    void reset() { k_.clear(); v_.clear(); prefix_len_ = 0; hp_ = wp_ = 0; }
    bool valid() const { return !k_.empty(); }
    int  prefix_len() const { return prefix_len_; }
    int  target_hp() const { return hp_; }
    int  target_wp() const { return wp_; }
    // Does this cache describe the same joint layout as (prefix_len, hp, wp)?
    bool matches(int prefix_len, int hp, int wp) const {
        return valid() && prefix_len_ == prefix_len && hp_ == hp && wp_ == wp;
    }

private:
    friend class QwenImage21Transformer2DModel;
    std::vector<brotensor::Tensor> k_;   // per layer: (prefix_len, heads*head_dim)
    std::vector<brotensor::Tensor> v_;
    int prefix_len_ = 0;
    int hp_ = 0, wp_ = 0;
};

class QwenImage21Transformer2DModel {
public:
    explicit QwenImage21Transformer2DModel(const QwenImage21Config& cfg);
    ~QwenImage21Transformer2DModel();

    QwenImage21Transformer2DModel(const QwenImage21Transformer2DModel&) = delete;
    QwenImage21Transformer2DModel& operator=(
        const QwenImage21Transformer2DModel&) = delete;

    // Load from a single file or a sharded set (first match wins across
    // shards; a name missing everywhere throws). Qwen-Image 2.1 ships the
    // transformer in 2 BF16 shards.
    void load_weights(const brotensor::safetensors::File& f,
                      const std::string& prefix = "");
    // should_cancel: optional cooperative-cancel hook, polled once per
    // transformer block during the load. Throws LoadCancelled when it returns
    // true. Empty = no cancellation.
    void load_weights(
        const std::vector<const brotensor::safetensors::File*>& shards,
        const std::string& prefix = "",
        const std::function<bool()>& should_cancel = {});

    // Run `txt_in` (QwenImage21TextProjection: zero-centered RMSNorm ->
    // in_layer -> GELU(tanh) -> out_layer) over the vision-language encoder's
    // hidden states. Timestep-independent, so it runs once per prompt rather
    // than once per denoise step.
    //   encoder_hidden_states: (L, context_in_dim), any dtype.
    //   txt_out: (L, hidden_size) at the compute dtype.
    // The caller passes only VALID text rows — right-padded prompt positions
    // must be dropped before this call (they would otherwise be attendable
    // keys, and the causal structure gives them no other exclusion).
    void encode_text(const brotensor::Tensor& encoder_hidden_states,
                     brotensor::Tensor& txt_out);

    // Text-to-image forward: the joint sequence is `[txt ; target]`.
    //   packed_latent: (hp*wp, in_channels) — the spatially flattened target
    //       latent (NOT 2x2 packed), FP32 or the compute dtype.
    //   hp, wp: target latent token grid (image_seq_len = hp*wp).
    //   txt: encode_text()'s output, (L, hidden_size) at the compute dtype.
    //   timestep: flow-matching time in [0,1].
    //   cache: null for a self-contained full forward; otherwise the prefix
    //       cache to extract into (first call) or decode from (later calls).
    //   out: (hp*wp, out_channels) velocity at the compute dtype.
    void forward(const brotensor::Tensor& packed_latent, int hp, int wp,
                 const brotensor::Tensor& txt, float timestep,
                 QwenImage21PrefixCache* cache, brotensor::Tensor& out);

    // General form: an arbitrary prefix of Text / Image runs followed by the
    // target image. `prefix` describes rows [0, prefix_len) of the joint
    // sequence; its Text runs consume `txt`'s rows in order and its Image runs
    // consume `cond_latents`' rows in order. `cond_latents` may be null when
    // the prefix holds no Image run. This is the seam edit mode plugs into —
    // the block-causal prefill, the RoPE layout and the cache all key off this
    // segment list, so nothing in the core changes when condition images
    // appear.
    void forward_joint(const brotensor::Tensor& packed_latent, int hp, int wp,
                       const brotensor::Tensor& txt,
                       const brotensor::Tensor* cond_latents,
                       const std::vector<QwenImage21Segment>& prefix,
                       float timestep, QwenImage21PrefixCache* cache,
                       brotensor::Tensor& out);

    const QwenImage21Config& config() const { return cfg_; }
    brotensor::Dtype compute_dtype() const;

private:
    // A bias-free linear. Every Linear in Qwen-Image 2.1 is bias-free; the
    // `b` slot is kept so the INT8 and dense paths share one shape. When the
    // layer was quantized at load (cfg.quantize_weights), W is empty and
    // W_int8 (out,in) + scales (out,1 FP32 per-row) carry the weight.
    struct Linear {
        brotensor::Tensor W;       // (out, in) at compute dtype; empty if quantized
        brotensor::Tensor W_int8;  // (out, in) INT8 when quantized
        brotensor::Tensor scales;  // (out, 1) FP32 per-row scales when quantized
        bool quantized() const { return W_int8.size() > 0; }
        int out_dim() const { return quantized() ? W_int8.rows : W.rows; }
        int in_dim()  const { return quantized() ? W_int8.cols : W.cols; }
    };

    struct Block {
        Linear to_q, to_k, to_v, to_out;
        brotensor::Tensor norm_q, norm_k;      // (head_dim, 1) FP32
        Linear mlp_gate, mlp_proj, mlp_out;    // img_mlp.{gate_layer,proj,out}
    };

    // The four (1, hidden) modulation rows a forward runs on, plus the
    // norm_out scale — computed once per forward and shared by every block.
    // `_t` is the sampled timestep's row, `_0` the t = 0 row that text and
    // condition-image tokens use under causal_condition.
    struct Modulation {
        brotensor::Tensor scale1_t, gate1_t, scale2_t, gate2_t;
        brotensor::Tensor scale1_0, gate1_0, scale2_0, gate2_0;
        brotensor::Tensor final_scale;   // norm_out, target rows (t)
    };

    void load_impl_(const std::vector<const brotensor::safetensors::File*>& shards,
                    const std::string& prefix,
                    const std::function<bool()>& should_cancel);

    // One linear at the compute dtype, dispatching dense vs INT8 (W8A16).
    brotensor::Tensor lin_(const Linear& l, const brotensor::Tensor& X);

    // Non-affine LayerNorm (eps = cfg_.eps) over (L, hidden).
    void layernorm_(const brotensor::Tensor& X, brotensor::Tensor& Y);

    // Timestep embedding + shared modulation chunks, for flow time `timestep`.
    void build_modulation_(float timestep, Modulation& m);

    // cos/sin RoPE tables for the joint sequence described by `prefix` +
    // the target (hp, wp). Both (L, head_dim/2) FP32 on the default device.
    void build_rope_tables_(const std::vector<QwenImage21Segment>& prefix,
                            int hp, int wp, brotensor::Tensor& cos_t,
                            brotensor::Tensor& sin_t);

    QwenImage21Config cfg_;
    bool loaded_ = false;

    Linear img_in_;                 // (hidden, in_channels)
    Linear time_l1_, time_l2_;      // time_text_embed.timestep_embedder.linear_{1,2}
    Linear modulation_;             // modulation.1: (4*hidden, hidden)
    brotensor::Tensor txt_norm_;    // txt_in.text_norm: (context_in_dim,1) FP32 (1+w)
    Linear txt_in_, txt_out_;       // txt_in.{in_layer,out_layer}
    std::vector<Block> blocks_;
    Linear norm_out_lin_;           // norm_out.linear: (hidden, hidden)
    Linear proj_out_;               // (out_channels, hidden)

    // LayerNorm gain/bias (ones / zeros at the compute dtype) — Qwen-Image
    // 2.1's block norms are non-affine, and brotensor's batched LayerNorm
    // takes explicit gamma/beta at the activation dtype.
    brotensor::Tensor ln_gamma_, ln_beta_;
    brotensor::Tensor zero_shift_;   // (1, hidden) zeros — modulate()'s shift

    // Reused per-forward scratch (see the cache note in the header comment:
    // the layer caches hold the PREFIX only, and these hold the
    // [prefix ; target] concatenation a cached step attends against).
    brotensor::Tensor k_full_, v_full_;
};

// QwenImage21Denoiser — QwenImage21Transformer2DModel behind brodiffusion's
// model-agnostic Denoiser interface.
//
// Latent layout: the Pipeline hands the flat NCHW row
// (1, 64*H_lat*W_lat); this class transposes it to the (H_lat*W_lat, 64)
// token tensor the transformer consumes and transposes the velocity back.
// There is no 2x2 patching — patch_size is 1.
//
// Prefix caching. prepare() runs encode_text once per CFG branch and creates
// that branch's (empty) prefix cache. The first forward() after prepare —
// and any forward whose latent grid changed — extracts the cache; every
// later forward decodes from it. Branch::Cond and Branch::Uncond keep
// separate caches because their prompts differ.
//
// CFG convention: uses_cfg() is true, but the reference pipeline's
// `true_cfg_scale` defaults to 1.0 (no CFG). Qwen-Image 2.1 is not a
// distilled/guidance-embedded model — pass --guidance-scale 1.0 for the
// reference default and > 1 to actually run both branches.
class QwenImage21Denoiser final : public Denoiser {
public:
    explicit QwenImage21Denoiser(const QwenImage21Config& cfg);
    ~QwenImage21Denoiser();

    QwenImage21Denoiser(const QwenImage21Denoiser&) = delete;
    QwenImage21Denoiser& operator=(const QwenImage21Denoiser&) = delete;

    // ── Denoiser interface ────────────────────────────────────────────────
    void load_weights(const brotensor::safetensors::File& f,
                      const std::string& prefix = "") override;
    // Sharded overload — the 2.1 transformer ships in 2 shards. should_cancel
    // is a cooperative-cancel hook forwarded to the model (polled per block).
    void load_weights(
        const std::vector<const brotensor::safetensors::File*>& shards,
        const std::string& prefix = "",
        const std::function<bool()>& should_cancel = {});
    void finalize_weights() override {}   // quantisation happens at load
    PreparedConditioning prepare(const Conditioning& cond) override;
    void forward(const brotensor::Tensor& latent, int H_lat, int W_lat,
                 float timestep, const PreparedConditioning& prepared,
                 Branch branch, brotensor::Tensor& out) override;
    int latent_channels() const override {
        return model_.config().latent_channels();
    }
    PredictionType prediction_type() const override {
        return PredictionType::Velocity;
    }
    bool uses_cfg() const override { return true; }
    brotensor::Dtype compute_dtype() const override {
        return model_.compute_dtype();
    }

    const QwenImage21Config& config() const { return model_.config(); }
    QwenImage21Transformer2DModel&       model()       { return model_; }
    const QwenImage21Transformer2DModel& model() const { return model_; }

    // Drop both branches' prefix caches on `prepared`, forcing the next
    // forward to re-extract. The Denoiser does this automatically when the
    // latent grid changes; call it after mutating a prepared conditioning's
    // text rows out of band.
    void reset_cache(PreparedConditioning& prepared);

    // Mutable access to a prepared conditioning's (L, hidden) text rows —
    // the txt_in output the joint sequence is assembled from. Throws if
    // `uncond` is true but no uncond branch was prepared. Mutating these
    // invalidates the matching prefix cache; call reset_cache() after.
    brotensor::Tensor& text_rows(PreparedConditioning& prepared, bool uncond);

private:
    QwenImage21Transformer2DModel model_;
    brotensor::Tensor packed_;   // (H_lat*W_lat, in_channels) input scratch
    brotensor::Tensor tf_out_;   // packed velocity scratch
};

}  // namespace brodiffusion::dit
