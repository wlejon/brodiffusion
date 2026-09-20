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

// Which of the two modulation rows a research delta addresses. Under
// causal_condition the prefix (text / condition-image) rows are modulated
// from the t = 0 row and the target image's rows from the sampled t, so the
// two are independently steerable — unlike Krea 2, where one shared vector
// drives every row.
//
// IMPORTANT: the prefix rows are only ever computed on the EXTRACT step (the
// first forward against a fresh prefix KV cache). A Prefix / Both delta armed
// after that step has no effect until the cache is reset — Pipeline's
// qi21_set_mod_delta() resets it for you; a bare DiT caller must call
// QwenImage21PrefixCache::reset() itself.
enum class QwenImage21ModTarget {
    Target = 0,   // the sampled-t row: the image being generated
    Prefix = 1,   // the t = 0 row: text and condition-image tokens
    Both   = 2
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
    int  num_layers() const { return static_cast<int>(k_.size()); }
    // Does this cache describe the same joint layout as (prefix_len, hp, wp)?
    bool matches(int prefix_len, int hp, int wp) const {
        return valid() && prefix_len_ == prefix_len && hp_ == hp && wp_ == wp;
    }

    // ── research surface over the cached prefix ───────────────────────────
    //
    // The cache IS the text conditioning as far as every step after the first
    // is concerned: attenuating or mixing it steers generation without
    // re-encoding a prompt. Both operations are in-place on the post-RoPE K/V
    // and take effect on the very next cached forward.

    // Multiply layers [layer_lo, layer_hi)'s cached K by `k_scale` and V by
    // `v_scale`. 1/1 over the full range is a no-op. Attenuating V alone
    // fades the prefix's contribution while leaving the attention pattern it
    // induces intact; attenuating K alone flattens that pattern instead.
    void scale_kv(int layer_lo, int layer_hi, float k_scale, float v_scale);

    // Linear blend towards another cache's prefix:
    //     k = (1-alpha)*k + alpha*other.k   (and likewise v)
    // `other` must describe the same layout (same prefix_len / target grid /
    // layer count), which in practice means two prompts that tokenize to the
    // same length. alpha 0 is a no-op, 1 replaces this prefix wholesale.
    // Throws std::runtime_error on a layout mismatch.
    void blend_from(const QwenImage21PrefixCache& other, float alpha);

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

    // ── research hooks ────────────────────────────────────────────────────
    //
    // Every hook below is armed on the model and applies to every subsequent
    // forward until cleared. They are the Qwen-Image 2.1 analogue of Krea 2's
    // set_mod_delta / gate dials (dit/krea2.h), shaped by the two structural
    // differences: there is ONE modulation vector for all 32 blocks (so a
    // block range is realised by computing a second, deltaed copy of it), and
    // the prefix and target rows read DIFFERENT rows of it.

    // Modulation delta (the AdaLN seam): add `delta` (1, 4*hidden_size, any
    // device/dtype) to the shared modulation OUTPUT — the pre-chunk vector
    // laid out [scale1, gate1, scale2, gate2] — for blocks
    // [block_lo, block_hi), on the row named by `target`. The delta lands
    // BEFORE the gates' tanh, so a gate component saturates rather than
    // running away. An empty tensor clears the hook.
    //
    // See QwenImage21ModTarget: a Prefix / Both delta only takes effect on an
    // extract step, so reset the prefix cache after arming one.
    void set_mod_delta(const brotensor::Tensor& delta, int block_lo,
                       int block_hi,
                       QwenImage21ModTarget target = QwenImage21ModTarget::Target);

    // Timestep-embedding readout at flow time `timestep`, no image forward.
    //   temb_out: (2, hidden_size) FP32 — row 0 the sampled t, row 1 t = 0.
    //   mod_out:  (2, 4*hidden_size) FP32, the raw modulation output in the
    //             same row order, BEFORE tanh — i.e. exactly the space
    //             set_mod_delta() adds into.
    // With causal_condition disabled both rows carry the sampled t.
    void compute_time_mod(float timestep, brotensor::Tensor& temb_out,
                          brotensor::Tensor& mod_out);

    // Gate scale: scalar multipliers on the POST-tanh residual gates of
    // blocks [block_lo, block_hi). `attn_scale` scales gate1 (the attention
    // sublayer), `mlp_scale` gate2 (the SwiGLU sublayer); orthogonally,
    // `txt_scale` scales the gate the PREFIX rows see and `img_scale` the one
    // the TARGET rows see. All four at 1 clears the hook.
    void set_gate_scale(float attn_scale, float mlp_scale, float txt_scale,
                        float img_scale, int block_lo, int block_hi);

    // Per-token gate mask over blocks [block_lo, block_hi): after the tanh
    // (and any set_gate_scale), both sublayers' gated residual for row r is
    // multiplied by mask[r]. `mask` holds prefix_len + hp*wp values in joint
    // forward order (any device/dtype); a cached step reads only its target
    // slice. Zeroing a row removes that token's residual updates entirely for
    // the masked blocks. An empty tensor clears; a forward whose joint length
    // differs from the mask skips it.
    void set_gate_mask(const brotensor::Tensor& mask, int block_lo,
                       int block_hi);

    // Gate activity capture: when `sink` is non-null, every subsequent
    // forward overwrites it with the per-row mean EFFECTIVE attention gate of
    // each block, layout (num_layers, prefix_len + hp*wp) row-major — the
    // value that actually multiplied that row's attention residual, i.e.
    // mean_d tanh(gate1)[d] times the row's set_gate_scale factor times its
    // set_gate_mask entry. On a cached step the prefix columns carry the
    // values the extract step would have applied (the prefix rows are not
    // recomputed), so the row layout stays stable across a denoise loop.
    //
    // Note what this can and cannot show: the modulation is SHARED by all 32
    // blocks, so with no hooks armed every block's row is the same pair of
    // constants (one for the prefix rows, one for the target rows). The
    // capture exists to verify a mask / scale landed where it was aimed and
    // to read the two gate magnitudes against the timestep — not to find
    // per-block structure, of which the architecture has none.
    // nullptr disables. The pointer must outlive captures.
    void capture_gates(std::vector<float>* sink);

    // norm_out scale delta: add `delta` (1, hidden_size) to the final
    // adaptive scale the target rows pass through on the way to proj_out —
    // the cheapest single knob on the model's output magnitude / colour.
    // An empty tensor clears.
    void set_norm_out_scale_delta(const brotensor::Tensor& delta);

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
        // set_gate_scale variants of the four gates, built once per forward
        // and used by the blocks inside [gate_lo_, gate_hi_). Valid only when
        // `scaled` is true.
        brotensor::Tensor gs1_t, gs2_t, gs1_0, gs2_0;
        bool scaled = false;
        // Mean over hidden of each gate, for capture_gates(). Index order
        // matches [gate1_t, gate1_0, gs1_t, gs1_0].
        float mean_g1_t = 0.0f, mean_g1_0 = 0.0f;
        float mean_gs1_t = 0.0f, mean_gs1_0 = 0.0f;
    };

    void load_impl_(const std::vector<const brotensor::safetensors::File*>& shards,
                    const std::string& prefix,
                    const std::function<bool()>& should_cancel);

    // One linear at the compute dtype, dispatching dense vs INT8 (W8A16).
    brotensor::Tensor lin_(const Linear& l, const brotensor::Tensor& X);

    // Non-affine LayerNorm (eps = cfg_.eps) over (L, hidden).
    void layernorm_(const brotensor::Tensor& X, brotensor::Tensor& Y);

    // Timestep embedding + shared modulation chunks, for flow time
    // `timestep`. When `m_delta` is non-null and a set_mod_delta() hook is
    // armed, it additionally receives the deltaed variant and *has_delta is
    // set — blocks inside the delta's range use that one. `raw_mod` /
    // `raw_temb`, when non-null, receive the (n_rows, 4*hidden) modulation
    // output and the (n_rows, hidden) time embedding before any chunking,
    // which is what compute_time_mod() reads back.
    void build_modulation_(float timestep, Modulation& m,
                           Modulation* m_delta = nullptr,
                           bool* has_delta = nullptr,
                           brotensor::Tensor* raw_temb = nullptr,
                           brotensor::Tensor* raw_mod = nullptr);
    // Slice one (n_rows, 4*hidden) modulation output into the per-sublayer
    // rows, applying tanh to the gates and the norm_out scale delta.
    void chunk_modulation_(const brotensor::Tensor& mod,
                           const brotensor::Tensor& final_all, Modulation& m);
    // Fill m's gs* variants from its gates under the set_gate_scale factors.
    void scale_gates_(Modulation& m);
    // Mean over hidden of a (1, hidden) row, on host.
    float row_mean_(const brotensor::Tensor& row) const;

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

    // ── research-hook state ───────────────────────────────────────────────
    brotensor::Tensor mod_delta_;      // (1, 4*hidden) compute dtype; empty = off
    int mod_delta_lo_ = 0, mod_delta_hi_ = 0;
    QwenImage21ModTarget mod_delta_target_ = QwenImage21ModTarget::Target;

    brotensor::Tensor norm_out_delta_; // (1, hidden) compute dtype; empty = off

    float gate_attn_scale_ = 1.0f, gate_mlp_scale_ = 1.0f;
    float gate_txt_scale_  = 1.0f, gate_img_scale_ = 1.0f;
    int gate_lo_ = 0, gate_hi_ = 0;

    brotensor::Tensor gate_mask_;      // (L, 1) compute dtype; empty = off
    std::vector<float> gate_mask_host_;  // the same values, for gate capture
    int gate_mask_lo_ = 0, gate_mask_hi_ = 0;
    // (Lq, hidden) rank-1 expansion of the active mask slice, built once per
    // forward and reused by every masked block.
    brotensor::Tensor gate_mask_full_;
    brotensor::Tensor gate_ones_row_;  // (1, hidden) ones

    std::vector<float>* gate_sink_ = nullptr;
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

    // Mutable access to a branch's prefix KV cache — the research seam over
    // the cached text K/V (QwenImage21PrefixCache::scale_kv / blend_from),
    // which after the extract step IS what the target attends to. Throws if
    // `uncond` is true but no uncond branch was prepared.
    QwenImage21PrefixCache& prefix_cache(PreparedConditioning& prepared,
                                         bool uncond);

    // True when prepare() built an uncond branch (guidance_scale > 1 at prime
    // time). Lets a caller apply a prefix-side hook to both branches without
    // guessing.
    bool has_uncond(const PreparedConditioning& prepared) const;

private:
    QwenImage21Transformer2DModel model_;
    brotensor::Tensor packed_;   // (H_lat*W_lat, in_channels) input scratch
    brotensor::Tensor tf_out_;   // packed velocity scratch
};

}  // namespace brodiffusion::dit
