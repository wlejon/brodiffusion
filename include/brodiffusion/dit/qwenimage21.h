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
#include "brodiffusion/detail/jit_fusion.h"
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

// ─── the research hooks, as ORDERED LISTS of bindings ───────────────────────
//
// Every in-network hook holds a list, not one configuration. All the bindings
// in a list apply to the same forward; for a given block the bindings whose
// range covers it compose the way each hook's semantics imply — deltas ADD,
// scales and masks MULTIPLY.
//
// The composition happens at BIND time, not per token. The modulation is four
// (1, hidden) rows shared by all 32 blocks, so the model folds every covering
// binding into one finished gate / scale row per distinct coverage pattern
// and the fused gated-residual kernel still consumes a single operand — an
// arbitrary number of armed bindings costs one extra (1, hidden) row each,
// never a second pass over the (L, hidden) activations.
//
// The single-binding set_*() calls remain, as sugar for "replace the list
// with this one entry" (or, given an empty tensor / an empty range, "clear").

// set_mod_delta / add_mod_delta: (1, 4*hidden) on the pre-chunk
// [scale1, gate1, scale2, gate2] modulation output, before the gates' tanh.
struct QwenImage21ModDeltaBinding {
    brotensor::Tensor delta;
    int block_lo = 0, block_hi = 0;
    QwenImage21ModTarget target = QwenImage21ModTarget::Target;
};

// Which sublayer a gate mask applies to. The mask scales the gated residual,
// and the two sublayers' gates behave completely differently late in the
// schedule: the attention gate is the only surface on this model that still
// has its full authority at step 6 of 8, while the SwiGLU one has spent most
// of its. A mask that hits both therefore drags the attention half's late
// authority down with it — which is why "restyle this region at step 6 and
// redirect nothing" needs Attn.
enum class QwenImage21GateSublayer {
    Both = 0,
    Attn = 1,   // gate1: the attention sublayer's residual
    Mlp  = 2    // gate2: the SwiGLU sublayer's residual
};

// set_gate_scale / add_gate_scale: scalar multipliers on the post-tanh gates,
// ONE PER (sublayer, row set) pair — four independent numbers, not a rank-1
// product of an (attn, mlp) pair with a (txt, img) pair.
//
// The difference matters for the cache. The prefix rows' gates are only ever
// applied on an extract step, so moving one costs a re-extract (+13% on the
// step). Under a rank-1 product the only route to "attention gate on the
// IMAGE rows" — the most useful dial on the model — was to move `attn`,
// which also moved attn*txt and dropped the cache. With the four independent,
// an image-side sweep is free and cannot disturb the prefix at all.
//
// set_gate_scale()'s (attn, mlp, txt, img) spelling is kept as sugar for the
// product; set_gate_scale_rows() addresses the four directly.
struct QwenImage21GateScaleBinding {
    float attn_txt = 1.0f, attn_img = 1.0f;
    float mlp_txt  = 1.0f, mlp_img  = 1.0f;
    int block_lo = 0, block_hi = 0;
};

// set_gate_delta / add_gate_delta: (1, 2*hidden) laid out [attn, mlp], added
// to the effective gate AFTER the tanh and after every gate scale.
struct QwenImage21GateDeltaBinding {
    brotensor::Tensor delta;
    int block_lo = 0, block_hi = 0;
    QwenImage21ModTarget target = QwenImage21ModTarget::Target;
};

// set_gate_mask / add_gate_mask: one value per joint-sequence row, multiplied
// into the gated residual of the sublayer(s) `which` names.
struct QwenImage21GateMaskBinding {
    brotensor::Tensor mask;
    int block_lo = 0, block_hi = 0;
    QwenImage21GateSublayer which = QwenImage21GateSublayer::Both;
};

// set_prefix_kv_scale / add_prefix_kv_scale: a per-layer attenuation of the
// CACHED prefix K/V, applied where the cache is read rather than written into
// it — so it is an idempotent dial (see set_prefix_kv_scales()).
//
// `row_scale`, when non-empty, holds one WEIGHT per PREFIX ROW saying how
// much of this binding's scales that row gets:
//
//     k_row[r] = 1 + row_scale[r] * (k_scale - 1)      (and likewise v)
//
// so all-ones is exactly the broadcast — which is what "empty = all rows"
// has to mean — all-zeros is the identity, and anything between fades the
// scale in. That is per-token prompt weighting, the "(word:1.3)" every image
// UI ships, done on the cached K/V where the whole generation actually reads
// the prompt. (The gate mask's text half is not a substitute — scaling a
// text token's residual updates is not scaling its contribution, because its
// K/V is dominated by what txt_in and the early blocks wrote before any gate
// applied.) Its length must equal the prefix length of the cache being read,
// or the forward throws saying both.
struct QwenImage21PrefixKvBinding {
    int layer_lo = 0, layer_hi = 0;
    float k_scale = 1.0f, v_scale = 1.0f;
    brotensor::Tensor row_scale;   // (prefix_len, 1) or empty = all rows
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
    //
    // Each hook holds an ORDERED LIST of bindings (see the QwenImage21*Binding
    // structs above); a block reads the composition of every binding whose
    // range covers it. The model resolves the lists into one finished
    // modulation per distinct coverage pattern at the top of each forward, so
    // a block loop with twelve bindings armed runs exactly the kernels a block
    // loop with one does.

    // Modulation delta (the AdaLN seam): add `delta` (1, 4*hidden_size, any
    // device/dtype) to the shared modulation OUTPUT — the pre-chunk vector
    // laid out [scale1, gate1, scale2, gate2] — for blocks
    // [block_lo, block_hi), on the row named by `target`. The delta lands
    // BEFORE the gates' tanh, so a gate component saturates rather than
    // running away. Deltas covering the same block ADD.
    //
    // set_mod_delta() replaces the whole list with this one binding; an empty
    // tensor or an empty range clears it. add_mod_delta() appends and returns
    // the new binding's index.
    //
    // See QwenImage21ModTarget: a Prefix / Both delta only takes effect on an
    // extract step, so reset the prefix cache after arming one.
    void set_mod_delta(const brotensor::Tensor& delta, int block_lo,
                       int block_hi,
                       QwenImage21ModTarget target = QwenImage21ModTarget::Target);
    int  add_mod_delta(const brotensor::Tensor& delta, int block_lo,
                       int block_hi,
                       QwenImage21ModTarget target = QwenImage21ModTarget::Target);
    void set_mod_deltas(const std::vector<QwenImage21ModDeltaBinding>& list);
    void clear_mod_deltas();
    const std::vector<QwenImage21ModDeltaBinding>& mod_deltas() const {
        return mod_deltas_;
    }

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
    // the TARGET rows see. Scales covering the same block MULTIPLY.
    //
    // This spelling is the rank-1 sugar: it binds the four independent
    // multipliers to attn*txt, attn*img, mlp*txt, mlp*img.
    // set_gate_scale_rows() sets them directly, which is what a caller wants
    // whenever only the image side should move — under the product, reaching
    // "attention gate on the image rows" had to move `attn`, which moved the
    // prefix product too and cost a re-extract.
    //
    // set_gate_scale() replaces the whole list with this one binding; all four
    // factors at 1, or an empty range, clears it. add_gate_scale() appends.
    void set_gate_scale(float attn_scale, float mlp_scale, float txt_scale,
                        float img_scale, int block_lo, int block_hi);
    int  add_gate_scale(float attn_scale, float mlp_scale, float txt_scale,
                        float img_scale, int block_lo, int block_hi);

    // The four independent multipliers, one per (sublayer, row set):
    //   attn_txt  gate1 on the t = 0 rows   (text / condition images)
    //   attn_img  gate1 on the sampled-t rows (the image being generated)
    //   mlp_txt   gate2 on the t = 0 rows
    //   mlp_img   gate2 on the sampled-t rows
    // Only the *_txt pair is prefix-side, so an *_img sweep never re-extracts.
    void set_gate_scale_rows(float attn_txt, float attn_img, float mlp_txt,
                             float mlp_img, int block_lo, int block_hi);
    int  add_gate_scale_rows(float attn_txt, float attn_img, float mlp_txt,
                             float mlp_img, int block_lo, int block_hi);
    void set_gate_scales(const std::vector<QwenImage21GateScaleBinding>& list);
    void clear_gate_scales();
    const std::vector<QwenImage21GateScaleBinding>& gate_scales() const {
        return gate_scales_;
    }

    // Gate delta: add `delta` (1, 2*hidden_size, laid out [attn, mlp], any
    // device/dtype) to the EFFECTIVE gate of blocks [block_lo, block_hi) —
    // the value that multiplies the residual, i.e.
    //
    //     g_eff = set_gate_scale_factor * tanh(gate) + delta
    //
    // `target` picks the row class the delta lands on, exactly as
    // set_mod_delta()'s does: Target is the sampled-t row (the image being
    // generated), Prefix the t = 0 row (text and condition-image tokens).
    // Deltas covering the same block ADD, and they are applied after every
    // covering gate scale has multiplied the tanh.
    //
    // set_gate_delta() replaces the whole list with this one binding; an empty
    // tensor or an empty range clears it. add_gate_delta() appends.
    //
    // Why this exists next to set_mod_delta(). That hook adds BEFORE the
    // tanh, so its authority over a gate channel is tanh'(g) — and the
    // research found 74% of gate2's channels sitting where tanh'(g) < 0.05,
    // which makes the model's largest modulation chunk its least responsive
    // dial: a pre-tanh delta of 1.0 moves those channels by under 0.05.
    // Adding after the tanh gives every channel unit authority, at the cost
    // of leaving the (-1, 1) range tanh guarantees — which is the point,
    // since a gate above 1 is not otherwise reachable.
    //
    // Cost: nothing per token. The modulation is four (1, hidden) rows shared
    // by all 32 blocks, so the delta is folded into the gate row once per
    // forward and the fused gated-residual kernel consumes the combined row
    // as its gate operand — there is no second pass over the activations.
    //
    // Like every prefix-side hook, a Prefix / Both delta only takes effect on
    // an extract step; Pipeline's qi21_set_gate_delta() resets the cache for
    // you, a bare DiT caller must reset it itself.
    void set_gate_delta(const brotensor::Tensor& delta, int block_lo,
                        int block_hi,
                        QwenImage21ModTarget target =
                            QwenImage21ModTarget::Target);
    int  add_gate_delta(const brotensor::Tensor& delta, int block_lo,
                        int block_hi,
                        QwenImage21ModTarget target =
                            QwenImage21ModTarget::Target);
    void set_gate_deltas(const std::vector<QwenImage21GateDeltaBinding>& list);
    void clear_gate_deltas();
    const std::vector<QwenImage21GateDeltaBinding>& gate_deltas() const {
        return gate_deltas_;
    }

    // Per-token gate mask over blocks [block_lo, block_hi): after the tanh
    // (and any set_gate_scale / set_gate_delta), the gated residual of row r
    // is multiplied by mask[r] for the sublayer(s) `which` names. `mask`
    // holds prefix_len + hp*wp values in joint forward order (any
    // device/dtype); a cached step reads only its target slice. Zeroing a row
    // removes that token's residual updates entirely for the masked blocks.
    // Masks covering the same block and sublayer MULTIPLY elementwise. An
    // empty tensor clears.
    //
    // `which` is what makes this a usable regional brush late in the
    // schedule: Attn keeps the one gate that still has authority at step 6
    // and leaves the SwiGLU half — which does not — alone.
    //
    // A forward whose joint length differs from an armed mask THROWS, naming
    // both lengths. It used to skip the mask silently, and the failure mode
    // was a study concluding the surface does nothing: an all-zero mask
    // written at img_len instead of prefix_len + img_len rendered the
    // baseline, to the pixel.
    //
    // set_gate_mask() replaces the whole list with this one binding;
    // add_gate_mask() appends.
    void set_gate_mask(
        const brotensor::Tensor& mask, int block_lo, int block_hi,
        QwenImage21GateSublayer which = QwenImage21GateSublayer::Both);
    int  add_gate_mask(
        const brotensor::Tensor& mask, int block_lo, int block_hi,
        QwenImage21GateSublayer which = QwenImage21GateSublayer::Both);
    void set_gate_masks(const std::vector<QwenImage21GateMaskBinding>& list);
    void clear_gate_masks();
    const std::vector<QwenImage21GateMaskBinding>& gate_masks() const {
        return gate_masks_;
    }

    // Prefix KV attenuation: multiply the CACHED prefix K of layers
    // [layer_lo, layer_hi) by `k_scale` and V by `v_scale` wherever a cached
    // forward reads them. Bindings covering the same layer MULTIPLY.
    //
    // Why this is a model hook rather than an operation on the cache. The
    // obvious implementation — scale the cache in place — is cumulative: the
    // dial compounds every time it is set, so it cannot be used through a
    // generate() loop at all, and repeated calls never mean the same thing
    // twice. Applying the composed per-layer factor at the point the cached
    // rows are copied into the attention's K/V instead makes it an ordinary
    // idempotent dial, costs one pass over the (prefix_len, hidden) prefix
    // per layer (against the whole joint attention), and needs no second copy
    // of the cache. It also survives a cache reset, because it describes what
    // the model does with a prefix, not what is in one.
    //
    // A scale takes effect on cached steps only: an extract step attends the
    // prefix rows it is computing, not the cache.
    //
    // `row_scale`, when non-empty, is one WEIGHT per PREFIX ROW on this
    // binding's scales: k_row[r] = 1 + row_scale[r] * (k_scale - 1). All
    // ones is the broadcast, all zeros the identity — per-token prompt
    // weighting on the K/V the whole generation attends to. Its length must
    // equal the cache's prefix length or the forward throws saying both.
    void set_prefix_kv_scale(int layer_lo, int layer_hi, float k_scale,
                             float v_scale,
                             const brotensor::Tensor& row_scale = {});
    int  add_prefix_kv_scale(int layer_lo, int layer_hi, float k_scale,
                             float v_scale,
                             const brotensor::Tensor& row_scale = {});
    void set_prefix_kv_scales(
        const std::vector<QwenImage21PrefixKvBinding>& list);
    void clear_prefix_kv_scales();
    const std::vector<QwenImage21PrefixKvBinding>& prefix_kv_scales() const {
        return prefix_kv_scales_;
    }

    // Gate activity capture: when `sink` is non-null, every subsequent
    // forward overwrites it with the per-row mean EFFECTIVE attention gate of
    // each block, layout (num_layers, prefix_len + hp*wp) row-major — the
    // value that actually multiplied that row's attention residual, i.e.
    // mean_d (set_gate_scale factor * tanh(gate1)[d] + set_gate_delta[d])
    // times its set_gate_mask entry. On a cached step the prefix columns carry the
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

    // The four (1, hidden) modulation rows a block runs on, plus the norm_out
    // scale. `_t` is the sampled timestep's row, `_0` the t = 0 row that text
    // and condition-image tokens use under causal_condition.
    //
    // Every hook is already FOLDED IN: the gates carry their composed scale
    // factor and their composed post-tanh delta, so a block reads one finished
    // row per sublayer per row class and the fused gated-residual kernel keeps
    // its single gate operand however many bindings are armed. A forward
    // builds one of these per distinct hook COVERAGE — with nothing armed
    // that is one; with the research controller's dial vector armed it is two
    // or three; it is bounded by the block count either way.
    struct Modulation {
        brotensor::Tensor scale1_t, gate1_t, scale2_t, gate2_t;
        brotensor::Tensor scale1_0, gate1_0, scale2_0, gate2_0;
        brotensor::Tensor final_scale;   // norm_out, target rows (t)
        // Mean over hidden of the effective attention gate of each row class,
        // for capture_gates(). Only filled when a sink is armed — the
        // readback is a device sync.
        float mean_g1_t = 0.0f, mean_g1_0 = 0.0f;
    };

    // One block's resolved hook coverage: which bindings of each list cover
    // it. Blocks with identical coverage share a Modulation.
    struct BlockCoverage {
        std::vector<int> mod_deltas, gate_scales, gate_deltas, gate_masks;
        bool operator==(const BlockCoverage& o) const {
            return mod_deltas == o.mod_deltas &&
                   gate_scales == o.gate_scales &&
                   gate_deltas == o.gate_deltas && gate_masks == o.gate_masks;
        }
    };

    void load_impl_(const std::vector<const brotensor::safetensors::File*>& shards,
                    const std::string& prefix,
                    const std::function<bool()>& should_cancel);

    // One linear at the compute dtype, dispatching dense vs INT8 (W8A16).
    brotensor::Tensor lin_(const Linear& l, const brotensor::Tensor& X);

    // The same linear writing into a caller-owned Y. The block loop uses this
    // for every activation a JIT site consumes: a trace binds to the buffer
    // addresses it was compiled against, and a fresh Y per block would move
    // them 32 times a step and force 32 re-traces.
    void lin_into_(const Linear& l, const brotensor::Tensor& X,
                   brotensor::Tensor& Y);

    // Non-affine LayerNorm (eps = cfg_.eps) over (L, hidden).
    void layernorm_(const brotensor::Tensor& X, brotensor::Tensor& Y);

    // Timestep embedding + the per-block modulation variants, for flow time
    // `timestep`. Fills mods_ with one Modulation per distinct hook coverage
    // and block_variant_ with each block's index into it — so the block loop
    // is a lookup, not a composition. `raw_mod` / `raw_temb`, when non-null,
    // receive the (n_rows, 4*hidden) modulation output and the (n_rows,
    // hidden) time embedding before any chunking, which is what
    // compute_time_mod() reads back.
    void build_modulation_(float timestep, bool build_variants = true,
                           brotensor::Tensor* raw_temb = nullptr,
                           brotensor::Tensor* raw_mod = nullptr);
    // Resolve the armed binding lists into per-block coverage. Fills
    // block_coverage_ and returns the number of distinct coverages.
    void resolve_coverage_();
    // Slice one (n_rows, 4*hidden) modulation output into the per-sublayer
    // rows, applying tanh to the gates and the norm_out scale delta.
    void chunk_modulation_(const brotensor::Tensor& mod,
                           const brotensor::Tensor& final_all, Modulation& m);
    // Fold one coverage's composed gate scale factors and post-tanh gate
    // deltas into m's four gate rows, and (when a capture sink is armed) the
    // resulting attention-gate means.
    void fold_gate_hooks_(const BlockCoverage& cov, Modulation& m);
    // Compose the armed prefix-KV bindings into the per-layer scalars and
    // per-row vectors the forward applies. Called on every list change.
    void compose_prefix_scales_();

    // Rank-1 expansion of a per-row host vector into an (n_rows, cols)
    // device tensor, so an elementwise multiply can apply it. Used by the
    // per-row prefix KV dial; `dst` is resized and overwritten.
    void expand_prefix_rows_(const std::vector<float>& rows, int n_rows,
                             int cols, brotensor::Tensor& dst);
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

    // Trace-JIT fusion sites. Each holds a compiled kernel for one of the
    // block's memory-bound seams and replays it for all 32 blocks and every
    // step; the `_pre` / `_post` pairs are the two modulation row ranges the
    // t = 0 prefix split creates (a cached step uses only `_post`). They live
    // on the model rather than in statics so two loaded transformers do not
    // fight over one binding.
    struct JitSites {
        detail::JitSite attn_norm_pre{"qi21.attn.norm_modulate[pre]"};
        detail::JitSite attn_norm_post{"qi21.attn.norm_modulate"};
        detail::JitSite mlp_norm_pre{"qi21.mlp.norm_modulate[pre]"};
        detail::JitSite mlp_norm_post{"qi21.mlp.norm_modulate"};
        detail::JitSite attn_gate_pre{"qi21.attn.gated_residual[pre]"};
        detail::JitSite attn_gate_post{"qi21.attn.gated_residual"};
        detail::JitSite mlp_gate_pre{"qi21.mlp.gated_residual[pre]"};
        detail::JitSite mlp_gate_post{"qi21.mlp.gated_residual"};
        detail::JitSite swiglu{"qi21.mlp.swiglu"};
        detail::JitSite final_norm{"qi21.norm_out"};
    };
    JitSites jit_;

    // ── research-hook state ───────────────────────────────────────────────
    //
    // The armed binding lists. A binding's tensors are held at the compute
    // dtype on the default device; a gate delta is kept pre-split into its
    // [attn] and [mlp] halves so folding it costs no slicing per forward, and
    // a gate mask keeps a host copy so gate capture can report the masked
    // value without a readback.
    std::vector<QwenImage21ModDeltaBinding>  mod_deltas_;
    std::vector<QwenImage21GateScaleBinding> gate_scales_;
    std::vector<QwenImage21GateDeltaBinding> gate_deltas_;
    std::vector<QwenImage21GateMaskBinding>  gate_masks_;
    std::vector<QwenImage21PrefixKvBinding>  prefix_kv_scales_;

    // Per gate-delta binding, its (1, hidden) halves.
    std::vector<brotensor::Tensor> gate_delta_attn_, gate_delta_mlp_;
    // Per gate-mask binding, its values on host, for gate capture.
    std::vector<std::vector<float>> gate_mask_host_;

    brotensor::Tensor norm_out_delta_; // (1, hidden) compute dtype; empty = off

    // Resolved once per forward by resolve_coverage_() / build_modulation_().
    // `mods_` holds one Modulation per distinct coverage and lives on the
    // model so its (1, hidden) rows keep their addresses across forwards — a
    // JIT site's binding is keyed on the gate operand's address.
    std::vector<BlockCoverage> coverages_;     // one per distinct coverage
    std::vector<int> block_variant_;           // block -> coverages_ index
    std::vector<Modulation> mods_;             // one per coverage

    // Per coverage, the composed per-token mask: the (Lq, hidden) rank-1
    // expansion the fused residual consumes, and the host values capture
    // reports. One pair per SUBLAYER, because a mask can name one of them.
    // An empty host vector = that coverage has no mask for that sublayer.
    std::vector<brotensor::Tensor> mask_full_attn_, mask_full_mlp_;
    std::vector<std::vector<float>> mask_host_attn_, mask_host_mlp_;
    brotensor::Tensor gate_ones_row_;  // (1, hidden) ones

    // Composed per-layer prefix KV factors, rebuilt whenever the list
    // changes. Empty = every layer at 1/1. `prefix_*_row_` holds the composed
    // per-row vector for each layer (empty = that layer is scalar-only); the
    // scalars are already folded into it when it is non-empty, and
    // prefix_row_len_ is the length every non-empty one has.
    std::vector<float> prefix_k_scale_, prefix_v_scale_;
    std::vector<std::vector<float>> prefix_k_row_, prefix_v_row_;
    int prefix_row_len_ = 0;
    // The (prefix_len, hidden) rank-1 expansion of one layer's row vector,
    // rebuilt per layer that needs one. Two buffers so K and V can differ.
    brotensor::Tensor prefix_row_full_k_, prefix_row_full_v_;

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
// One CFG branch's image-conditioned prefix, as prepare_edit() takes it.
//
// `segments` describes the WHOLE joint prefix — the interleaved Text and
// Image runs the vision-language encoder emitted, in order. Its Text runs
// consume the branch's (compacted) conditioning rows in order and must sum to
// their count; its Image runs consume `cond_latents`' rows in order and must
// sum to that tensor's row count. An empty `segments` means "no condition
// images", i.e. the plain text-to-image prefix, which is also what a
// default-constructed value says.
struct QwenImage21EditPrefix {
    std::vector<QwenImage21Segment> segments;
    // (sum of the Image segments' h*w, in_channels) — every condition image's
    // VAE latent, spatially flattened and concatenated in template order,
    // already normalised by latents_mean / latents_std.
    brotensor::Tensor cond_latents;

    bool empty() const { return segments.empty(); }
};

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

    // prepare() for the image-conditioned path: the same text encode, plus a
    // per-branch description of the joint prefix the condition images
    // occupy. `uncond_prefix` may be null, in which case the uncond branch
    // (when `cond` carries one) reuses `cond_prefix` — the usual case, since
    // the condition images do not change between the two prompts and only
    // the text run lengths do. Pass an empty QwenImage21EditPrefix for a
    // branch with no condition images.
    //
    // The prepared payload owns the condition latents, so they stay resident
    // for the whole generation: the prefix is re-read on every extract step
    // (each reset cache, each resolution change), not just once.
    PreparedConditioning prepare_edit(const Conditioning& cond,
                                      const QwenImage21EditPrefix& cond_prefix,
                                      const QwenImage21EditPrefix* uncond_prefix
                                          = nullptr);

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
