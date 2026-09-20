#pragma once

/* qwenimage21_capi.h — flat C API over the Qwen-Image 2.1 components, for
 * research bindings.
 *
 * Built as the `qwenimage21_capi` shared library (qwenimage21_capi.dll) and
 * consumed from any language with a C FFI. The sibling of krea2_capi.h, and
 * the same shape: the three components — Qwen3-VL-8B text encoder, the 7.1B
 * block-causal DiT, the 16x RGBA autoencoder — are wrapped INDIVIDUALLY so
 * research code loads only what a question needs, injects at the component
 * boundaries, and drives the step loop itself (Euler flow matching and the
 * x0-preview identity are host-side math).
 *
 * What is different from krea2_capi, and why:
 *
 *   - The DiT keeps a PREFIX KV CACHE. qi_forward manages it internally: the
 *     first forward after a prompt / grid change extracts it, later forwards
 *     decode from it. That is the difference between a 4200-token forward and
 *     a 4096-token one at 1024x1024. Any change to the prefix side — the text
 *     rows, a prefix-row modulation delta, a text-side gate factor, a gate
 *     mask — needs qi_reset_cache() before it lands. Every such entry point
 *     says so.
 *   - Prefix and target rows read DIFFERENT rows of the one shared modulation
 *     vector (t = 0 vs the sampled t), so qi_set_mod_delta takes a target
 *     selector. A QI_MOD_TARGET delta survives the cache untouched; the other
 *     two do not.
 *   - The conditioning is ONE (n, 4096) run of Qwen3-VL hidden states, not a
 *     layer stack, and its length depends on the prompt — hence the two-call
 *     encode protocol below rather than a fixed max_seq buffer.
 *
 * Conventions:
 *   - Every array crosses the boundary as caller-allocated FP32, C-contiguous,
 *     row-major, shapes fixed by the model facts below. Internals cast to the
 *     compute dtype (BF16 on CUDA).
 *   - Functions return 0 on success, -1 on failure (qi_last_error() has the
 *     message; per-thread). Functions that return a count return -1 on failure.
 *   - One qi_ctx per model dir; not thread-safe (drive from one thread).
 */

#include <stdint.h>

#if defined(_WIN32)
#  if defined(QI_CAPI_BUILD)
#    define QI_API __declspec(dllexport)
#  else
#    define QI_API __declspec(dllimport)
#  endif
#else
#  define QI_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qi_ctx qi_ctx;

enum {
    QI_LOAD_TE  = 1,   /* Qwen3-VL-8B text encoder + tokenizer */
    QI_LOAD_DIT = 2,   /* the 7.1B block-causal transformer */
    QI_LOAD_VAE = 4,   /* the 16x RGBA autoencoder (both halves) */
    QI_LOAD_ALL = 7
};

/* Which modulation row a qi_set_mod_delta applies to. Under causal_condition
 * the prefix (text / condition-image) rows are modulated from t = 0 and the
 * target image's rows from the sampled t. */
enum {
    QI_MOD_TARGET = 0,   /* the image being generated; survives the cache */
    QI_MOD_PREFIX = 1,   /* text tokens; needs qi_reset_cache() to land */
    QI_MOD_BOTH   = 2
};

/* Message for the most recent failure on this thread ("" if none). */
QI_API const char* qi_last_error(void);

/* Open a Qwen-Image 2.1 model directory (the diffusers layout with
 * model_index.json) and load the requested components. quantize != 0 loads
 * the DiT and the text encoder INT8 weight-only, which is the only
 * configuration that fits 1024x1024 on a 24 GB card (peak ~22 GiB).
 * NULL on failure. */
QI_API qi_ctx* qi_open(const char* model_dir, int components, int quantize);
QI_API void    qi_close(qi_ctx* c);

/* ── model facts (valid after any successful qi_open) ─────────────────── */
QI_API int qi_hidden_size(const qi_ctx* c);      /* 4096 */
QI_API int qi_num_layers(const qi_ctx* c);       /* 32   */
QI_API int qi_text_hidden_dim(const qi_ctx* c);  /* 4096 (Qwen3-VL-8B width) */
QI_API int qi_latent_channels(const qi_ctx* c);  /* 64 (patch_size is 1) */
QI_API int qi_vae_scale(const qi_ctx* c);        /* 16 */

/* ── text encoder (QI_LOAD_TE) ────────────────────────────────────────────
 * The conditioning's length depends on the prompt, so encoding is a
 * two-call protocol: qi_encode_prompt runs the encoder and parks the result
 * in the ctx, returning the row count; the getters copy it out.
 *
 *   qi_encode_prompt -> n_valid, or -1.
 *   qi_get_prompt_embeds: (n_valid, text_hidden_dim) floats — the last
 *       decoder layer's residual stream, pre-final-norm, system rows dropped.
 *       This is the injection point for control axes and edited rows.
 *   qi_prompt_num_ids / qi_get_prompt_ids: the FULL template token sequence
 *       (system prefix included, so its length is n_valid + drop_idx). Kept
 *       for debugging and so an edit path can find the <|image_pad|> slots. */
QI_API int qi_encode_prompt(qi_ctx* c, const char* prompt);
QI_API int qi_get_prompt_embeds(qi_ctx* c, float* out);
QI_API int qi_prompt_num_ids(qi_ctx* c);
QI_API int qi_get_prompt_ids(qi_ctx* c, int32_t* out);

/* ── DiT (QI_LOAD_DIT) ─────────────────────────────────────────────────── */

/* txt_in over the encoder's hidden states — the timestep-independent half.
 * embeds: (n, text_hidden_dim), as qi_get_prompt_embeds produced them or
 * modified. txt_out must hold (n, hidden_size). Returns n, or -1.
 * Changing the text rows invalidates the prefix cache; qi_forward notices a
 * row-count change by itself, but an EDIT AT THE SAME LENGTH does not change
 * the layout, so call qi_reset_cache() after one. */
QI_API int qi_encode_text(qi_ctx* c, const float* embeds, int n,
                          float* txt_out);

/* One transformer forward: the flow-matching velocity for the image tokens.
 *   latent: (h_lat*w_lat, latent_channels) — the target latent, spatially
 *           flattened, NOT 2x2 packed (patch_size is 1).
 *   txt:    (n_txt, hidden_size) — qi_encode_text output, or any injected
 *           replacement.
 *   timestep in [0,1] (the flow time, not the 0..1000 scale).
 *   out:    (h_lat*w_lat, latent_channels).
 * The prefix KV cache is managed inside: the first call after a prompt or
 * grid change extracts it, later calls decode from it. */
QI_API int qi_forward(qi_ctx* c, const float* latent, int h_lat, int w_lat,
                      const float* txt, int n_txt, float timestep, float* out);

/* Drop the prefix KV cache so the next qi_forward re-extracts. Required after
 * any prefix-side change (see the header note). Always succeeds. */
QI_API int qi_reset_cache(qi_ctx* c);

/* Modulation delta: add delta (4 * hidden_size, laid out
 * [scale1, gate1, scale2, gate2]) to the shared modulation output for blocks
 * [block_lo, block_hi), on the row named by `target` (QI_MOD_*). The delta
 * lands before the gates' tanh. delta == NULL clears. A QI_MOD_PREFIX /
 * QI_MOD_BOTH delta needs qi_reset_cache() to take effect. */
QI_API int qi_set_mod_delta(qi_ctx* c, const float* delta, int block_lo,
                            int block_hi, int target);

/* Timestep readout, no image forward, at flow time `timestep`:
 *   temb_out: (2 * hidden_size)     row 0 = the sampled t, row 1 = t = 0
 *   mod_out:  (2 * 4 * hidden_size) same row order, BEFORE tanh — exactly
 *                                   the space qi_set_mod_delta adds into.
 * Either out may be NULL. */
QI_API int qi_time_mod(qi_ctx* c, float timestep, float* temb_out,
                       float* mod_out);

/* Post-tanh gate dials for blocks [block_lo, block_hi): attn_scale scales the
 * attention sublayer's gate and mlp_scale the SwiGLU's; orthogonally,
 * txt_scale scales the gate the PREFIX rows see and img_scale the TARGET
 * rows'. All four at 1 clears. A txt_scale != 1 needs qi_reset_cache(). */
QI_API int qi_set_gate_scale(qi_ctx* c, float attn_scale, float mlp_scale,
                             float txt_scale, float img_scale, int block_lo,
                             int block_hi);

/* Per-token gate mask: both sublayers' gated residual for row r of blocks
 * [block_lo, block_hi) is multiplied by mask[r], after the tanh and after any
 * qi_set_gate_scale. `n` must equal the forward's n_txt + h_lat*w_lat; a
 * forward with a different joint length skips the mask. NULL clears. Needs
 * qi_reset_cache() for its prefix half to land. */
QI_API int qi_set_gate_mask(qi_ctx* c, const float* mask, int64_t n,
                            int block_lo, int block_hi);

/* norm_out scale delta: add delta (hidden_size) to the final adaptive scale
 * the target rows pass through before proj_out. NULL clears. */
QI_API int qi_set_norm_out_scale_delta(qi_ctx* c, const float* delta);

/* Gate activity capture: enable, run qi_forward, then read the per-block
 * per-row mean EFFECTIVE attention gate. Layout (num_layers, n_txt +
 * h_lat*w_lat) row-major from the most recent forward; qi_gates_size gives
 * the float count (0 if none). Note the modulation is shared by all 32
 * blocks, so with no hooks armed every row is the same pair of constants —
 * this reads back what a mask / scale actually did, not per-block structure
 * the architecture does not have. */
QI_API int     qi_capture_gates(qi_ctx* c, int enable);
QI_API int64_t qi_gates_size(qi_ctx* c);
QI_API int     qi_get_gates(qi_ctx* c, float* out);

/* ── prefix KV cache surface (QI_LOAD_DIT) ─────────────────────────────────
 * After the extract step the cache IS the conditioning as far as every later
 * forward is concerned, so attenuating or mixing it steers generation without
 * re-encoding anything. Both operations apply to the LIVE cache and take
 * effect on the very next qi_forward, with no re-extraction. */

/* Multiply the cached K of layers [layer_lo, layer_hi) by k_scale and V by
 * v_scale. Attenuating V alone fades the prefix's contribution while leaving
 * the attention pattern it induces intact; K alone flattens that pattern. */
QI_API int qi_scale_prefix_kv(qi_ctx* c, int layer_lo, int layer_hi,
                              float k_scale, float v_scale);

/* Copy the live cache into slot `slot` (0..QI_PREFIX_SLOTS-1), and blend the
 * live cache towards a saved one:
 *     k = (1-alpha)*k + alpha*saved.k     (and likewise v)
 * The two must share a layout — same prefix length and target grid — which in
 * practice means two prompts that tokenize to the same length. */
enum { QI_PREFIX_SLOTS = 4 };
QI_API int qi_save_prefix(qi_ctx* c, int slot);
QI_API int qi_blend_prefix(qi_ctx* c, int slot, float alpha);

/* ── VAE (QI_LOAD_VAE) ─────────────────────────────────────────────────────
 * Both halves are loaded together — the encoder is ~200 MB against the DiT's
 * 7-14 GB, and having it resident is what makes the image-conditioned paths a
 * call rather than a reload. */

/* pixels: (3 * H * W) NCHW in [0, 1]; H and W multiples of qi_vae_scale().
 * An opaque alpha plane is appended internally (the autoencoder is RGBA on
 * both ends). out: (latent_channels * H/16 * W/16) NCHW, pipeline scale. */
QI_API int qi_encode_image(qi_ctx* c, const float* pixels, int H, int W,
                           float* out);

/* latent: (latent_channels * h_lat * w_lat) NCHW, raw pipeline scale (the
 * per-channel denormalisation happens inside). out: (3 * 16*h_lat * 16*w_lat)
 * NCHW in [-1, 1], alpha dropped. */
QI_API int qi_decode(qi_ctx* c, const float* latent, int h_lat, int w_lat,
                     float* out);

/* ── utilities (no ctx needed) ──────────────────────────────────────────── */

/* brotensor's Philox N(0,1) stream — bit-identical to the pipeline's noise.
 * The initial latent of a generation seeded S is qi_randn(S, 0, n). */
QI_API int qi_randn(uint64_t key, uint64_t counter, int64_t n, float* out);

QI_API int qi_mem_info(uint64_t* free_bytes, uint64_t* total_bytes);
QI_API int qi_mem_trim(void);

#ifdef __cplusplus
}
#endif
