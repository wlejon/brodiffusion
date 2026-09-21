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
 *
 * Multi-slot hooks. Every ranged hook below holds an ORDERED LIST of
 * bindings, all of which apply to the same forward:
 *
 *     qi_set_*    replace the list with this one binding. A NULL tensor, an
 *                 identity scale or an empty range clears it instead.
 *     qi_add_*    append a binding; returns its index, or -1.
 *     qi_clear_*  empty the list.
 *     qi_*_count  how many bindings are armed.
 *
 * For a block covered by several, deltas ADD and scales / masks MULTIPLY. The
 * composition happens at bind time — the model folds every covering binding
 * into one finished (1, hidden) row per distinct coverage over the 32 blocks —
 * so an arbitrary number of bindings costs the forward nothing per token.
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

/* Which sublayer's gated residual a qi_set_gate_mask scales. Masking both is
 * the blunt form: the MLP half drags a late-step edit's retention from 100%+
 * of its step-0 effect down to ~30%, so "restyle this region at step 6"
 * wants QI_GATE_ATTN. */
enum {
    QI_GATE_BOTH = 0,
    QI_GATE_ATTN = 1,
    QI_GATE_MLP  = 2
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

/* ── image-conditioned prompt encoding (QI_LOAD_TE) ────────────────────────
 * The same two-call protocol, with condition images: the template reserves a
 * run of rows per image and the Qwen3-VL vision tower's output stands in
 * their place. Returns n_valid (which now COUNTS the image rows), or -1.
 *
 * `image_paths` are decoded and resized to
 * calculate_dimensions(output_resolution², aspect) — each side a multiple of
 * 32 — so the vision grid and the autoencoder's latent grid describe the same
 * picture. Pass 0 for output_resolution to take the default 1024.
 *
 * The three getters below say where the images landed. Driving a forward from
 * here means:
 *   1. qi_get_prompt_embeds -> (n_valid, text_hidden_dim)
 *   2. drop the rows qi_get_prompt_pad_mask marks, and run qi_encode_text
 *      over what is left — txt_in is row-independent, and the DiT fills those
 *      rows from the condition LATENTS (through a different projection),
 *      not from the encoder's output
 *   3. qi_encode_image each condition image at the grid
 *      qi_get_prompt_image_grid reports, concatenate the results in template
 *      order, and hand them to qi_set_condition_latents
 *   4. qi_forward as usual — it reads the armed prefix. */
QI_API int qi_encode_prompt_images(qi_ctx* c, const char* prompt,
                                   const char* const* image_paths,
                                   int n_images, int output_resolution);

/* image_pad_mask: n_valid int32s, 1 at a condition-image slot. All zero after
 * a plain qi_encode_prompt. */
QI_API int qi_get_prompt_pad_mask(qi_ctx* c, int32_t* out);

/* How many condition images the parked prompt carries (0 for text-only), and
 * the latent grid the i-th one needs from the autoencoder. Each image
 * occupies h_lat*w_lat latent tokens and four times fewer encoder rows. */
QI_API int qi_prompt_num_images(qi_ctx* c);
QI_API int qi_get_prompt_image_grid(qi_ctx* c, int index, int* h_lat,
                                    int* w_lat);

/* Arm the condition latents for every later qi_forward: (n_tokens,
 * latent_channels) in template order, already at pipeline scale (what
 * qi_encode_image returns, transposed to token-major). The prefix layout
 * comes from the parked prompt's image runs, so encode the prompt first.
 * latents == NULL clears, returning to text-to-image. Resets the prefix KV
 * cache either way. */
QI_API int qi_set_condition_latents(qi_ctx* c, const float* latents,
                                    int n_tokens);

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
QI_API int qi_add_mod_delta(qi_ctx* c, const float* delta, int block_lo,
                            int block_hi, int target);
QI_API int qi_clear_mod_deltas(qi_ctx* c);
QI_API int qi_mod_delta_count(qi_ctx* c);

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
QI_API int qi_add_gate_scale(qi_ctx* c, float attn_scale, float mlp_scale,
                             float txt_scale, float img_scale, int block_lo,
                             int block_hi);

/* The same four dials, set INDEPENDENTLY instead of as the rank-1 product
 * above: one multiplier per (sublayer x row set). The rank-1 form cannot say
 * "the attention gate, on the image rows only" — raising attn_scale raises
 * the prefix product too, which re-extracts the cache (+13%/step) and used
 * to discard a live qi_scale_prefix_kv edit. attn_img alone leaves the
 * prefix product at 1, so nothing re-extracts. Only attn_txt / mlp_txt need
 * qi_reset_cache(); it is issued for you. */
QI_API int qi_set_gate_scale_rows(qi_ctx* c, float attn_txt, float attn_img,
                                  float mlp_txt, float mlp_img, int block_lo,
                                  int block_hi);
QI_API int qi_add_gate_scale_rows(qi_ctx* c, float attn_txt, float attn_img,
                                  float mlp_txt, float mlp_img, int block_lo,
                                  int block_hi);
QI_API int qi_clear_gate_scales(qi_ctx* c);
QI_API int qi_gate_scale_count(qi_ctx* c);

/* Post-tanh gate delta: add delta (2 * hidden_size, laid out [attn, mlp]) to
 * the EFFECTIVE gate of blocks [block_lo, block_hi) on the row named by
 * `target` (QI_MOD_*), i.e.
 *     g_eff = qi_set_gate_scale factor * tanh(gate) + delta
 * Unlike qi_set_mod_delta's gate chunks, which land before the tanh where
 * most of gate2's channels are saturated, this has unit authority over every
 * channel. delta == NULL clears. A QI_MOD_PREFIX / QI_MOD_BOTH delta needs
 * qi_reset_cache() to take effect. */
QI_API int qi_set_gate_delta(qi_ctx* c, const float* delta, int block_lo,
                             int block_hi, int target);
QI_API int qi_add_gate_delta(qi_ctx* c, const float* delta, int block_lo,
                             int block_hi, int target);
QI_API int qi_clear_gate_deltas(qi_ctx* c);
QI_API int qi_gate_delta_count(qi_ctx* c);

/* Per-token gate mask: the gated residual of the sublayer(s) named by `which`
 * (QI_GATE_*) for row r of blocks [block_lo, block_hi) is multiplied by
 * mask[r], after the tanh and after any qi_set_gate_scale. `n` must equal the
 * forward's n_txt + h_lat*w_lat; a forward with a different joint length now
 * FAILS with the expected length in qi_last_error() rather than silently
 * doing nothing. NULL clears. Needs qi_reset_cache() for its prefix half to
 * land; it is issued for you. */
QI_API int qi_set_gate_mask(qi_ctx* c, const float* mask, int64_t n,
                            int block_lo, int block_hi, int which);
QI_API int qi_add_gate_mask(qi_ctx* c, const float* mask, int64_t n,
                            int block_lo, int block_hi, int which);
QI_API int qi_clear_gate_masks(qi_ctx* c);
QI_API int qi_gate_mask_count(qi_ctx* c);

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
 * re-encoding anything. Both take effect on the very next qi_forward, with no
 * re-extraction. */

/* Multiply the cached K of layers [layer_lo, layer_hi) by k_scale and V by
 * v_scale. Attenuating V alone fades the prefix's contribution while leaving
 * the attention pattern it induces intact; K alone flattens that pattern.
 *
 * This is an idempotent DIAL on the model, not a multiply into the cache: the
 * factor is applied where a cached forward READS the cache, so setting it
 * twice means the same thing as setting it once, it works through a whole
 * step loop, and it survives qi_reset_cache(). It needs nothing to have been
 * extracted first, and 1/1 over the full layer range clears it.
 *
 * row_scale, when non-NULL, holds `n_rows` WEIGHTS — one per PREFIX ROW —
 * saying how much of k_scale/v_scale each row gets:
 *     k_row[r] = 1 + row_scale[r] * (k_scale - 1)
 * All ones is the broadcast, all zeros the identity. That is per-token
 * prompt weighting over the cache with no re-encode: set 1 on one phrase's
 * rows and 0 on the rest, and only that phrase is attenuated. n_rows must
 * equal the cached prefix length or the next qi_forward fails.
 * row_scale == NULL (n_rows ignored) is every row. */
QI_API int qi_scale_prefix_kv(qi_ctx* c, int layer_lo, int layer_hi,
                              float k_scale, float v_scale,
                              const float* row_scale, int64_t n_rows);
QI_API int qi_add_prefix_kv_scale(qi_ctx* c, int layer_lo, int layer_hi,
                                  float k_scale, float v_scale,
                                  const float* row_scale, int64_t n_rows);
QI_API int qi_clear_prefix_kv_scales(qi_ctx* c);
QI_API int qi_prefix_kv_scale_count(qi_ctx* c);

/* Copy the live cache into slot `slot` (0..QI_PREFIX_SLOTS-1), and blend the
 * live cache towards a saved one:
 *     k = (1-alpha)*k + alpha*saved.k     (and likewise v)
 * The two must share a layout — same prefix length and target grid — which in
 * practice means two prompts that tokenize to the same length.
 * qi_clear_prefix_slots drops every saved cache and trims the allocator; they
 * hold real VRAM. qi_prefix_slot_valid returns 1/0 and never fails. */
enum { QI_PREFIX_SLOTS = 4 };
QI_API int qi_save_prefix(qi_ctx* c, int slot);
QI_API int qi_blend_prefix(qi_ctx* c, int slot, float alpha);
QI_API int qi_clear_prefix_slots(qi_ctx* c);
QI_API int qi_prefix_slot_valid(qi_ctx* c, int slot);

/* ── conditioning control axes and between-step schedules ─────────────────
 *
 * The axes are BCD1 dictionary directions in the text encoder's 4096-d space
 * (see cond_control.h), and the injected vector is alpha * scale * dir added
 * to every row of the (n, 4096) conditioning. Applying that once before step 0
 * is the whole of the static "desk"; a SCHEDULE re-applies it with a per-step
 * alpha instead, which is what the denoise loop needed and could not express.
 *
 * On this API the caller drives the loop, so the schedule is two calls:
 *
 *     qi_encode_text(c, embeds, n, txt)     // parks `embeds` as the BASE
 *     for (s = 0; s < steps; s++) {
 *         if (qi_control_step(c, s, txt) > 0) { ... }   // rebuilds txt
 *         qi_forward(c, latent, h, w, txt, n_txt, t, out);
 *     }
 *
 * qi_control_step rebuilds txt as txt_in(base + the step's stack) and drops
 * the prefix cache, returning the new row count — or 0, meaning this step's
 * alpha is the one `txt` already carries and nothing was written or reset. A
 * flat schedule therefore pays for one re-extract, not one per step.
 *
 * Ranges are HALF-OPEN [lo_step, hi_step), hi_step < 0 means "to the end",
 * and `alpha` is indexed by the ABSOLUTE step index, not by the offset from
 * lo_step. The _dir forms take the direction outright, for an axis that is in
 * no bank (a diff-of-means built from qi_get_prompt_embeds, say). */

/* Load a BCD1 dictionary. merge != 0 adds its axes to those already loaded
 * (same-named axes overwritten) instead of replacing them. */
QI_API int qi_load_control_dictionary(qi_ctx* c, const char* path, int merge);
/* How many axes are loaded, and axis `index`'s name into `out` (NUL
 * terminated, truncated to `cap`). qi_control_axis_name returns the full name
 * length, or -1. */
QI_API int qi_control_axis_count(qi_ctx* c);
QI_API int qi_control_axis_name(qi_ctx* c, int index, char* out, int cap);
/* Axis `name`'s stored direction (qi_text_hidden_dim() floats) and its baked
 * scale; either pointer may be NULL. */
QI_API int qi_control_axis_vector(qi_ctx* c, const char* name, float* dir_out,
                                  float* scale_out);

/* Arm a schedule. qi_set_* replaces the list with this one, qi_add_* appends
 * and returns the slot index; every armed slot contributes to the same step.
 * `alpha` is n_alpha floats. Returns the slot index, or -1. */
QI_API int qi_set_control_schedule(qi_ctx* c, const char* name,
                                   const float* alpha, int n_alpha,
                                   int lo_step, int hi_step);
QI_API int qi_add_control_schedule(qi_ctx* c, const char* name,
                                   const float* alpha, int n_alpha,
                                   int lo_step, int hi_step);
QI_API int qi_set_control_schedule_dir(qi_ctx* c, const float* dir, int dim,
                                       float scale, const float* alpha,
                                       int n_alpha, int lo_step, int hi_step);
QI_API int qi_add_control_schedule_dir(qi_ctx* c, const float* dir, int dim,
                                       float scale, const float* alpha,
                                       int n_alpha, int lo_step, int hi_step);
QI_API int qi_clear_control_schedules(qi_ctx* c);
QI_API int qi_control_schedule_count(qi_ctx* c);

/* Apply the armed schedules for step `step`. Rewrites `txt_out` (which must
 * hold n_valid * hidden_size floats — the same buffer qi_encode_text filled)
 * and drops the prefix cache. Returns the row count when it rebuilt, 0 when
 * this step's stack is already the one the rows carry, -1 on failure. */
QI_API int qi_control_step(qi_ctx* c, int step, float* txt_out);

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
