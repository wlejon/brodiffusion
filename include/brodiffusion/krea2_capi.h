#pragma once

/* krea2_capi.h — flat C API over the Krea 2 components, for research bindings.
 *
 * Built as the `krea2_capi` shared library (krea2_capi.dll) and consumed from
 * Python via ctypes (D:/projects/krea-research). The surface wraps the three
 * Krea 2 components INDIVIDUALLY — Qwen3-VL text encoder, the flow DiT, the
 * Qwen-Image VAE decoder — so research code can load only what a question
 * needs, inject at the component boundaries, and drive the step loop itself
 * (Euler flow matching and the x0-preview identity are host-side math).
 *
 * Conventions:
 *   - Every array crosses the boundary as caller-allocated FP32, C-contiguous,
 *     row-major, shapes fixed by the model facts below. Internals cast to the
 *     compute dtype (BF16 on CUDA).
 *   - Functions return 0 on success, -1 on failure (k2_last_error() has the
 *     message; per-thread). Functions that return a count return -1 on failure.
 *   - One k2_ctx per model dir; not thread-safe (drive from one thread).
 */

#include <stdint.h>

#if defined(_WIN32)
#  if defined(K2_CAPI_BUILD)
#    define K2_API __declspec(dllexport)
#  else
#    define K2_API __declspec(dllimport)
#  endif
#else
#  define K2_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct k2_ctx k2_ctx;

enum {
    K2_LOAD_TE  = 1,   /* Qwen3-VL-4B text encoder + tokenizer */
    K2_LOAD_DIT = 2,   /* the 12.9B flow transformer (incl. text_fusion) */
    K2_LOAD_VAE = 4,   /* Qwen-Image VAE decoder */
    K2_LOAD_ALL = 7
};

/* Message for the most recent failure on this thread ("" if none). */
K2_API const char* k2_last_error(void);

/* Open a Krea 2 model directory (the diffusers layout with model_index.json)
 * and load the requested components. quantize != 0 loads the DiT body and TE
 * INT8 weight-only (the fit-on-24GB configuration). NULL on failure. */
K2_API k2_ctx* k2_open(const char* model_dir, int components, int quantize);
K2_API void    k2_close(k2_ctx* c);

/* ── model facts (valid after any successful k2_open) ─────────────────── */
K2_API int k2_hidden_size(const k2_ctx* c);      /* 6144 */
K2_API int k2_text_hidden_dim(const k2_ctx* c);  /* 2560 */
K2_API int k2_num_text_layers(const k2_ctx* c);  /* 12   */
K2_API int k2_max_text_seq(const k2_ctx* c);     /* 512  */
K2_API int k2_in_channels(const k2_ctx* c);      /* 64 = latent_channels*4 */
K2_API int k2_latent_channels(const k2_ctx* c);  /* 16   */

/* ── text encoder (K2_LOAD_TE) ────────────────────────────────────────────
 * Prompt -> the tapped-depth conditioning stack.
 *   embeds_out: (max_text_seq * num_text_layers, text_hidden_dim)
 *               token-major / layer-minor: row t*12 + l is token t's tap of
 *               Qwen3-VL layer {2,5,8,...,35}[l].
 *   mask_out:   (max_text_seq) 1.0 = valid token, 0.0 = filler. */
K2_API int k2_encode_prompt(k2_ctx* c, const char* prompt,
                            float* embeds_out, float* mask_out);

/* ── DiT (K2_LOAD_DIT) ──────────────────────────────────────────────────── */

/* text_fusion + txt_in over a tapped stack (the timestep-independent half).
 * embeds/mask as k2_encode_prompt produces them — or modified: this is the
 * injection point for per-layer tap and mixture experiments. txt_out must
 * hold (max_text_seq, hidden_size); returns n_valid, the rows written. */
K2_API int k2_encode_text(k2_ctx* c, const float* embeds, const float* mask,
                          float* txt_out);

/* One transformer forward: flow-matching velocity for the image tokens.
 *   packed: (hp*wp, in_channels) packed noisy latent tokens.
 *   txt:    (n_txt, hidden_size) text rows — normally k2_encode_text output,
 *           or any injected replacement.
 *   timestep in [0,1] (sigma).  out: (hp*wp, in_channels). */
K2_API int k2_forward(k2_ctx* c, const float* packed, int hp, int wp,
                      const float* txt, int n_txt, float timestep,
                      float* out);

/* ── VAE (K2_LOAD_VAE) ────────────────────────────────────────────────────
 * latent: (latent_channels * h_lat * w_lat) NCHW plane-major, raw pipeline
 * scale. out: (3 * 8*h_lat * 8*w_lat) NCHW in [-1, 1]. */
K2_API int k2_decode(k2_ctx* c, const float* latent, int h_lat, int w_lat,
                     float* out);

/* ── utilities (no ctx needed) ──────────────────────────────────────────── */

/* brotensor's Philox N(0,1) stream — bit-identical to the pipeline's noise.
 * The initial latent of a generation seeded S is k2_randn(S, 0, n). */
K2_API int k2_randn(uint64_t key, uint64_t counter, int64_t n, float* out);

K2_API int k2_mem_info(uint64_t* free_bytes, uint64_t* total_bytes);
K2_API int k2_mem_trim(void);

#ifdef __cplusplus
}
#endif
