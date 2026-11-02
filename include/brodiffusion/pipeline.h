#pragma once

// Stable Diffusion 1.5 text-to-image pipeline.
//
// Owns the tokenizer, CLIP text encoder, UNet, VAE decoder, and DDIM
// scheduler — wires them into a single `generate(prompt, ...)` call that
// produces a raw FP32 RGB image in [-1, 1] (the VAE output range). Callers
// clamp + rescale to uint8 / encode to PNG outside this library.
//
// Inference-only, batch size N = 1. Runs on whichever backend brotensor
// resolves at runtime — CPU by default, CUDA when a GPU is available — at
// that backend's compute dtype (FP32 on CPU, FP16 on a GPU). Classifier-free
// guidance runs the U-Net twice per step (conditional + unconditional)
// rather than batching, since the rest of the inference stack is N=1.

#include "brodiffusion/pipeline_types.h"

namespace brodiffusion::pipeline {

// LoadCancelled (thrown by from_model_dir when ModelDirOptions::should_cancel
// fires) lives in denoiser.h — the sharded DiT loaders that throw it sit below
// this header and only include denoiser.h.

class Pipeline {
public:
    // StableDiffusion constructor: builds a UNet denoiser. Valid only when
    // cfg.model_class == ModelClass::StableDiffusion.
    Pipeline(const PipelineConfig& cfg, brolm::clip::Tokenizer tokenizer);

    // Flux constructor: builds a FluxDenoiser plus the second (T5) text
    // encoder + tokenizer. Valid only when cfg.model_class == ModelClass::Flux.
    Pipeline(const PipelineConfig& cfg, brolm::clip::Tokenizer clip_tok,
             brolm::t5::Tokenizer t5_tok);

    // Sana constructor: builds a SanaDenoiser (linear DiT) + the DC-AE f32c32
    // decoder + the Gemma-2 text encoder, and owns the Gemma tokenizer. Valid
    // only when cfg.model_class == ModelClass::Sana. The CLIP / KL-VAE members
    // are default-constructed and unused.
    Pipeline(const PipelineConfig& cfg, brolm::gemma::Tokenizer gemma_tok);

    // PixArt constructor: builds a PixArtDenoiser (DiT) + the KL-VAE decoder +
    // the T5-XXL text encoder, and owns the T5 tokenizer. Valid only when
    // cfg.model_class == ModelClass::PixArt. There is no CLIP frontend (the
    // CLIP tokenizer/encoder members stay default-constructed and unused).
    Pipeline(const PipelineConfig& cfg, brolm::t5::Tokenizer t5_tok);

    // Qwen3-VL-conditioned constructor, shared by the two model families whose
    // text frontend is a Qwen3-VL backbone:
    //
    //   Krea 2       — Krea2Denoiser (single-stream flow DiT) + the Qwen-Image
    //                  VAE decoder + the Qwen3-VL-4B text encoder and its
    //                  vision tower (for image-as-prompt).
    //   QwenImage21  — QwenImage21Denoiser (block-causal single-stream DiT) +
    //                  the 16x RGBA Qwen-Image 2.1 VAE (decoder + encoder) +
    //                  the Qwen3-VL-8B text encoder and its vision tower
    //                  (which the image-conditioned edit path reads).
    //
    // Owns the Qwen3-VL tokenizer either way. Valid only when cfg.model_class
    // is Krea2 or QwenImage21; no CLIP frontend / KL-VAE (those members stay
    // default-constructed).
    Pipeline(const PipelineConfig& cfg, brolm::qwen3vl::Tokenizer qwen3vl_tok);

    // Build a fully-loaded Pipeline from a diffusers model directory: reads the
    // JSON configs, constructs the right sub-modules, loads all component
    // weights and tokenizers. Supports StableDiffusion and Flux model
    // directories. Pipeline is move-only; this returns by value.
    using ModelDirOptions = PipelineModelDirOptions;
    static Pipeline from_model_dir(const std::string& model_dir, const ModelDirOptions& opts);
    // Overload instead of a defaulted `opts = {}` argument: GCC rejects the
    // brace-init default because ModelDirOptions has an in-class member
    // initializer (default member init needed outside a member function).
    static Pipeline from_model_dir(const std::string& model_dir) {
        return from_model_dir(model_dir, ModelDirOptions{});
    }

    // Krea 2 only: reload JUST the Qwen3-VL-4B text backbone in place, leaving
    // the (25 GB) DiT, the VAE and the vision tower resident and untouched.
    // The backbone is reconstructed from the pipeline's stored config (a clean
    // slate — no stale quant/dense weight slots) and its language-model weights
    // are reloaded from `text_encoder_path` (a .gguf, or a diffusers
    // safetensors file/dir), or from `<model_dir>/text_encoder` when the path
    // is empty. `quantize` matches ModelDirOptions::quantize (ignored on the
    // gguf path, which keeps its own quant). Cross-attention conditioning is
    // rebuilt on the next prime()/generate(), so no cache flush is needed.
    // This is the cheap swap for comparing text encoders without re-reading
    // the whole ~26 GB checkpoint. Throws if this is not a Krea 2 pipeline.
    void reload_krea2_text_encoder(const std::string& model_dir,
                                   const std::string& text_encoder_path, bool quantize);

    // Move-only (owns move-only sub-modules; copies make no sense). The
    // move members and destructor are defined in pipeline.cpp where
    // StepGraphSession is complete.
    ~Pipeline();
    Pipeline(Pipeline&&) noexcept;
    Pipeline& operator=(Pipeline&&) noexcept;
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    // Load every sub-module's weights from a single safetensors file using
    // the prefixes used by an SD1.5 full checkpoint:
    //   "cond_stage_model.transformer.text_model." for CLIP
    //   "model.diffusion_model."                   for the U-Net
    //   "first_stage_model.decoder."               for the VAE decoder
    void load_weights(const brotensor::safetensors::File& f);

    // Load with explicit prefixes (e.g. when sub-modules ship in separate
    // diffusers exports).
    void load_weights(const brotensor::safetensors::File& f, const std::string& text_prefix,
                      const std::string& unet_prefix, const std::string& vae_prefix);

    // Load each module from its own diffusers-format safetensors file. Uses
    // the diffusers default prefixes ("text_model.", "", "decoder.").
    void load_weights(const brotensor::safetensors::File& text_file,
                      const brotensor::safetensors::File& unet_file,
                      const brotensor::safetensors::File& vae_file);

    // Apply a LoRA file.
    //
    // SD1.5: merges the deltas into the base UNet and CLIP weights.
    // Krea 2: attaches the file as ONE runtime-adapter group on the DiT (the
    // base linears may be INT8-quantized, so nothing is merged; each adapted
    // linear adds scale * (x @ down^T) @ up^T per forward). Runtime groups
    // are indexed in apply order — see set_lora_scale() / clear_loras().
    //
    // Must be called *after* load_weights() and *before* generate(). The
    // cross-attention K/V cache is re-primed inside every generate() call,
    // so calling apply_lora() between generates is safe — the next generate
    // will rebuild the cache against the updated K/V projections. For Krea 2
    // the fused text conditioning is rebuilt at every prime(), so a LoRA on
    // the text-fusion blocks likewise lands on the next generation.
    //
    // `scale` is a user multiplier applied on top of the per-LoRA alpha/rank
    // factor (default 1.0 = use as-shipped). Negative values are allowed
    // (subtract / undo). May be called more than once to stack multiple
    // LoRAs.
    //
    // Accepts kohya-ss/A1111 and diffusers/PEFT key conventions for SD1.5,
    // plus the Krea2-style DiT conventions (transformer. / diffusion_model. /
    // kohya-mangled transformer_blocks) — see lora.h. The format is
    // auto-detected from the key prefixes. Throws if the file contains LoRA
    // tensors that don't map to a target of the ACTIVE model family.
    //
    // Returns the runtime-adapter group index (Krea 2; pass it to
    // set_lora_scale), or -1 when the LoRA was merged in place (SD1.5).
    int apply_lora(const brotensor::safetensors::File& f, float scale = 1.0f);

    // Runtime-adapter LoRA controls — Krea 2 only (SD1.5 LoRAs are merged
    // irreversibly; these throw for other model classes). `index` is the
    // apply_lora() call order, 0-based.
    void set_lora_scale(int index, float scale);
    void clear_loras();
    int  num_loras() const;

    // Register a ControlNet safetensors file. The returned index is the
    // addressing key used elsewhere (remove_controlnet, plus the position
    // into GenerateOptions::controls). SD1.5 only; throws on Flux. The
    // default config matches HF's lllyasviel/sd-controlnet-* zoo; pass an
    // explicit ControlNetConfig if a non-default checkpoint shape is needed.
    int add_controlnet(const brotensor::safetensors::File& f);
    int add_controlnet(const brotensor::safetensors::File& f,
                       const controlnet::ControlNetConfig& cfg);

    // Drop one registered ControlNet by index. Subsequent indices shift down.
    void remove_controlnet(int index);

    // Drop all registered ControlNets.
    void clear_controlnets();

    // Number of currently registered ControlNets.
    int num_controlnets() const { return static_cast<int>(controlnets_.size()); }

    // True iff at least one ControlNet has been loaded.
    bool has_controlnet() const { return !controlnets_.empty(); }

    // Generate an image. Returns a freshly-allocated host buffer of
    // 3 * height * width FP32 values in NCHW (C=3, [-1, 1]).
    //
    // Internally: prime() → loop(step_once) → decode(). Bit-equivalent to
    // calling those three primitives directly.
    std::vector<float> generate(std::string_view prompt, const GenerateOptions& opts);

    // ── Step-wise API (for cross-attention tree search and similar) ───────
    //
    // prime():        encode prompt(s), build xattn caches, allocate initial
    //                 latent noise. Returns a state with step_index=0.
    // step_once():    advance one denoising step (mutates state). If
    //                 trace_out is non-null, the UNet forward runs in trace
    //                 mode (no K/V cache reuse, no INT8 quantization) and
    //                 fills the attention-map trace. Throws if state is
    //                 already at n_steps.
    // decode():       VAE-decode a state's latent to an FP32 host buffer
    //                 (same shape and units as generate()).
    //
    // The xattn cache rides the returned state (state.prepared) and is shared
    // across all branched states cloned from it; states from separate prime()
    // calls each keep their own, so they can be stepped interleaved (the
    // spatial-paint dual-state loop). Calling apply_lora() between prime()
    // and step_once() leaves the cache stale — re-prime() to refresh.
    PipelineState prime(std::string_view prompt, const GenerateOptions& opts);
    // If `attn_logit_biases` is non-null, trace mode is used and the bias
    // vector is forwarded to the active denoiser's forward_traced (length must
    // equal num_xattn_blocks(); per-entry nulls allowed for no-bias blocks).
    // When biases are supplied, trace_out may still be null — the trace is
    // computed but discarded. Trace mode requires a denoiser whose
    // num_xattn_blocks() > 0.
    void step_once(PipelineState& state, const GenerateOptions& opts,
                   AttentionTrace* trace_out = nullptr,
                   const std::vector<const brotensor::Tensor*>* attn_logit_biases = nullptr);
    std::vector<float> decode(const PipelineState& state);

    // The sigma schedule of the most recent prime()/generate() when the
    // active scheduler is FlowMatch: length num_inference_steps + 1 with a
    // trailing 0.0 (see FlowMatch::sigmas()). Empty for non-flow-match
    // schedulers or before any schedule has been set. Lets a step-wise
    // caller compute the x0 preview x_i - sigmas[i]*v from consecutive
    // latents without re-running the model.
    std::vector<float> schedule_sigmas() const;

    // Accessors for research callers (e.g. tree search needs xattn block count
    // to size the bias vector). Throws if the active denoiser is not a UNet.
    const unet::UNet& unet() const;

    // Denoiser-generic count of traceable / steerable cross-attention blocks
    // for the active denoiser (16 for the SD1.5 UNet, 57 for the Flux DiT,
    // 0 for a denoiser with no trace support). Lets a caller size an
    // attn_logit_biases vector or a trace array without knowing the denoiser
    // type. Always valid — never throws.
    int num_xattn_blocks() const { return denoiser_->num_xattn_blocks(); }

    // Latent→pixel upscale factor of the active autoencoder: 8 for the SD /
    // Flux / Krea 2 KL-VAE, 32 for Sana's DC-AE f32c32, 16 for Qwen-Image
    // 2.1's residual RGBA VAE. A decoded image is therefore
    // (state.H_lat * vae_scale_factor()) × (state.W_lat * vae_scale_factor())
    // pixels — the dimensions of decode()'s buffer. Always valid; never throws.
    int vae_scale_factor() const {
        if (model_class_ == ModelClass::Sana) return 32;
        if (model_class_ == ModelClass::QwenImage21) return 16;
        return 8;
    }

    const PipelineConfig& config() const { return cfg_; }

    // Conditioning-space control axes (the sana-research seam). Load a
    // dictionary, set per-axis weights, and every subsequent prime() injects
    // the weighted directions into the positive text conditioning before the
    // denoiser. Empty / all-zero by default (no effect). See cond_control.h.
    CondControl&       cond_control()       { return cond_control_; }
    const CondControl& cond_control() const { return cond_control_; }

    // Encode `prompt` into the model's text-conditioning sequence — the exact
    // (L, hidden) embeddings the denoiser cross-attends to (row 0 = BOS). This
    // is the same encode prime() runs, exposed so callers can build control
    // directions in the encoder's own space (e.g. a diff-of-means axis from two
    // phrase sets). Routes by model class: Gemma-2 for Sana, CLIP otherwise.
    // Requires weights to be loaded.
    brotensor::Tensor encode_conditioning(std::string_view prompt);

    // ── Reference-attention identity anchor (the frame/sana-research seam) ──
    //
    // Sana's self-attention is linear, so a reference image's appearance can be
    // injected training-free: capture a neutral anchor's per-(branch,step,block)
    // attention summaries during one denoise, then add them (scaled by a weight)
    // into every later generation. The anchor carries visual IDENTITY; each
    // generation's own prompt still sets pose / expression. This lets a steered
    // attribute (e.g. an emotion cond_control axis pushed to the extreme) move
    // freely while the subject stays the same person — text-carries-structure,
    // image-carries-identity, no training and no clamp. Sana-only (throws on
    // other model classes); see dit::SanaDenoiser's ref_* methods.
    //
    // capture_identity_anchor() runs one full generation of `prompt` with the
    // denoiser recording, returns that anchor image, and arms the seam. The
    // captured step count should match later generations' num_inference_steps
    // for tight t-alignment (a shorter run reuses the anchor's final step).
    std::vector<float> capture_identity_anchor(std::string_view prompt,
                                               const GenerateOptions& opts);
    // Injection strength. 0 (default) disables injection even with an anchor
    // armed; ~1 reproduces the reference faithfully; higher over-anchors. Takes
    // effect on the next generate() / prime().
    void  set_identity_weight(float weight);
    float identity_weight() const;
    // True once an anchor has been captured (until clear_identity_anchor()).
    bool  has_identity_anchor() const;
    // Drop the cached anchor and zero the weight (frees the summary cache).
    void  clear_identity_anchor();

    // ── Krea 2 research hooks (the krea2_capi / krea-research seam) ─────────
    //
    // Direct forwarding to dit::Krea2Transformer2DModel's research hooks (see
    // dit/krea2.h for full semantics of each). Krea2-only; every method below
    // throws if model_class_ != Krea2. These mirror the Sana identity-anchor
    // methods above in spirit (denoiser-specific research plumbing exposed at
    // the Pipeline level) but are independent of that seam.

    // AdaLN mod-delta: add `delta` (1, 6*krea_hidden_size()) to the shared
    // per-step modulation for body blocks [block_lo, block_hi) on every
    // subsequent step_once(). Empty tensor clears.
    void krea_set_mod_delta(const brotensor::Tensor& delta, int block_lo, int block_hi);

    // Timestep-embedding readout at `timestep` (the SAME 0..1000-scale value
    // krea_step_timestep() returns / step_once() consumes internally — this
    // method does the flow-time (/1000) conversion itself). No forward pass.
    // temb_out: (1, krea_hidden_size()); mod_out: (1, 6*krea_hidden_size()).
    void krea_time_mod(float timestep, brotensor::Tensor& temb_out, brotensor::Tensor& mod_out);

    // The active scheduler's timestep for `state.step_index` — the same
    // 0..1000-scale value step_once() feeds the denoiser internally. Lets a
    // caller build a krea_time_mod() query / mod-delta for the step about to
    // run. Krea2 pairs exclusively with the FlowMatch scheduler.
    float krea_step_timestep(const PipelineState& state) const;

    // Attention-gate scale: scale the sigmoid output gate of body-block
    // attention, text rows and image rows separately, for blocks
    // [block_lo, block_hi). (1, 1) clears.
    void krea_set_gate_scale(float txt_scale, float img_scale, int block_lo, int block_hi);

    // Per-token gate mask over blocks [block_lo, block_hi); mask holds
    // text_seq + img_len values in forward order. Empty tensor clears.
    void krea_set_gate_mask(const brotensor::Tensor& mask, int block_lo, int block_hi);

    // Gate activity capture. When enabled, every subsequent step_once()
    // overwrites the internal sink; krea_gates() reads it back, row-major
    // (krea_num_layers(), text_seq + img_len).
    void krea_capture_gates(bool enable);
    std::vector<float> krea_gates() const;

    // Sizing accessors so a caller can allocate buffers without hardcoding
    // Krea 2's constants (6144 / 28).
    int krea_hidden_size() const;
    int krea_num_layers() const;

    // ── Krea 2 raw-taps entry points (band dial / image-as-prompt seam) ────
    //
    // Krea2's conditioning pipeline exposes two independently-callable
    // stages: raw per-layer Qwen3-VL taps (krea_encode_prompt_taps) and the
    // fusion stack that collapses them into the (n_valid, krea_hidden_size())
    // conditioning the DiT cross-attends to (krea_encode_text — the same
    // space cond_control axes are applied in, see cond_control()). Between
    // the two, a caller can edit specific tap rows (e.g. scale layers 7-10
    // for the "deep-band" literal<->stylized dial) or substitute an entirely
    // different tap source (e.g. image tokens through the same Qwen3-VL
    // backbone — see krea_encode_image_prompt). krea_prime_from_taps() then
    // primes a step-wise generation from the (possibly edited) raw taps,
    // exactly as prime() does internally for a plain text prompt.
    krea2::TextConditioning krea_encode_prompt_taps(std::string_view prompt);

    brotensor::Tensor krea_encode_text(const brotensor::Tensor& prompt_embeds,
                                       const brotensor::Tensor& prompt_embeds_mask);

    // Prime a step-wise generation from caller-supplied raw taps (as
    // returned by krea_encode_prompt_taps(), optionally edited) instead of
    // encoding `prompt` internally. uncond_embeds/uncond_mask may be null —
    // when guidance_scale != 1.0 (do_cfg) the uncond branch then falls back
    // to encoding opts.negative_prompt normally.
    PipelineState krea_prime_from_taps(const brotensor::Tensor& embeds,
                                       const brotensor::Tensor& mask,
                                       const brotensor::Tensor* uncond_embeds,
                                       const brotensor::Tensor* uncond_mask,
                                       const GenerateOptions& opts);

    // Krea 2 image-as-prompt: encode `pixels` (FP32 CHW, [0,1] range, shape
    // (3, H, W)) through Krea 2's own Qwen3-VL-4B vision tower into the SAME
    // raw-taps shape krea_encode_prompt_taps() produces for text — feed the
    // result straight into krea_encode_text()/krea_prime_from_taps(). No
    // separate model — the checkpoint's text encoder ships its vision tower
    // too (unused for plain text prompting).
    krea2::TextConditioning krea_encode_image_prompt(const float* pixels, int H, int W);

    // ── Qwen-Image 2.1 research hooks (the qwenimage21_capi seam) ─────────
    //
    // Direct forwarding to dit::QwenImage21Transformer2DModel's research
    // hooks (dit/qwenimage21.h carries the full semantics of each) plus the
    // conditioning entry points the model class needs. Every method below
    // throws if model_class_ != QwenImage21.
    //
    // Two things differ from the Krea 2 block above, both consequences of
    // 2.1's prefix KV cache:
    //   * a hook that touches the PREFIX (the t = 0 modulation row, the
    //     cached text K/V, the prepared text rows) only takes effect on an
    //     extract step, so these methods reset the most recently primed
    //     conditioning's caches for you — see qi21_reset_cache().
    //   * the conditioning is ONE (n, 4096) tensor of Qwen3-VL hidden states
    //     rather than a layer stack, so the control-axis machinery
    //     (cond_control(), loadControlDictionary / setControlVector) applies
    //     to it directly, before prepare(), exactly as it does for Sana.
    //
    // Every ranged hook holds an ORDERED LIST of bindings: qi21_set_* replaces
    // the list with one entry (or clears it), qi21_add_* appends, and
    // qi21_clear_* empties it. All the bindings in a list apply to the same
    // generation, composing per block the way each hook's semantics imply —
    // see dit/qwenimage21.h. Whether the cache needs re-extracting is decided
    // over the WHOLE list, so adding a prefix-side binding to a target-side
    // list resets the cache exactly once.

    // Modulation delta on blocks [block_lo, block_hi); `delta` is
    // (1, 4*qi21_hidden_size()) or empty to clear. `target` picks the target
    // row, the prefix row or both; a prefix-side delta resets the live prefix
    // cache so the next step re-extracts under it.
    void qi21_set_mod_delta(const brotensor::Tensor& delta, int block_lo, int block_hi,
                            dit::QwenImage21ModTarget target = dit::QwenImage21ModTarget::Target);
    int  qi21_add_mod_delta(const brotensor::Tensor& delta, int block_lo, int block_hi,
                            dit::QwenImage21ModTarget target = dit::QwenImage21ModTarget::Target);
    void qi21_clear_mod_deltas();
    int  qi21_mod_delta_count() const;

    // Timestep readout at `timestep` — the SAME 0..1000-scale value
    // qi21_step_timestep() returns and step_once() consumes; the flow-time
    // conversion happens here. temb_out: (2, hidden); mod_out: (2, 4*hidden),
    // row 0 = the sampled t, row 1 = t = 0.
    void qi21_time_mod(float timestep, brotensor::Tensor& temb_out, brotensor::Tensor& mod_out);

    // The active scheduler's timestep for `state.step_index`. Qwen-Image 2.1
    // pairs exclusively with the FlowMatch scheduler.
    float qi21_step_timestep(const PipelineState& state) const;

    // Post-tanh gate dials over blocks [block_lo, block_hi). The rank-1
    // sugar: attn/mlp pick the sublayer, txt/img the row class, and the
    // binding holds their four products. All four at 1 clears.
    void qi21_set_gate_scale(float attn_scale, float mlp_scale, float txt_scale, float img_scale,
                             int block_lo, int block_hi);
    int  qi21_add_gate_scale(float attn_scale, float mlp_scale, float txt_scale, float img_scale,
                             int block_lo, int block_hi);

    // The four multipliers, one per (sublayer x row set), set independently.
    // The rank-1 form cannot say "attention gate on the IMAGE rows only":
    // raising attn_scale raises the prefix product too, which re-extracts
    // the cache (+13%/step). attn_img alone leaves the prefix at 1, so the
    // cache and every armed prefix edit stay live.
    void qi21_set_gate_scale_rows(float attn_txt, float attn_img, float mlp_txt, float mlp_img,
                                  int block_lo, int block_hi);
    int  qi21_add_gate_scale_rows(float attn_txt, float attn_img, float mlp_txt, float mlp_img,
                                  int block_lo, int block_hi);
    void qi21_clear_gate_scales();
    int  qi21_gate_scale_count() const;

    // Post-tanh gate delta over blocks [block_lo, block_hi): `delta` is
    // (1, 2*qi21_hidden_size()) laid out [attn, mlp], added to the effective
    // gate AFTER the tanh (and after any qi21_set_gate_scale factor). Empty
    // clears. This is the responsive counterpart to qi21_set_mod_delta()'s
    // gate chunks, which land before the tanh where most of gate2's channels
    // are saturated — see dit/qwenimage21.h's set_gate_delta(). A prefix-side
    // delta resets the live prefix cache so the next step re-extracts.
    void qi21_set_gate_delta(const brotensor::Tensor& delta, int block_lo, int block_hi,
                             dit::QwenImage21ModTarget target = dit::QwenImage21ModTarget::Target);
    int  qi21_add_gate_delta(const brotensor::Tensor& delta, int block_lo, int block_hi,
                             dit::QwenImage21ModTarget target = dit::QwenImage21ModTarget::Target);
    void qi21_clear_gate_deltas();
    int  qi21_gate_delta_count() const;

    // Per-token gate mask over blocks [block_lo, block_hi); `mask` holds
    // prefix_len + img_len values in joint forward order. Empty clears. A
    // mask whose length is not the joint length THROWS on the next forward.
    //
    // `which` picks the sublayer the mask scales. Masking both is the blunt
    // form: the MLP half drags a late-step edit's retention from 100%+ down
    // to ~30%, so "restyle this region at step 6" wants Attn.
    void qi21_set_gate_mask(const brotensor::Tensor& mask, int block_lo, int block_hi,
                            dit::QwenImage21GateSublayer which = dit::QwenImage21GateSublayer::Both);
    int  qi21_add_gate_mask(const brotensor::Tensor& mask, int block_lo, int block_hi,
                            dit::QwenImage21GateSublayer which = dit::QwenImage21GateSublayer::Both);
    void qi21_clear_gate_masks();
    int  qi21_gate_mask_count() const;

    // Gate activity capture; qi21_gates() reads back the most recent step,
    // row-major (qi21_num_layers(), prefix_len + img_len). qi21_gates_mlp()
    // is the SwiGLU half of the same step, in the same layout — a second
    // reading, because the two sublayers carry independent gates, scales and
    // mask compositions.
    void qi21_capture_gates(bool enable);
    std::vector<float> qi21_gates() const;
    std::vector<float> qi21_gates_mlp() const;

    // norm_out scale delta, (1, qi21_hidden_size()) or empty to clear.
    void qi21_set_norm_out_scale_delta(const brotensor::Tensor& delta);

    // Sizing accessors, so nothing downstream hardcodes 4096 / 32 / 64 / 16.
    int qi21_hidden_size() const;
    int qi21_num_layers() const;
    int qi21_text_hidden_dim() const;
    int qi21_latent_channels() const;

    // ── prefix cache surface ──────────────────────────────────────────────

    // Drop both CFG branches' prefix KV caches on the most recently primed
    // conditioning, forcing the next step to re-extract. Needed after any
    // change to the prefix side — edited text rows, a prefix-side mod delta,
    // a control axis applied out of band. A no-op when nothing has been
    // primed yet (or the state that owned it has been destroyed).
    void qi21_reset_cache();

    // Attenuate the cached prefix K/V for layers [layer_lo, layer_hi): the
    // cached K of those layers is multiplied by k_scale and the V by v_scale
    // wherever a cached step reads them. Takes effect on the very next step,
    // with no re-extraction.
    //
    // This is an idempotent DIAL, not an operation on the cache: calling it
    // twice with the same value means the same thing as calling it once, it
    // behaves through generate() like every other hook, and it survives a
    // cache reset. (It used to multiply the live cache in place, which
    // compounded across steps and made it unusable from generate() at all.)
    // 1/1 over the full layer range clears; it needs no step to have run.
    //
    // `row_scale`, when non-empty, holds one WEIGHT per PREFIX ROW saying
    // how much of k_scale/v_scale that row gets:
    //     k_row[r] = 1 + row_scale[r] * (k_scale - 1)
    // All ones is the broadcast, all zeros the identity. That is per-token
    // prompt weighting over the cache, with no re-encode. Its length must
    // equal the cached prefix length or the next forward throws. Empty =
    // every row, i.e. the broadcast.
    void qi21_scale_prefix_kv(int layer_lo, int layer_hi, float k_scale, float v_scale,
                              const brotensor::Tensor& row_scale = {});
    int  qi21_add_prefix_kv_scale(int layer_lo, int layer_hi, float k_scale, float v_scale,
                                  const brotensor::Tensor& row_scale = {});
    void qi21_clear_prefix_kv_scales();
    int  qi21_prefix_kv_scale_count() const;

    // ── prefix cache slots ────────────────────────────────────────────────
    //
    // N deep copies of an extracted prefix, so the conditioning of prompt A
    // can be mixed into the live conditioning of prompt B without either one
    // being re-encoded. The blend is on the post-RoPE K/V of the joint
    // sequence's prefix — one level BELOW the text rows, and two below the
    // text encoder, which may be released by then.
    //
    // Both branches (cond and uncond) are saved and blended together, so a
    // CFG generation stays coherent. The layouts must match: same prefix
    // length, same target grid, same layer count — in practice two prompts
    // that tokenize to the same length. A mismatch throws.
    static constexpr int kQi21PrefixSlots = 4;

    // Copy the live extracted prefix into `slot`. Throws when nothing has
    // been primed, or when the prefix has not been extracted yet (run one
    // step first).
    void qi21_save_prefix_cache(int slot);
    // Blend the live prefix towards the saved one:
    //     k = (1-alpha)*k + alpha*saved.k     (and likewise v)
    // alpha 0 is a no-op, 1 replaces the live prefix wholesale.
    void qi21_blend_prefix_cache(int slot, float alpha);
    // Drop every saved slot. They hold real VRAM (prefix_len * hidden * 2 per
    // layer per branch), so a long-running session should.
    void qi21_clear_prefix_slots();
    // Whether `slot` currently holds an extracted prefix.
    bool qi21_prefix_slot_valid(int slot) const;

    // Read / replace the prepared text rows — txt_in's (n_valid, hidden)
    // output, the joint sequence's text half. The setter resets the prefix
    // cache so the change lands. `uncond` selects the negative branch (only
    // present when guidance_scale > 1 at prime time).
    brotensor::Tensor qi21_text_rows(bool uncond = false) const;
    void qi21_set_text_rows(const brotensor::Tensor& rows, bool uncond = false);

    // ── the between-step control schedule ─────────────────────────────────
    //
    // A control-dictionary direction re-applied with a PER-STEP alpha during
    // denoising. See control_schedule.h for why this is a separate surface
    // from cond_control(): the conditioning axes carry an order of magnitude
    // more steering authority than anything in-network, and a prime-time
    // seam can only aim them once.
    //
    // At every step_once() the armed schedules' stack for that step index is
    // built, added to the conditioning the generation was PRIMED with, pushed
    // back through txt_in and installed as the text rows:
    //
    //     rows(s) = txt_in( primed_embeds + Σ_k alpha_k[s] * scale_k * dir_k )
    //
    // Because the base is the primed embedding, a schedule composes with
    // whatever prime-time setControl() asked for, and its own contribution is
    // independent of that value. Only the POSITIVE branch is steered, exactly
    // as cond_control() steers only the positive branch.
    //
    // Ranges are HALF-OPEN [lo_step, hi_step) like the rest of the 2.1
    // surface, hi_step < 0 means "to the end", and `alpha` is indexed by the
    // ABSOLUTE step index (not the offset from lo_step), so a schedule and
    // its window can be edited independently.
    //
    // Cost: a step whose alpha stack MOVED rebuilds the rows and resets the
    // prefix KV cache, so it pays one extra prefill; a step whose stack sat
    // still costs nothing at all. Armed prefix edits (gate scales, prefix-KV
    // dials) survive the re-extract the way they always have — they are
    // applied where the cache is read, not written into it.

    // Append a schedule over dictionary axis `name`; returns its slot index.
    // Throws when no dictionary is loaded or the axis is unknown.
    int qi21_add_control_schedule(const std::string& name, const std::vector<float>& alpha,
                                  int lo_step, int hi_step);
    // Append a schedule over an explicit direction — a diff-of-means axis the
    // caller minted, which need not be in any bank. `dir` is taken as-is and
    // the injection is alpha * scale * dir, the same contract
    // CondControl::set_vector() has.
    int qi21_add_control_schedule_dir(const std::vector<float>& dir, float scale,
                                      const std::vector<float>& alpha, int lo_step, int hi_step);
    // Replace the whole list with one schedule; returns 0.
    int qi21_set_control_schedule(const std::string& name, const std::vector<float>& alpha,
                                  int lo_step, int hi_step);
    int qi21_set_control_schedule_dir(const std::vector<float>& dir, float scale,
                                      const std::vector<float>& alpha, int lo_step, int hi_step);
    // Drop every schedule. When a schedule had already moved the rows of a
    // live generation, this puts the primed rows back (and re-extracts), so
    // clearing mid-denoise means what it says.
    void qi21_clear_control_schedules();
    int  qi21_control_schedule_count() const;

    // Apply the armed schedules for step `step` to the live generation.
    // step_once() calls this itself, so a caller only needs it when driving
    // the denoiser out of band. Returns true when the rows were rebuilt (and
    // the prefix cache therefore dropped), false when this step's stack was
    // already the one the rows carry.
    bool qi21_apply_control_step(PipelineState& state, int step);

    // ── conditioning entry points ─────────────────────────────────────────

    // Encode `prompt` into 2.1's raw text conditioning: the (n, 4096)
    // Qwen3-VL-8B hidden-state rows the DiT's txt_in consumes, an all-ones
    // validity mask, and the template's token ids. This is the exact tensor
    // prime() builds internally and the space control axes are minted in
    // (encode_conditioning() returns its `embeds`).
    qwenimage21::TextConditioning qi21_encode_prompt(std::string_view prompt);

    // ── the prompt memo ───────────────────────────────────────────────────
    //
    // The last K prompts this pipeline encoded, kept as prompt -> rows. It is
    // what lets prime(prompt) / generate(prompt) work after
    // qi21_release_text_encoder(): releasing the 8.5 GiB backbone used to
    // throw for EVERY prompt, cached embeddings or not, which forced a
    // research loop to hand-roll encode / release / prime_from_text. Now a
    // prompt that has already been encoded primes with the encoder gone, and
    // one that has not throws an error naming it.
    //
    // Filled by qi21_encode_prompt() and by every prime()/generate() that
    // encoded — including the negative prompt, so a CFG generation replays
    // whole. Bounded and least-recently-used: the rows are ~n*4096 floats
    // each, which is a few MB, so the memo is cheap but not free.
    static constexpr std::size_t kQi21PromptMemo = 8;
    std::vector<std::string> qi21_memoized_prompts() const;
    void qi21_clear_prompt_memo();

    // Prime a step-wise generation from caller-supplied (n, 4096) rows
    // instead of encoding a prompt — the analogue of krea_prime_from_taps().
    // `mask` may be empty (all rows valid). uncond_embeds/uncond_mask may be
    // null; when guidance_scale > 1 the uncond branch then falls back to
    // encoding opts.negative_prompt normally.
    PipelineState qi21_prime_from_text(const brotensor::Tensor& embeds,
                                       const brotensor::Tensor& mask,
                                       const brotensor::Tensor* uncond_embeds,
                                       const brotensor::Tensor* uncond_mask,
                                       const GenerateOptions& opts);

    // Encode `prompt` TOGETHER WITH condition images: the same (n, 4096)
    // Qwen3-VL rows qi21_encode_prompt() returns, except that the template
    // reserves a run of rows per image and the vision tower's output stands
    // in their place. The returned conditioning's `image_pad_mask` /
    // `image_runs` say which rows those are and what latent grid each image
    // needs — everything a caller has to know to assemble the joint prefix
    // itself.
    //
    // Images are resized with calculate_dimensions(resolution², aspect)
    // first, so the vision grid and the VAE's latent grid describe the same
    // picture. Throws when the vision tower is not resident.
    qwenimage21::TextConditioning qi21_encode_prompt_images(
        std::string_view prompt, const std::vector<ConditionImage>& images,
        int output_resolution = 1024);

    // Resolve the output canvas a generation with `opts` will use: (width,
    // height) verbatim when both are positive, otherwise the size derived
    // from the LAST condition image's aspect at opts.output_resolution.
    // prime() applies the same rule; this lets a caller size its own output
    // buffer without priming first.
    void qi21_resolve_size(const GenerateOptions& opts, int& width, int& height);

    // ── image seam (the resident 16x RGBA VAE) ────────────────────────────

    // Encode RGB pixels into a pipeline-scale latent. `pixels` is FP32 CHW in
    // [0,1], length 3*H*W; H and W must be multiples of vae_scale_factor()
    // (16). An opaque alpha plane is appended internally (2.1's autoencoder
    // is RGBA on both ends). Returns (1, 64*H/16*W/16) NCHW — the shape
    // PipelineState::latent carries.
    brotensor::Tensor qi21_encode_image(const float* pixels, int H, int W);

    // Decode a pipeline-scale latent (1, 64*h_lat*w_lat) to RGB, FP32 CHW in
    // [-1,1], length 3*(16*h_lat)*(16*w_lat) — alpha dropped, exactly as
    // decode() does for a generation.
    std::vector<float> qi21_decode(const brotensor::Tensor& latent, int h_lat, int w_lat);

    // ── text-encoder residency ────────────────────────────────────────────

    // Free the Qwen3-VL-8B backbone. It is ~8.5 GiB and sits completely idle
    // once prime() has run, so releasing it is what buys a BF16 DiT or a
    // bigger canvas on a 24 GB card. A primed PipelineState keeps working:
    // its conditioning is already encoded and the prefix cache is downstream
    // of it. qi21_encode_prompt() / prime() throw until the encoder is
    // reloaded. Safe to call twice.
    void qi21_release_text_encoder();
    bool qi21_text_encoder_resident() const;
    // Reload the backbone into a pipeline it was released from (or swap in a
    // different one). `text_encoder_path` is a .gguf or a diffusers
    // safetensors file/dir; empty means `<model_dir>/text_encoder`.
    void qi21_reload_text_encoder(const std::string& model_dir,
                                  const std::string& text_encoder_path, bool quantize);

private:
    // Encode a prompt to the CLIP (77, hidden) conditioning. If content_end is
    // non-null, it receives the EOS index (first eos_id) = end of the content
    // rows, for the conditioning-control seam's content-rows policy.
    void encode_prompt_(std::string_view prompt, brotensor::Tensor& out,
                        int* content_end = nullptr);
    // Per-model-family construction + component loading for from_model_dir().
    // Defined in pipeline_krea2.cpp / pipeline_qwenimage21.cpp so the big
    // family-specific weight-plumbing stays out of pipeline.cpp. `cfg` is the
    // PipelineConfig from_model_dir already built from the model directory's
    // JSON (including the quantize flags).
    static Pipeline from_model_dir_krea2_(const std::string& model_dir,
                                          const PipelineConfig& cfg, const ModelDirOptions& opts);
    static Pipeline from_model_dir_qwenimage21_(const std::string& model_dir,
                                                const PipelineConfig& cfg,
                                                const ModelDirOptions& opts);

    // Sana txt2img priming: Gemma-encode prompt(s), prepare conditioning, and
    // allocate the FP32 initial latent (32x downsample, 32 channels). Returns a
    // step_index=0 state. Called from prime() when model_class_ == Sana.
    PipelineState prime_sana_(std::string_view prompt, const GenerateOptions& opts);
    // Sana-Sprint denoising step (SCMScheduler / TrigFlow). The few-step,
    // guidance-distilled (no-CFG) path: maps the latent into the sCM input
    // parameterisation, runs one DiT forward with the embedded guidance scale,
    // reconstructs the model output, and applies one TrigFlow scheduler step.
    // Called from step_once() when the active scheduler is scheduler::SCM.
    void step_once_scm_(PipelineState& state, const GenerateOptions& opts);
    // SD1.5 / Flux img2img (and SD1.5 inpaint) priming: VAE-encode the init image, noise it to
    // t_start, and (when a mask is given) cache the broadcast inpaint masks.
    // Sets state.latent and state.step_index; called from prime() once the
    // schedule is set, when opts.init_image_path is non-empty.
    void prime_img2img_(const GenerateOptions& opts, PipelineState& state,
                        int n_lat, int H_lat, int W_lat, int C_lat);

    PipelineConfig            cfg_;
    ModelClass                model_class_;
    // CLIP tokenizer for SD / Flux. Empty for Sana (which has no CLIP
    // tokenizer; clip::Tokenizer is not default-constructible, hence optional).
    std::optional<brolm::clip::Tokenizer> tokenizer_;
    brolm::clip::TextEncoder  text_encoder_;    // CLIP — SD / Flux (unused Sana)
    std::optional<brolm::t5::Tokenizer>   t5_tokenizer_;   // Flux only
    std::optional<brolm::t5::TextEncoder> t5_encoder_;     // Flux only
    std::unique_ptr<Denoiser> denoiser_;
    vae::Decoder              vae_;
    // VAE encoder mirrors vae_; used by img2img / inpaint priming. Always
    // constructed (config derived from cfg.vae); weights are loaded by every
    // load_weights() overload and by the Flux model-dir load (img2img).
    // Robustness: if encoder weights are missing in the safetensors file
    // (some decoder-only derivative checkpoints), the load throws clearly.
    vae::Encoder              vae_encoder_;
    std::variant<scheduler::DDIM, scheduler::LCM, scheduler::FlowMatch,
                 scheduler::SCM, scheduler::DPMSolverMultistep>
        scheduler_;

    // ── Sana-only sub-modules ─────────────────────────────────────────────
    // The SanaDenoiser lives in denoiser_; these are the pieces that have no
    // SD / Flux analogue. decode() routes to dcae_ for Sana (vae_ is unused);
    // prime_sana_ Gemma-encodes the prompt via gemma_model_ / gemma_tokenizer_.
    // All empty / default for non-Sana pipelines.
    std::optional<dcae::Decoder>             dcae_;
    std::optional<brolm::gemma::Gemma2Model> gemma_model_;
    std::optional<brolm::gemma::Tokenizer>   gemma_tokenizer_;

    // ── Krea 2-only sub-modules ───────────────────────────────────────────
    // The Krea2Denoiser lives in denoiser_; these have no SD / Flux analogue.
    // decode() routes to vae_qwen_ for Krea 2; prime() Qwen3-VL-encodes the
    // prompt (tapped hidden states + validity mask) via qwen3vl_model_ /
    // qwen3vl_tokenizer_. All empty for non-Krea2 pipelines.
    std::optional<vae_qwenimage::Decoder>    vae_qwen_;
    std::optional<brolm::qwen3vl::TextModel> qwen3vl_model_;
    std::optional<brolm::qwen3vl::Tokenizer> qwen3vl_tokenizer_;
    // The Qwen3-VL vision tower + image preprocessor config — unused by plain
    // text prompting, loaded (from the SAME text_encoder shard(s)
    // qwen3vl_model_ loads) so the image-conditioned paths can work:
    // krea_encode_image_prompt() for Krea 2, qi21_encode_prompt_images() and
    // the edit-mode prime for Qwen-Image 2.1. Left unconstructed for every
    // other model class.
    std::optional<brolm::qwen3vl::VisionTower> qwen3vl_vision_;
    brolm::qwen3vl::PreprocessConfig           qwen3vl_pp_;

    // ── Qwen-Image 2.1-only sub-modules ───────────────────────────────────
    // The QwenImage21Denoiser lives in denoiser_; the text frontend reuses
    // qwen3vl_model_ / qwen3vl_tokenizer_ / qwen3vl_vision_ above (an 8B
    // backbone here, and its own tower). decode() routes to vae_qi21_,
    // which is the 16x RGBA residual VAE: it emits four planes and decode()
    // drops the alpha one. The encoder is constructed alongside so the
    // image-conditioned paths have it resident; it is loaded from the same
    // single VAE file. Both empty for every other model class.
    std::optional<vae_qwenimage21::Decoder>  vae_qi21_;
    std::optional<vae_qwenimage21::Encoder>  vae_qi21_encoder_;

    // Model-agnostic raw conditioning, rebuilt each prime(). Kept around for
    // trace-mode / ControlNet access to the raw text context (those paths
    // reflect the LATEST prime()). The per-denoiser prepared payload (K/V
    // caches) is NOT held here — prime() moves it onto the returned
    // PipelineState (state.prepared), so states from different prime() calls
    // each step under their own conditioning.
    Conditioning          conditioning_;

    // Conditioning-space control axes, applied to the positive text embeddings
    // in every prime() (no-op until a dictionary is loaded + a weight set).
    CondControl           cond_control_;

    // Between-step control schedules (Qwen-Image 2.1). The conditioning rows
    // they rebuild come from conditioning_ above — which is the primed
    // embedding, cond_control_ already folded in — so a schedule is a delta
    // on the static desk rather than a replacement for it.
    ControlSchedule       qi21_ctl_sched_;

    // Reference-attention identity anchor state (Sana only). identity_anchor_ is
    // set once capture_identity_anchor() succeeds; identity_weight_ scales the
    // injection (0 = off). capturing_anchor_ is true only while
    // capture_identity_anchor() drives the recording generation, so prime_sana_
    // leaves the denoiser in Capture mode instead of switching it to Inject.
    bool  identity_anchor_   = false;
    bool  capturing_anchor_  = false;
    float identity_weight_   = 0.0f;

    // Backing store for krea_capture_gates()/krea_gates() — owned here so its
    // lifetime outlives the Krea2Transformer2DModel::capture_gates() pointer
    // registration across step_once() calls.
    std::vector<float> krea_gate_sink_;

    // Set by krea_prime_from_taps() immediately before delegating to prime();
    // prime()'s Krea2 branch consumes (moves out of) these instead of calling
    // krea2::encode_prompt() when present, then clears them. Lets the two
    // prime paths share every non-text-encode step (latent alloc, RNG,
    // scheduler setup, CUDA graph keying) with zero duplication.
    std::optional<krea2::TextConditioning> krea_taps_override_;
    std::optional<krea2::TextConditioning> krea_uncond_taps_override_;

    // The Qwen-Image 2.1 counterparts of the four members above.
    // qi21_gate_sink_ backs qi21_capture_gates()/qi21_gates();
    // qi21_text_override_ is consumed by prime()'s QwenImage21 branch when
    // qi21_prime_from_text() set it.
    std::vector<float> qi21_gate_sink_;
    std::vector<float> qi21_gate_sink_mlp_;
    std::optional<qwenimage21::TextConditioning> qi21_text_override_;
    std::optional<qwenimage21::TextConditioning> qi21_uncond_text_override_;
    // A weak handle on the conditioning the most recent prime() produced.
    // Qwen-Image 2.1's prefix KV cache lives inside it, and every prefix-side
    // research hook has to invalidate that cache — but the payload is owned
    // by the PipelineState the caller holds, not by Pipeline, so this is
    // deliberately non-owning: a hook called after the state was dropped is a
    // no-op rather than a dangling write.
    std::weak_ptr<PreparedConditioning> last_prepared_;
    // The image-conditioned joint prefix for the generation being primed.
    // prime()'s QwenImage21 branch fills these from opts.condition_images and
    // the prepare_edit() call a few lines later consumes them; they are
    // cleared on every prime so a text-only run after an edit run cannot
    // inherit an image prefix. Not part of PipelineState because the prepared
    // conditioning already owns its own copy once prepare_edit() has run.
    dit::QwenImage21EditPrefix qi21_edit_prefix_;
    dit::QwenImage21EditPrefix qi21_edit_uncond_prefix_;
    bool qi21_edit_active_ = false;
    // Decode + resize + VAE-encode every condition image, and describe the
    // joint prefix each branch's conditioning implies. Fills the three
    // members above and both branches of `conditioning_`. Lives in
    // pipeline_qwenimage21_edit.cpp.
    void qi21_prime_edit_(std::string_view prompt, const GenerateOptions& opts, bool do_cfg);
    // What the currently-armed Qwen-Image 2.1 hooks do to the PREFIX side.
    // Only an extract step applies a prefix-side hook, so the cache has to be
    // dropped both when one is armed AND when one is cleared — the latter is
    // why these are remembered rather than read off the arguments.
    bool  qi21_mod_delta_hits_prefix_  = false;
    bool  qi21_gate_delta_hits_prefix_ = false;
    bool  qi21_gate_mask_armed_        = false;
    float qi21_prefix_gate_attn_      = 1.0f;
    float qi21_prefix_gate_mlp_       = 1.0f;
    // Re-read the whole armed state and drop the cache if the prefix side of
    // it changed. Called after every hook mutation, so a list edit is judged
    // as a list rather than as the one binding that moved.
    void qi21_sync_prefix_state_();

    // The prefix cache slots. One pair per slot because a CFG generation has
    // two branches whose prompts differ; `uncond_valid` says whether the
    // saved branch pair carried one.
    using Qi21PrefixSlot = detail::Qi21PrefixSlot;
    std::array<Qi21PrefixSlot, kQi21PrefixSlots> qi21_prefix_slots_;

    // The prompt memo, most recently encoded FIRST. A plain vector: K is 8,
    // so a linear scan is faster than any map and the eviction order is the
    // vector's own.
    using Qi21PromptMemo = detail::Qi21PromptMemo;
    std::vector<Qi21PromptMemo> qi21_prompt_memo_;
    // Remember `prompt`'s rows, evicting the least recently used. A re-encode
    // of a memoized prompt moves it to the front.
    void qi21_memo_put_(const std::string& prompt, const qwenimage21::TextConditioning& tc);
    // The memoized rows for `prompt`, or nullptr. Moves the hit to the front.
    const qwenimage21::TextConditioning* qi21_memo_get_(const std::string& prompt);
    // Encode `prompt`, or serve it from the memo when the backbone has been
    // released. Throws an error NAMING the prompt when neither is possible.
    qwenimage21::TextConditioning qi21_encode_or_memo_(const std::string& prompt);

    // Rebuild `prepared`'s positive text rows as txt_in(primed embeds + the
    // injection `delta` carries), dropping the prefix KV cache. An inactive
    // `delta` restores the primed rows exactly. The one place the between-step
    // schedule touches the conditioning.
    void qi21_rebuild_text_rows_(PreparedConditioning& prepared, const CondControl& delta);

    // Working buffers reused across step_once() calls. The current latent
    // lives on PipelineState, not here.
    brotensor::Tensor noise_pred_cond_, noise_pred_uncond_;
    brotensor::Tensor decoded_, scratch_;
    // Per-step Gaussian noise used by LCM (resampled at every non-final step
    // from the same RNG stream as the initial latent noise). Unused in DDIM.
    brotensor::Tensor noise_step_;

    // Inpaint blending state. Built in prime() when opts.mask_image_path is
    // non-empty; consumed by step_once() to blend the unmasked region of the
    // generated latent with a re-noised version of x0 at each step. `mask_b`
    // is the per-channel-broadcast mask (shape matches the latent); raw
    // `mask` of shape (1, H_lat*W_lat) is kept around for tests / debug.
    // Empty tensors + inpaint_active_=false mean inpaint mode is off for the
    // current generation. Lives on Pipeline (not PipelineState) — shared
    // across branched states from the same prime() call, same as the K/V
    // cache.
    brotensor::Tensor inpaint_x0_;          // (1, C_lat*H_lat*W_lat)
    brotensor::Tensor inpaint_mask_;        // (1, H_lat*W_lat), {0,1}
    brotensor::Tensor inpaint_mask_b_;      // (1, C_lat*H_lat*W_lat) broadcast
    brotensor::Tensor inpaint_one_minus_b_; // (1, C_lat*H_lat*W_lat) = 1 - mask_b
    brotensor::Tensor inpaint_renoise_buf_; // (1, C_lat*H_lat*W_lat) scratch
    brotensor::Tensor inpaint_noise_step_;  // (1, C_lat*H_lat*W_lat) Philox draw
    bool              inpaint_active_ = false;

    // ControlNet plumbing. `controlnets_` is the registered stack (lifetime
    // tied to Pipeline; appended via add_controlnet, dropped via
    // remove_controlnet / clear_controlnets). `control_inputs_` is the
    // per-run input list copied from GenerateOptions::controls in prime();
    // size == controlnets_.size() when active. `control_images_` holds one
    // image tensor per registered ControlNet, built in prime() at the
    // active device + compute dtype and reused for every step in the
    // current generation. `controlnet_active_` mirrors "GenerateOptions
    // .controls is non-empty AND the registered count matches" — checked
    // by step_once to branch into the residual-aware UNet path.
    //
    // Step buffers: `cn_down_residuals_` / `cn_mid_residual_` are the
    // summed-across-nets residuals fed into the UNet skips; the first net
    // writes directly into them and subsequent nets accumulate via
    // bt::add_inplace through the *_scratch_ buffers. With one registered
    // ControlNet the scratch buffers are unused.
    std::vector<std::unique_ptr<controlnet::ControlNet>> controlnets_;
    std::vector<ControlNetInput>                         control_inputs_;
    std::vector<brotensor::Tensor>                       control_images_;
    bool                                                 controlnet_active_ = false;
    std::vector<brotensor::Tensor>                       cn_down_residuals_;
    brotensor::Tensor                                    cn_mid_residual_;
    std::vector<brotensor::Tensor>                       cn_down_residuals_scratch_;
    brotensor::Tensor                                    cn_mid_residual_scratch_;

    // CUDA-graph denoising-step session (defined in pipeline.cpp; CUDA-only
    // payload). When the denoiser supports step capture and the latent is
    // CUDA-resident, step_once runs the first two steps of a generation
    // eagerly through the capture seam (settling every scratch buffer at its
    // high-water capacity), records the cond(+uncond) forward bodies as one
    // CUDA graph at the end of the second step, and replays that graph for
    // every later step — one launch instead of hundreds. The session is keyed
    // on (latent pointer, prepared payload, H, W, CFG); any mismatch (new
    // prime(), forked state, resolution change) falls back to eager stepping
    // and re-captures. Trace-mode and ControlNet steps always run eager.
    struct StepGraphSession;
    std::unique_ptr<StepGraphSession> step_graph_;
    // Runs cond(+uncond) denoiser passes for one step via the capture seam,
    // managing warm-up / capture / replay. Outputs land in noise_pred_cond_
    // (+ noise_pred_uncond_), exactly like the eager path.
    void step_denoise_captured_(PipelineState& state, float t, bool do_cfg);
};

}  // namespace brodiffusion::pipeline
