# Qwen-Image 2.1 — image-conditioned generation (edit / reference images)

Qwen-Image 2.1 generates from condition images as well as from a prompt. The
condition images enter the model TWICE, through two different doors, and every
binding below exists because of that split:

1. **Through the text encoder.** The chat template reserves a run of
   `<|image_pad|>` rows per image; the Qwen3-VL vision tower fills them (with
   DeepStack features spliced into the first three decoder layers). Those rows
   make the prompt *about* the picture — "make the sky red" resolves against
   what the encoder saw.
2. **Through the autoencoder.** The same resized image is VAE-encoded to a
   64-channel latent grid and prepended to the DiT's joint sequence as its own
   block-causal segment. Those tokens carry the pixels the denoiser copies
   from.

The rows from door 1 are **dropped before `txt_in`**: the DiT fills the image
positions from door 2's latents, through a different projection. The library
does that internally (`strip_image_rows`), which is exact because
`txt_in` (zero-centre RMSNorm + linear) is row-independent.

## Geometry (the rule that makes the two doors line up)

Each condition image is resized once, before anything else, to

```
calculate_dimensions(output_resolution², aspect)
  width  = round(sqrt(area * ratio) / 32) * 32
  height = round((width / ratio)    / 32) * 32
```

Both sides are multiples of 32, and that is the whole trick: the vision tower
emits one token per 32 px, the autoencoder one latent per 16 px, so

```
grid_h * grid_w  ==  4 * n_vision_slots
```

Each `<|image_pad|>` slot therefore stands for exactly **four consecutive
latent tokens** in the row-major flatten (`_IMG_TOKENS_PER_SLOT = 4`) — four
consecutive, not a 2x2 spatial group; the reference scatters packed latents
into expanded mask positions in order. The library asserts the identity, so a
mismatched resize fails loudly instead of silently mis-aligning.

The **output canvas** defaults to the aspect of the LAST condition image at
`output_resolution` (1024 unless overridden) — the same rule the diffusers
pipeline follows. Passing an explicit width/height overrides it.

## CLI

```sh
brodiffusion txt2img --model <qi21-dir> \
  --prompt "make the sky red" \
  --image cat.png \
  --output-resolution 1024 \
  --steps 8 --cfg 4.0 --out out.png
```

`--image` is repeatable (template order = argument order). `--width/--height`
stay optional; with `--image` and neither given, the canvas is derived.
`--output-resolution` is QwenImage21-only and rejected elsewhere.

## JS (bronze / bro.diffusion)

### `generate` / `prime` options

```js
const img = pipeline.generate("make the sky red", {
  conditionImages: ["cat.png"],      // or [{ path }] / [{ pixels, width, height, channels }]
  outputResolution: 1024,            // canvas area the aspect is fitted to
  steps: 8, guidanceScale: 4.0, seed: 1234n,
  // width/height omitted -> derived from the last condition image
});
```

`pixels` is a planar CHW `Float32Array` in `[0, 1]`; `channels` may be 3 or 4
(RGBA is composited over white for the vision tower and carried whole into the
autoencoder, which is RGBA on both ends). Set exactly one of `path` and
`pixels` per entry; both, or neither, is an error.

When `conditionImages` is present and no numeric `width`/`height` was given,
the binding resolves the derived size before priming, so the returned image
object's `width`/`height` are the real ones.

### `qwenImage21EncodePromptImages(prompt, images, outputResolution?)`

The image-conditioned counterpart of `qwenImage21EncodePrompt`. Returns

```js
{
  embeds: { rows, cols, data },   // (n_valid, 4096), image rows INCLUDED
  mask:   { rows, cols, data },
  ids:    Int32Array,             // full template ids (n_valid + dropIdx)
  dropIdx: 14,
  imagePadMask: Int32Array,       // n_valid, 1 at a condition-image row
  imageRuns: [{ row, slots, hLat, wLat }, ...]
}
```

`imageRuns[i]` gives the first pad row of image `i`, its slot count, and the
latent grid to encode that image at — `slots * 4 === hLat * wLat`. This is the
hook for editing what the model *saw*: rewrite rows inside a run and feed the
result to `qwenImage21PrimeFromText`.

### `qwenImage21PrimeEdit(prompt, images, opts?)`

`prime()` with condition images, returning a `PipelineState` the caller steps
itself (`stepOnce`, the x0 preview, the gate/mod hooks — all unchanged). The
`images` argument takes the same entries as `conditionImages` and overrides
any `conditionImages` inside `opts`. With no explicit width/height in `opts`
the canvas is derived as above.

Everything in the existing qwenImage21 research block keeps working across an
edit prefix: `qwenImage21ScalePrefixKv` now addresses the interleaved
text+image prefix, and `qwenImage21ResetCache` re-extracts it.

## C API (`qwenimage21_capi.dll`)

Four steps, because the C surface wraps the components individually and the
caller owns the seam between them:

```c
/* 1. encode the prompt WITH the images (vision tower runs here) */
const char* paths[] = { "cat.png" };
int n_valid = qi_encode_prompt_images(ctx, "make the sky red", paths, 1, 1024);

/* 2. pull the rows, drop the ones the pad mask marks, project the rest */
qi_get_prompt_embeds(ctx, embeds);           /* (n_valid, text_hidden_dim) */
qi_get_prompt_pad_mask(ctx, pad);            /* n_valid int32s */
/* ...compact `embeds` by `pad`... */
qi_encode_text(ctx, text_rows, n_text, txt); /* (n_text, hidden_size) */

/* 3. VAE-encode each image at the grid the prompt reserved, concatenate in
      template order, and arm the prefix */
int n_img = qi_prompt_num_images(ctx);
for (int i = 0; i < n_img; ++i) qi_get_prompt_image_grid(ctx, i, &h_lat, &w_lat);
qi_set_condition_latents(ctx, cond_latents, n_tokens);  /* NULL clears */

/* 4. step as usual — qi_forward reads the armed prefix */
qi_forward(ctx, latent, h_lat, w_lat, txt, n_text, t, out);
```

Signatures:

```c
int qi_encode_prompt_images(qi_ctx*, const char* prompt,
                            const char* const* image_paths,
                            int n_images, int output_resolution);
int qi_get_prompt_pad_mask(qi_ctx*, int32_t* out);
int qi_prompt_num_images(qi_ctx*);
int qi_get_prompt_image_grid(qi_ctx*, int index, int* h_lat, int* w_lat);
int qi_set_condition_latents(qi_ctx*, const float* latents, int n_tokens);
```

`qi_encode_prompt` (text-only) clears any armed condition latents, so the two
modes cannot be crossed by accident. `qi_set_condition_latents` resets the
prefix KV cache; so does a prompt or grid change.

## C++ (`brodiffusion::pipeline::Pipeline`)

```cpp
qwenimage21::TextConditioning qi21_encode_prompt_images(
    std::string_view prompt, const std::vector<ConditionImage>& images,
    int output_resolution = 1024);
void qi21_resolve_size(const GenerateOptions& opts, int& width, int& height);
```

plus `GenerateOptions::condition_images` / `::output_resolution`, which is what
`generate()` and `prime()` read.

## Prefix cache across image segments

The joint sequence is `[text | image₀ | image₁ | ... | target]`, block-causal:
key *j* is visible to query *i* iff `i >= j` or both are in the same image
block. Text and condition-image tokens modulate from **t = 0**
(`causal_condition`), so the whole prefix is timestep-independent: it is
extracted on step 0 and decoded from afterwards. The cache is keyed on the
total prefix length together with the target grid, so changing the image
count, the images, or the canvas invalidates it automatically.

## Measured parity

Against the fp32 diffusers reference, image-mode prompt encoding:

| quantity | value |
| --- | --- |
| token ids | identical |
| `image_pad_mask` | identical |
| overall cosine | 0.997615 |
| image-slot median cosine | 0.99988 |
| text-row median cosine | 0.999994 |

End-to-end edit (brodiffusion INT8 DiT + INT8 TE vs BF16 diffusers, 512², 8
steps, shared initial noise and shared resized condition image): final-latent
cosine **0.999355**, rel L2 0.036, per-channel min 0.9937.

Reproduce with `scripts/qwenimage21_text_parity.sh` (set `QI21_TEXT_IMAGE`)
and `scripts/qwenimage21_edit_parity.sh`.
