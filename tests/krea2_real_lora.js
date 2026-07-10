// Real-LoRA verification against the official Krea-released adapters
// (krea/Krea-2-LoRA-* on HF, rank-32 PEFT exports covering every DiT linear
// plus time_embed and the text_fusion projector row). Run headless from a
// neutral app dir (NOT krea2-lab — it auto-loads the model itself):
//
//   bro-headless ../broworkshop/demos/example ../../brodiffusion/tests/krea2_real_lora.js
//
// Fetch the LoRAs first (rank-32 F32 PEFT exports, ~470 MB each):
//   hf download krea/Krea-2-LoRA-retroanime retroanime.safetensors \
//       --local-dir weights/krea-2-loras
//   hf download krea/Krea-2-LoRA-dotmatrix dotmatrix.safetensors \
//       --local-dir weights/krea-2-loras
//
// Renders: base, retroanime@1, retroanime@0 (must return to base),
// retroanime@1 + dotmatrix@1 stacked. PNGs land in the OS temp dir for
// visual judgement; the script itself asserts the mechanical invariants.

const MODEL_DIR = 'D:/projects/brodiffusion/weights/krea-2-turbo';
const LORA_DIR = 'D:/projects/brodiffusion/weights/krea-2-loras';
const OUT_DIR = require('os').tmpdir().replace(/\\/g, '/');

const PROMPT = 'a red fox sitting in a snowy forest clearing at dawn';
const GEN = { width: 512, height: 512, steps: 8, guidanceScale: 1.0, seed: 7 };

console.log('loading Krea 2 Turbo (INT8)…');
const t0 = Date.now();
const pipe = bro.diffusion.loadModel(MODEL_DIR, { quantizeWeights: true });
console.log('loaded in ' + ((Date.now() - t0) / 1000).toFixed(1) + 's');

function png(name, img) {
  // img: {width, height, data} — RGBA bytes.
  assert(!img.cancelled, 'generation was not cancelled');
  bro.image.encodePngFile(OUT_DIR + '/' + name, img.data, img.width,
                          img.height, 4);
  console.log('wrote ' + name);
  return img.data;
}

function diffBytes(a, b) {
  let n = 0;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) n++;
  return n;
}

const base = png('krea2_real_base.png', pipe.generate(PROMPT, GEN));
const total = base.length;

console.log('applying retroanime…');
const g0 = pipe.applyLora(LORA_DIR + '/retroanime.safetensors', 1.0);
console.log('retroanime applied, group ' + g0 + ', numLoras=' + pipe.numLoras());
assert(pipe.numLoras() === 1, 'one adapter group attached');
const retro = png('krea2_real_retroanime.png', pipe.generate(PROMPT, GEN));
const dRetro = diffBytes(base, retro);
console.log('retroanime vs base: ' + dRetro + ' / ' + total + ' differing bytes');
assert(dRetro > total * 0.05, 'retroanime at scale 1 visibly changes the render');

pipe.setLoraScale(g0, 0.0);
const zero = pipe.generate(PROMPT, GEN).data;
const dz = diffBytes(base, zero);
console.log('scale 0 vs base: ' + dz + ' differing bytes');
assert(dz === 0, 'scale 0 returns exactly to the base render');
pipe.setLoraScale(g0, 1.0);

console.log('stacking dotmatrix…');
const g1 = pipe.applyLora(LORA_DIR + '/dotmatrix.safetensors', 1.0);
assert(g1 === 1 && pipe.numLoras() === 2, 'two adapter groups attached');
const both = png('krea2_real_stacked.png', pipe.generate(PROMPT, GEN));
assert(diffBytes(retro, both) > total * 0.05,
       'stacked dotmatrix changes the render');

pipe.clearLoras();
assert(pipe.numLoras() === 0, 'clearLoras drops both groups');
const cleared = pipe.generate(PROMPT, GEN).data;
assert(diffBytes(base, cleared) === 0, 'cleared returns exactly to base');

console.log('PASS — official Krea LoRAs load, restyle, rescale, stack, clear');
