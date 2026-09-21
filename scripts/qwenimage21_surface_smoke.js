// Qwen-Image 2.1 research control surface — end-to-end smoke against the real
// checkpoint, driven from bro-headless.
//
//   bro-headless <any-app-dir> scripts/qwenimage21_surface_smoke.js
//
// Any app directory with an index.html works — a scratch dir holding an empty
// one is enough. Do NOT use broworkshop/demos/krea2-lab: it loads a model of
// its own on boot and the two will not fit on the card together.
//
// PNGs and the log land in $QI21_OUT when it is set, else the OS temp dir.
//
// What it asserts, in order:
//
//   1. the baseline render is DETERMINISTIC — same seed, same bytes. Every
//      later claim ("this hook changed the image") is worthless without it.
//   2. a modulation delta on the last third of the stack changes the image.
//   3. a gate scale changes the image.
//   4. a text-embedding control axis — a diff-of-means direction built from
//      encodeConditioning() over three warm/cold scene pairs — changes the
//      image, and setControl(0) returns to baseline exactly.
//   5. gate capture returns a (numLayers, joint) grid of finite values.
//   6. the prefix KV cache surface: qwenImage21ScalePrefixKv() mid-denoise
//      changes the image with no re-extraction.
//   7. the VAE seam round-trips: encodeImage -> decode reconstructs the render.
//   8. the text encoder can be released and the pipeline still steps.
//
// Every difference is reported as pixel MSE against the baseline, and every
// MSE is checked finite and > 0. PNGs land in OUT_DIR for visual judgement.

const MODEL_DIR = 'D:/projects/brodiffusion/weights/qwen-image-2.1';
const OUT_DIR = (globalThis.QI21_OUT ||
                 (typeof process !== 'undefined' && process.env && process.env.QI21_OUT) ||
                 require('os').tmpdir()).replace(/\\/g, '/');

const PROMPT = 'a lighthouse on a rocky coast at sunset, dramatic clouds';
const GEN = { width: 512, height: 512, steps: 8, guidanceScale: 1.0, seed: 1 };

let failures = 0;
function check(cond, msg) {
  if (cond) { console.log('  ok   ' + msg); }
  else { console.log('  FAIL ' + msg); failures++; }
}

function png(name, img) {
  bro.image.encodePngFile(OUT_DIR + '/' + name, img.data, img.width,
                          img.height, 4);
  console.log('  wrote ' + name);
  return img;
}

// Mean squared error over RGB bytes, in [0, 255^2].
function mse(a, b) {
  let acc = 0, n = 0;
  for (let i = 0; i < a.length; i += 4) {
    for (let c = 0; c < 3; c++) {
      const d = a[i + c] - b[i + c];
      acc += d * d; n++;
    }
  }
  return acc / n;
}

function timed(label, fn) {
  const t = Date.now();
  const r = fn();
  const s = (Date.now() - t) / 1000;
  console.log('  [' + s.toFixed(2) + 's] ' + label);
  return r;
}

// ── load ────────────────────────────────────────────────────────────────────

console.log('loading Qwen-Image 2.1 (INT8 DiT + INT8 text encoder)…');
const pipe = timed('loadModel', () =>
  bro.diffusion.loadModel(MODEL_DIR, { quantizeWeights: true }));

const cfg = pipe.config();
console.log('modelClass = ' + cfg.modelClass);
check(cfg.modelClass === 'QwenImage21', 'modelClass is QwenImage21');
const H = pipe.qwenImage21HiddenSize();
const NL = pipe.qwenImage21NumLayers();
const TH = pipe.qwenImage21TextHiddenDim();
console.log('hidden=' + H + ' layers=' + NL + ' textHidden=' + TH);
check(H === 4096, 'hidden size is 4096');
check(NL === 32, 'num layers is 32');
check(TH === 4096, 'text hidden dim is 4096');

// ── 1. baseline determinism ────────────────────────────────────────────────

console.log('\n[1] baseline determinism');
const base = timed('baseline render', () => pipe.generate(PROMPT, GEN));
png('qi21_base.png', base);
const base2 = timed('baseline re-render', () => pipe.generate(PROMPT, GEN));
const dBase = mse(base.data, base2.data);
console.log('  baseline vs itself MSE = ' + dBase);
check(dBase === 0, 'the same seed renders bit-identically');

// ── 2. modulation delta on blocks [20, 32) ─────────────────────────────────

console.log('\n[2] modulation delta, target rows, blocks [20, 32)');
{
  const st = pipe.prime(PROMPT, GEN);
  const t = st.qwenImage21StepTimestep();
  const { temb, modTarget, modPrefix } = pipe.qwenImage21TimeMod(t);
  console.log('  temb ' + temb.rows + 'x' + temb.cols +
              ', modTarget ' + modTarget.rows + 'x' + modTarget.cols);
  check(temb.rows === 2 && temb.cols === H, 'timeMod temb is (2, hidden)');
  check(modTarget.cols === 4 * H, 'timeMod mod row is (1, 4*hidden)');
  let tdiff = 0;
  for (let i = 0; i < modTarget.cols; i++) {
    tdiff = Math.max(tdiff, Math.abs(modTarget.data[i] - modPrefix.data[i]));
  }
  console.log('  max |modTarget - modPrefix| = ' + tdiff.toFixed(5));
  check(tdiff > 0, 'the sampled-t and t=0 modulation rows differ');

  // Push the SwiGLU scale chunk (the third of four) by 10% of its own value.
  const d = { rows: 1, cols: 4 * H, data: new Float32Array(4 * H) };
  for (let i = 2 * H; i < 3 * H; i++) d.data[i] = 0.10 * modTarget.data[i];
  pipe.qwenImage21SetModDelta(d, 20, 32, 'target');
  const img = timed('mod-delta render', () => pipe.generate(PROMPT, GEN));
  pipe.qwenImage21SetModDelta(null, 0, 0);
  png('qi21_moddelta.png', img);
  const m = mse(base.data, img.data);
  console.log('  mod-delta vs base MSE = ' + m.toFixed(3));
  check(isFinite(m) && m > 0, 'a target-row mod delta changes the image');

  // ...and clearing it returns to baseline exactly.
  const back = pipe.generate(PROMPT, GEN);
  check(mse(base.data, back.data) === 0, 'clearing the delta restores baseline');
}

// ── 3. gate scale ──────────────────────────────────────────────────────────

console.log('\n[3] gate scale, attention gate on blocks [16, 32)');
{
  pipe.qwenImage21SetGateScale(0.85, 1.0, 1.0, 1.0, 16, 32);
  const img = timed('gate-scale render', () => pipe.generate(PROMPT, GEN));
  pipe.qwenImage21SetGateScale(1.0, 1.0, 1.0, 1.0, 0, 0);
  png('qi21_gatescale.png', img);
  const m = mse(base.data, img.data);
  console.log('  gate-scale vs base MSE = ' + m.toFixed(3));
  check(isFinite(m) && m > 0, 'a gate scale changes the image');
  const back = pipe.generate(PROMPT, GEN);
  check(mse(base.data, back.data) === 0, 'clearing the gate scale restores baseline');
}

// ── 4. a text-embedding control axis ───────────────────────────────────────
//
// The generic control-axis machinery works on any model whose conditioning is
// a per-token embedding sequence. For 2.1 that is the raw (n, 4096) Qwen3-VL
// rows, which encodeConditioning() hands back — so a diff-of-means direction
// over warm/cold phrasings of the same three scenes is a valid axis with no
// dictionary file anywhere.

console.log('\n[4] control axis from an encodeConditioning diff-of-means');
{
  const SCENES = [
    'a harbour at dusk',
    'a kitchen table with fruit',
    'a city street after rain',
  ];
  const warm = new Float64Array(TH);
  const cold = new Float64Array(TH);

  function accumulate(prompt, sink) {
    const e = pipe.encodeConditioning(prompt);
    // Mean over token rows. The template's fixed head/tail rows are shared by
    // both members of a pair, so they cancel in the difference.
    for (let r = 0; r < e.rows; r++) {
      for (let c = 0; c < TH; c++) sink[c] += e.data[r * e.cols + c] / e.rows;
    }
  }
  timed('6 conditioning encodes', () => {
    for (const s of SCENES) {
      accumulate(s + ', warm golden light', warm);
      accumulate(s + ', cold blue light', cold);
    }
  });

  const dir = new Float32Array(TH);
  let norm = 0;
  for (let c = 0; c < TH; c++) {
    dir[c] = (warm[c] - cold[c]) / SCENES.length;
    norm += dir[c] * dir[c];
  }
  norm = Math.sqrt(norm);
  console.log('  raw axis norm = ' + norm.toFixed(4));
  check(isFinite(norm) && norm > 0, 'the diff-of-means axis is non-degenerate');
  for (let c = 0; c < TH; c++) dir[c] /= norm;   // unit direction

  pipe.setControlVector('warmth', dir, 3.0, norm);
  const cn = pipe.controlNorm();
  console.log('  controlNorm = ' + cn.norm.toFixed(3) +
              ' (budget ' + cn.budget + ', clamped ' + cn.clamped + ')');
  check(cn.norm > 0, 'the axis stack reports a non-zero norm');
  const img = timed('axis render', () => pipe.generate(PROMPT, GEN));
  png('qi21_axis_warm.png', img);
  const m = mse(base.data, img.data);
  console.log('  axis vs base MSE = ' + m.toFixed(3));
  check(isFinite(m) && m > 0, 'a text-embedding control axis changes the image');

  // The opposite sign is a different image again.
  pipe.setControl('warmth', -3.0);
  const cold_img = pipe.generate(PROMPT, GEN);
  png('qi21_axis_cold.png', cold_img);
  const m2 = mse(img.data, cold_img.data);
  console.log('  +3 vs -3 MSE = ' + m2.toFixed(3));
  check(isFinite(m2) && m2 > 0, 'the axis is signed');

  pipe.removeControl('warmth');
  const back = pipe.generate(PROMPT, GEN);
  check(mse(base.data, back.data) === 0, 'removing the axis restores baseline');
}

// ── 5. gate capture ────────────────────────────────────────────────────────

console.log('\n[5] gate capture');
{
  pipe.qwenImage21CaptureGates(true);
  const st = pipe.prime(PROMPT, GEN);
  timed('one captured step', () => st.stepOnce());
  const g = pipe.qwenImage21Gates();
  pipe.qwenImage21CaptureGates(false);
  console.log('  gates ' + g.rows + 'x' + g.cols +
              ' (' + g.data.length + ' floats)');
  check(g.rows === NL, 'gate capture has one row per block');
  check(g.cols > 0 && g.rows * g.cols === g.data.length, 'gate grid is dense');
  let nonFinite = 0, mn = Infinity, mx = -Infinity;
  for (let i = 0; i < g.data.length; i++) {
    const v = g.data[i];
    if (!isFinite(v)) nonFinite++;
    else { if (v < mn) mn = v; if (v > mx) mx = v; }
  }
  console.log('  gate range [' + mn.toFixed(5) + ', ' + mx.toFixed(5) + ']');
  check(nonFinite === 0, 'every captured gate is finite');
}

// ── 6. prefix KV cache steering ────────────────────────────────────────────

console.log('\n[6] prefix KV attenuation, mid-denoise, deep layers');
{
  const st = pipe.prime(PROMPT, GEN);
  const half = Math.floor(GEN.steps / 2);
  timed('8 steps with a mid-run prefix attenuation', () => {
    for (let i = 0; i < GEN.steps; i++) {
      if (i === half) pipe.qwenImage21ScalePrefixKv(16, NL, 1.0, 0.5);
      st.stepOnce();
    }
  });
  const img = png('qi21_prefixkv.png', st.decode());
  const m = mse(base.data, img.data);
  console.log('  prefix-KV vs base MSE = ' + m.toFixed(3));
  check(isFinite(m) && m > 0, 'attenuating the cached prefix V changes the image');
}

// ── 7. the VAE seam ────────────────────────────────────────────────────────

console.log('\n[7] VAE encode / decode round-trip');
{
  // The baseline render, back to CHW [0,1].
  const W = base.width, Hpx = base.height;
  const chw = new Float32Array(3 * Hpx * W);
  const plane = Hpx * W;
  for (let i = 0; i < plane; i++) {
    for (let c = 0; c < 3; c++) chw[c * plane + i] = base.data[4 * i + c] / 255;
  }
  const lat = timed('encodeImage', () => pipe.qwenImage21EncodeImage(chw, Hpx, W));
  console.log('  latent ' + lat.rows + 'x' + lat.cols +
              ' hLat=' + lat.hLat + ' wLat=' + lat.wLat);
  check(lat.hLat === Hpx / 16 && lat.wLat === W / 16, 'latent grid is image/16');
  check(lat.cols === 64 * lat.hLat * lat.wLat, 'latent holds 64 channels');
  let nf = 0;
  for (let i = 0; i < lat.data.length; i++) if (!isFinite(lat.data[i])) nf++;
  check(nf === 0, 'the encoded latent is finite');

  const recon = timed('decode', () =>
    pipe.qwenImage21Decode(lat, lat.hLat, lat.wLat));
  png('qi21_vae_roundtrip.png', recon);
  const m = mse(base.data, recon.data);
  console.log('  round-trip MSE = ' + m.toFixed(3) +
              '  (PSNR ' + (10 * Math.log10(255 * 255 / Math.max(m, 1e-9))).toFixed(2) + ' dB)');
  check(isFinite(m), 'the round-trip is finite');
  check(m < 200, 'the round-trip reconstructs the image (MSE < 200)');
}

// ── 8. releasing the text encoder ──────────────────────────────────────────

console.log('\n[8] releasing the 8.5 GiB text encoder');
{
  check(pipe.qwenImage21TextEncoderResident(), 'the encoder starts resident');
  const st = pipe.prime(PROMPT, GEN);
  pipe.qwenImage21ReleaseTextEncoder();
  check(!pipe.qwenImage21TextEncoderResident(), 'the encoder is released');
  timed('8 steps with no text encoder resident', () => {
    for (let i = 0; i < GEN.steps; i++) st.stepOnce();
  });
  const img = png('qi21_released_te.png', st.decode());
  const m = mse(base.data, img.data);
  console.log('  released-TE vs base MSE = ' + m.toFixed(6));
  check(m === 0, 'a primed state steps identically with the encoder gone');

  let threw = false;
  try { pipe.qwenImage21EncodePrompt('anything'); } catch (e) { threw = true; }
  check(threw, 'encoding a new prompt throws once the encoder is released');

  pipe.qwenImage21ReloadTextEncoder(MODEL_DIR, '', { quantizeWeights: true });
  check(pipe.qwenImage21TextEncoderResident(), 'the encoder reloads');
  const back = pipe.generate(PROMPT, GEN);
  check(mse(base.data, back.data) === 0,
        'a reloaded encoder reproduces the baseline exactly');
}

console.log('\n' + (failures === 0
  ? 'qi21 surface smoke: OK'
  : 'qi21 surface smoke: ' + failures + ' failure(s)'));
assert(failures === 0, 'qi21 surface smoke had ' + failures + ' failure(s)');
