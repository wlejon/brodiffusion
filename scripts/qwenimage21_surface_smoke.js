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
//   9. every hook is a LIST: Add/Clear/Count, two bindings over disjoint block
//      ranges, and Set as "replace the list with one entry".
//  10. the post-tanh gate delta, which has unit authority over channels the
//      pre-tanh mod delta cannot reach.
//  11. gate masks name their sublayer: attn-only, mlp-only and both are three
//      different pictures over the same region.
//  12. the four gate multipliers are independent: attn on the image rows is
//      not attn on both row sets, and only the txt half re-extracts.
//  13. the prefix KV dial is idempotent and takes a per-row weight — per-token
//      prompt weighting on the cache, with no re-encode.
//  14. the prefix cache slots: save, blend, clear.
//  15. the prompt memo: prime() works on an already-encoded prompt after the
//      text encoder is gone, and names the prompt when it cannot.
//  16. the edit bindings: encodePromptImages / primeEdit / conditionImages.
//  17. a wrong-length gate mask throws, naming the length it wanted.
//  18. the between-step control schedule: a flat alpha reproduces the
//      prime-time setControl render to the pixel, a late-only alpha does not,
//      an explicit direction schedules the same as a bank name, two half
//      schedules sum to one whole, and a re-extract is priced at 512 and 1024.
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

  // Clear it. The dial is a model hook now, not cache content, so leaving it
  // armed steers every later section — which is the point of the change, and
  // exactly what this script used to get away with when the same call was an
  // in-place multiply that the next prime() threw away.
  pipe.qwenImage21ClearPrefixKvScales();
  check(pipe.qwenImage21PrefixKvScaleCount() === 0, 'the dial is cleared');
  check(mse(base.data, pipe.generate(PROMPT, GEN).data) === 0,
        'clearing the dial restores baseline');
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

// ── 9. every hook is a list ────────────────────────────────────────────────
//
// The single-slot surface is what the research round ran out of first: a
// second SetGateScale replaced the first, so nothing could scale the shallow
// blocks and delta the deep ones in the same generation.

console.log('\n[9] the binding lists: Add / Clear / Count');
{
  check(pipe.qwenImage21GateScaleCount() === 0, 'the gate-scale list starts empty');
  check(pipe.qwenImage21ModDeltaCount() === 0, 'the mod-delta list starts empty');
  check(pipe.qwenImage21GateDeltaCount() === 0, 'the gate-delta list starts empty');
  check(pipe.qwenImage21GateMaskCount() === 0, 'the gate-mask list starts empty');
  check(pipe.qwenImage21PrefixKvScaleCount() === 0, 'the prefix-kv list starts empty');

  // Two bindings over DISJOINT halves of the stack, which the old surface
  // could not hold at once.
  const s0 = pipe.qwenImage21AddGateScale(0.85, 1.0, 1.0, 1.0, 0, 16);
  const s1 = pipe.qwenImage21AddGateScale(1.0, 1.15, 1.0, 1.0, 16, 32);
  check(s0 === 0 && s1 === 1, 'Add returns ascending slot indices');
  check(pipe.qwenImage21GateScaleCount() === 2, 'both bindings are armed');
  const two = timed('two-binding render', () => pipe.generate(PROMPT, GEN));
  png('qi21_two_scales.png', two);
  const mTwo = mse(base.data, two.data);
  console.log('  two disjoint gate scales vs base MSE = ' + mTwo.toFixed(3));
  check(isFinite(mTwo) && mTwo > 0, 'two disjoint gate scales change the image');

  // Only the shallow one: a different picture, which is what proves the deep
  // binding was doing something of its own.
  pipe.qwenImage21ClearGateScales();
  check(pipe.qwenImage21GateScaleCount() === 0, 'Clear empties the list');
  pipe.qwenImage21AddGateScale(0.85, 1.0, 1.0, 1.0, 0, 16);
  const one = pipe.generate(PROMPT, GEN);
  const mOne = mse(two.data, one.data);
  console.log('  shallow-only vs both MSE = ' + mOne.toFixed(3));
  check(isFinite(mOne) && mOne > 0, 'the deep binding contributed');

  // Set is Add-after-Clear.
  pipe.qwenImage21SetGateScale(0.85, 1.0, 1.0, 1.0, 0, 16);
  check(pipe.qwenImage21GateScaleCount() === 1, 'Set replaces the list');
  check(mse(one.data, pipe.generate(PROMPT, GEN).data) === 0,
        'Set is exactly Add after Clear');

  pipe.qwenImage21ClearGateScales();
  check(mse(base.data, pipe.generate(PROMPT, GEN).data) === 0,
        'clearing the list restores baseline');
}

// ── 10. the post-tanh gate delta ───────────────────────────────────────────
//
// 74% of gate2's channels sit where tanh'(g) < 0.05, so a pre-tanh mod delta
// has almost no authority over them. This adds AFTER the tanh.

console.log('\n[10] post-tanh gate delta, blocks [16, 32)');
{
  const d = { rows: 1, cols: 2 * H, data: new Float32Array(2 * H) };
  for (let i = 0; i < H; i++) d.data[i] = 0.02;            // attn half
  pipe.qwenImage21SetGateDelta(d, 16, 32, 'target');
  check(pipe.qwenImage21GateDeltaCount() === 1, 'the gate delta is armed');
  const img = timed('gate-delta render', () => pipe.generate(PROMPT, GEN));
  png('qi21_gatedelta.png', img);
  const m = mse(base.data, img.data);
  console.log('  gate-delta (+0.02 attn) vs base MSE = ' + m.toFixed(3));
  check(isFinite(m) && m > 0, 'a post-tanh gate delta changes the image');

  // The mlp half is a separate axis.
  const d2 = { rows: 1, cols: 2 * H, data: new Float32Array(2 * H) };
  for (let i = H; i < 2 * H; i++) d2.data[i] = 0.02;       // mlp half
  pipe.qwenImage21SetGateDelta(d2, 16, 32, 'target');
  const img2 = pipe.generate(PROMPT, GEN);
  const m2 = mse(img.data, img2.data);
  console.log('  attn-half vs mlp-half MSE = ' + m2.toFixed(3));
  check(isFinite(m2) && m2 > 0, 'the attn and mlp halves are separate axes');

  pipe.qwenImage21ClearGateDeltas();
  check(mse(base.data, pipe.generate(PROMPT, GEN).data) === 0,
        'clearing the gate delta restores baseline');
}

// ── the joint-sequence geometry every mask depends on ──────────────────────
//
// A mask addresses the WHOLE joint sequence: text rows first, then the image
// tokens row-major. At 512x512 that is a 32x32 token grid, one token per
// 16x16 px, so the token covering pixel (px, py) is at
//     textRows + (py >> 4) * 32 + (px >> 4)

const GRID = GEN.width / 16;                      // 32 at 512
const IMG_LEN = GRID * GRID;                      // 1024
const TEXT_ROWS = (() => {
  pipe.prime(PROMPT, GEN);
  return pipe.qwenImage21TextRows().rows;
})();
const JOINT = TEXT_ROWS + IMG_LEN;
console.log('\njoint sequence: ' + TEXT_ROWS + ' text rows + ' + IMG_LEN +
            ' image tokens (' + GRID + 'x' + GRID + ') = ' + JOINT);

// A mask that damps the left half of the image and leaves the text alone.
function leftHalfMask(value) {
  const m = { rows: JOINT, cols: 1, data: new Float32Array(JOINT) };
  m.data.fill(1.0);
  for (let y = 0; y < GRID; y++) {
    for (let x = 0; x < GRID / 2; x++) {
      m.data[TEXT_ROWS + y * GRID + x] = value;
    }
  }
  return m;
}

// ── 11. gate masks name their sublayer ─────────────────────────────────────

console.log('\n[11] gate mask per sublayer, left half, blocks [12, 32)');
{
  const mask = leftHalfMask(0.5);
  const shots = {};
  for (const which of ['both', 'attn', 'mlp']) {
    pipe.qwenImage21SetGateMask(mask, 12, 32, which);
    check(pipe.qwenImage21GateMaskCount() === 1, 'the ' + which + ' mask is armed');
    shots[which] = timed('mask render (' + which + ')',
                         () => pipe.generate(PROMPT, GEN));
    png('qi21_mask_' + which + '.png', shots[which]);
    const m = mse(base.data, shots[which].data);
    console.log('  mask ' + which + ' vs base MSE = ' + m.toFixed(3));
    check(isFinite(m) && m > 0, "a '" + which + "' mask changes the image");
  }
  const mAB = mse(shots.attn.data, shots.mlp.data);
  const mAO = mse(shots.attn.data, shots.both.data);
  console.log('  attn vs mlp MSE = ' + mAB.toFixed(3) +
              ', attn vs both MSE = ' + mAO.toFixed(3));
  check(mAB > 0, 'the attn and mlp halves are different pictures');
  check(mAO > 0, "'attn' is not the blunt 'both'");

  // 'both' is exactly 'attn' and 'mlp' armed together.
  pipe.qwenImage21ClearGateMasks();
  pipe.qwenImage21AddGateMask(mask, 12, 32, 'attn');
  pipe.qwenImage21AddGateMask(mask, 12, 32, 'mlp');
  check(pipe.qwenImage21GateMaskCount() === 2, 'two masks, one per sublayer');
  check(mse(shots.both.data, pipe.generate(PROMPT, GEN).data) === 0,
        "'both' is bit-identically 'attn' + 'mlp'");

  pipe.qwenImage21ClearGateMasks();
  check(mse(base.data, pipe.generate(PROMPT, GEN).data) === 0,
        'clearing the masks restores baseline');
}

// ── 12. the four independent gate multipliers ──────────────────────────────

console.log('\n[12] four gate multipliers, blocks [16, 32)');
{
  // The rank-1 form IS the four products.
  pipe.qwenImage21SetGateScale(1.2, 1.0, 1.1, 0.9, 16, 32);
  const viaProduct = timed('rank-1 render', () => pipe.generate(PROMPT, GEN));
  pipe.qwenImage21SetGateScaleRows(1.2 * 1.1, 1.2 * 0.9, 1.0 * 1.1, 1.0 * 0.9,
                                   16, 32);
  check(mse(viaProduct.data, pipe.generate(PROMPT, GEN).data) === 0,
        'the rank-1 form is four multipliers spelled out');

  // The thing the product cannot say: the attention gate on the IMAGE rows.
  pipe.qwenImage21SetGateScaleRows(1.0, 1.25, 1.0, 1.0, 16, 32);
  const imgOnly = timed('attn.img render', () => pipe.generate(PROMPT, GEN));
  png('qi21_attn_img.png', imgOnly);
  pipe.qwenImage21SetGateScale(1.25, 1.0, 1.0, 1.0, 16, 32);   // both row sets
  const bothRows = pipe.generate(PROMPT, GEN);
  const mIT = mse(imgOnly.data, bothRows.data);
  console.log('  attn.img vs attn.both MSE = ' + mIT.toFixed(3));
  check(isFinite(mIT) && mIT > 0,
        'attn on the image rows is not attn on both row sets');

  // The txt half alone is a third picture again.
  pipe.qwenImage21SetGateScaleRows(1.25, 1.0, 1.0, 1.0, 16, 32);
  const txtOnly = pipe.generate(PROMPT, GEN);
  const mTI = mse(txtOnly.data, imgOnly.data);
  console.log('  attn.txt vs attn.img MSE = ' + mTI.toFixed(3));
  check(isFinite(mTI) && mTI > 0, 'the txt and img row sets are separate axes');

  // ...and they compose: txt-only plus img-only is the binding that sets both.
  pipe.qwenImage21ClearGateScales();
  pipe.qwenImage21AddGateScaleRows(1.25, 1.0, 1.0, 1.0, 16, 32);
  pipe.qwenImage21AddGateScaleRows(1.0, 1.25, 1.0, 1.0, 16, 32);
  const composed = pipe.generate(PROMPT, GEN);
  pipe.qwenImage21SetGateScaleRows(1.25, 1.25, 1.0, 1.0, 16, 32);
  check(mse(composed.data, pipe.generate(PROMPT, GEN).data) === 0,
        'the four axes compose exactly');

  pipe.qwenImage21ClearGateScales();
  check(mse(base.data, pipe.generate(PROMPT, GEN).data) === 0,
        'clearing restores baseline');
}

// ── 13. the prefix KV dial, per row ────────────────────────────────────────

console.log('\n[13] prefix KV: idempotent, and weighted per row');
{
  function stepAll(arm) {
    const st = pipe.prime(PROMPT, GEN);
    if (arm) arm();
    for (let i = 0; i < GEN.steps; i++) st.stepOnce();
    return st.decode();
  }

  pipe.qwenImage21ScalePrefixKv(16, NL, 1.0, 0.5);
  const once = timed('prefix-kv render', () => stepAll());
  png('qi21_prefixkv_rows_base.png', once);
  const mOnce = mse(base.data, once.data);
  console.log('  prefix-kv 0.5 vs base MSE = ' + mOnce.toFixed(3));
  check(isFinite(mOnce) && mOnce > 0, 'the prefix-KV dial changes the image');

  // IDEMPOTENT: arming it again is the same picture. The old in-place
  // version squared here, so it could not be used from generate() at all.
  pipe.qwenImage21ScalePrefixKv(16, NL, 1.0, 0.5);
  check(mse(once.data, stepAll().data) === 0,
        'setting the same dial twice is bit-identical');

  // A whole generate() loop honours it too — the in-place version compounded
  // once per step and blew the image out.
  const viaGenerate = pipe.generate(PROMPT, GEN);
  check(mse(once.data, viaGenerate.data) === 0,
        'generate() and a manual step loop agree under the dial');

  // An all-ones row weight IS the broadcast.
  const allRows = { rows: TEXT_ROWS, cols: 1, data: new Float32Array(TEXT_ROWS) };
  allRows.data.fill(1.0);
  pipe.qwenImage21ScalePrefixKv(16, NL, 1.0, 0.5, allRows);
  check(mse(once.data, pipe.generate(PROMPT, GEN).data) === 0,
        'an all-ones row weight is the broadcast it generalises');

  // ...and an all-zeros one is the identity.
  const noRows = { rows: TEXT_ROWS, cols: 1, data: new Float32Array(TEXT_ROWS) };
  pipe.qwenImage21ScalePrefixKv(16, NL, 1.0, 0.5, noRows);
  check(mse(base.data, pipe.generate(PROMPT, GEN).data) === 0,
        'an all-zeros row weight is the identity');

  // Per-token prompt weighting: attenuate only the last third of the rows.
  const tail = { rows: TEXT_ROWS, cols: 1, data: new Float32Array(TEXT_ROWS) };
  for (let i = Math.floor((2 * TEXT_ROWS) / 3); i < TEXT_ROWS; i++) tail.data[i] = 1.0;
  pipe.qwenImage21ScalePrefixKv(16, NL, 1.0, 0.5, tail);
  const weighted = timed('row-weighted render', () => pipe.generate(PROMPT, GEN));
  png('qi21_prefixkv_tail.png', weighted);
  const mW = mse(once.data, weighted.data);
  console.log('  tail-only vs all-rows MSE = ' + mW.toFixed(3) +
              ', tail-only vs base MSE = ' + mse(base.data, weighted.data).toFixed(3));
  check(isFinite(mW) && mW > 0, 'a per-row weight selects which rows are damped');

  // A wrong-length row weight throws rather than scaling the wrong rows.
  let threw = false, why = '';
  try {
    const bad = { rows: TEXT_ROWS + 3, cols: 1,
                  data: new Float32Array(TEXT_ROWS + 3) };
    bad.data.fill(1.0);
    pipe.qwenImage21ScalePrefixKv(16, NL, 1.0, 0.5, bad);
    pipe.generate(PROMPT, GEN);
  } catch (e) { threw = true; why = String(e); }
  check(threw, 'a wrong-length row weight throws');
  console.log('  ' + why);

  pipe.qwenImage21ClearPrefixKvScales();
  check(mse(base.data, pipe.generate(PROMPT, GEN).data) === 0,
        'clearing the dial restores baseline');
}

// ── 14. the prefix cache slots ─────────────────────────────────────────────
//
// After the extract step the cache IS the conditioning, so mixing two of them
// is a prompt interpolation that costs no re-encode. The two must share a
// layout, which in practice means two prompts that tokenize to the same
// length — so the second prompt here is the first with words swapped.

console.log('\n[14] prefix cache slots: save, blend, clear');
{
  const ALT = 'a lighthouse on a rocky coast at sunrise, gentle clouds';
  check(pipe.qwenImage21PrefixSlots() >= 2, 'there are at least two slots');
  check(!pipe.qwenImage21PrefixSlotValid(0), 'slot 0 starts empty');

  // Prime the alternative prompt, extract, and park the cache in slot 0.
  const altState = pipe.prime(ALT, GEN);
  altState.stepOnce();                      // the extract step
  pipe.qwenImage21SavePrefixCache(0);
  check(pipe.qwenImage21PrefixSlotValid(0), 'slot 0 holds a cache');

  const altRows = pipe.qwenImage21TextRows().rows;
  console.log('  alt prompt is ' + altRows + ' text rows (base ' + TEXT_ROWS + ')');
  if (altRows !== TEXT_ROWS) {
    console.log('  skipping the blend: the two prompts tokenize differently');
    check(true, 'blend skipped (layouts differ)');
  } else {
    const st = pipe.prime(PROMPT, GEN);
    st.stepOnce();                          // extract the base prompt
    pipe.qwenImage21BlendPrefixCache(0, 0.5);
    for (let i = 1; i < GEN.steps; i++) st.stepOnce();
    const img = png('qi21_blend_half.png', st.decode());
    const m = mse(base.data, img.data);
    console.log('  50% blended prefix vs base MSE = ' + m.toFixed(3));
    check(isFinite(m) && m > 0, 'blending the prefix cache changes the image');

    // alpha = 0 is a no-op on the live cache.
    const st0 = pipe.prime(PROMPT, GEN);
    st0.stepOnce();
    pipe.qwenImage21BlendPrefixCache(0, 0.0);
    for (let i = 1; i < GEN.steps; i++) st0.stepOnce();
    check(mse(base.data, st0.decode().data) === 0, 'alpha = 0 changes nothing');
  }

  pipe.qwenImage21ClearPrefixSlots();
  check(!pipe.qwenImage21PrefixSlotValid(0), 'clearing drops the saved cache');
}

// ── 15. the prompt memo ────────────────────────────────────────────────────
//
// prime() used to throw for ANY prompt once the text encoder was released,
// including one it had just encoded — so the 8.5 GiB saving cost you the
// ability to re-prime the prompt you were studying.

console.log('\n[15] the prompt memo survives releasing the text encoder');
{
  if (!pipe.qwenImage21TextEncoderResident()) {
    pipe.qwenImage21ReloadTextEncoder(MODEL_DIR, '', { quantizeWeights: true });
  }
  pipe.qwenImage21ClearPromptMemo();
  check(pipe.qwenImage21MemoizedPrompts().length === 0, 'the memo starts empty');

  pipe.generate(PROMPT, GEN);               // fills the memo
  const memo = pipe.qwenImage21MemoizedPrompts();
  console.log('  memoized: ' + JSON.stringify(memo));
  check(memo.indexOf(PROMPT) >= 0, 'the generated prompt is memoized');

  pipe.qwenImage21ReleaseTextEncoder();
  check(!pipe.qwenImage21TextEncoderResident(), 'the encoder is released');

  // The whole point: re-prime the SAME prompt with no encoder resident.
  const st = pipe.prime(PROMPT, GEN);
  for (let i = 0; i < GEN.steps; i++) st.stepOnce();
  check(mse(base.data, st.decode().data) === 0,
        'a memoized prompt re-primes identically with no encoder');

  // An unknown prompt throws, and the message names it.
  let threw = false, why = '';
  const UNKNOWN = 'a prompt this pipeline has never encoded';
  try { pipe.prime(UNKNOWN, GEN); } catch (e) { threw = true; why = String(e); }
  check(threw, 'an unmemoized prompt still throws');
  check(why.indexOf(UNKNOWN) >= 0, 'the error names the prompt it cannot encode');
  console.log('  ' + why);

  pipe.qwenImage21ReloadTextEncoder(MODEL_DIR, '', { quantizeWeights: true });
}

// ── 16. the edit bindings ──────────────────────────────────────────────────
//
// A condition image enters as prefix rows, modulated from t = 0 like the text
// is — so it is cached with the text and every image-side hook above applies
// to a generation driven by it.

console.log('\n[16] condition images: encodePromptImages / primeEdit / generate');
{
  const W = base.width, Hpx = base.height;
  const plane = Hpx * W;
  const chw = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    for (let c = 0; c < 3; c++) chw[c * plane + i] = base.data[4 * i + c] / 255;
  }
  const condition = { pixels: chw, height: Hpx, width: W, channels: 3 };
  const EDIT = 'the same lighthouse, in heavy fog';

  // Returns a record, not a bare tensor: { embeds, mask, ids, dropIdx,
  // imagePadMask, imageRuns }. imagePadMask marks the condition-image slots
  // inside the joint prefix, which is what a mask over a conditioned
  // generation has to line up with.
  const taps = timed('encodePromptImages', () =>
    pipe.qwenImage21EncodePromptImages(EDIT, [condition]));
  const emb = taps.embeds;
  console.log('  prompt+image rows = ' + emb.rows + 'x' + emb.cols +
              ', image runs = ' + JSON.stringify(taps.imageRuns));
  check(emb.rows > TEXT_ROWS, 'the image run adds rows to the prefix');
  check(emb.cols === TH, 'the rows are text-hidden wide');

  const edited = timed('conditionImages generate', () =>
    pipe.generate(EDIT, Object.assign({}, GEN, {
      conditionImages: [condition],
      outputResolution: [GEN.width, GEN.height],
    })));
  png('qi21_edit.png', edited);
  const mE = mse(base.data, edited.data);
  console.log('  edited vs base MSE = ' + mE.toFixed(3));
  check(isFinite(mE) && mE > 0, 'a condition image changes the render');

  // The same edit without the condition image is a different picture, which
  // is what shows the image was doing the work rather than the prompt.
  const textOnly = pipe.generate(EDIT, GEN);
  const mT = mse(edited.data, textOnly.data);
  console.log('  with vs without the condition image MSE = ' + mT.toFixed(3));
  check(isFinite(mT) && mT > 0, 'the condition image drives the render');

  // An image-side hook applies to an image-conditioned generation too.
  pipe.qwenImage21SetGateScaleRows(1.0, 1.2, 1.0, 1.0, 16, 32);
  const hooked = pipe.generate(EDIT, Object.assign({}, GEN, {
    conditionImages: [condition],
    outputResolution: [GEN.width, GEN.height],
  }));
  const mH = mse(edited.data, hooked.data);
  console.log('  image-conditioned + attn.img MSE = ' + mH.toFixed(3));
  check(isFinite(mH) && mH > 0, 'an image-side hook steers an edit');
  pipe.qwenImage21ClearGateScales();
}

// ── 17. a wrong-length mask throws ─────────────────────────────────────────
//
// It used to be a silent no-op: the hook armed only when the length matched,
// so an all-zero mask written at img_len instead of textRows + img_len
// rendered the baseline to the pixel and read as "this surface does nothing".

console.log('\n[17] a wrong-length gate mask throws');
{
  const bad = { rows: IMG_LEN, cols: 1, data: new Float32Array(IMG_LEN) };
  pipe.qwenImage21SetGateMask(bad, 0, NL, 'both');
  let threw = false, why = '';
  try { pipe.generate(PROMPT, GEN); } catch (e) { threw = true; why = String(e); }
  check(threw, 'a mask of img_len rows throws instead of doing nothing');
  check(why.indexOf(String(JOINT)) >= 0, 'the message names the joint length');
  console.log('  ' + why);

  pipe.qwenImage21ClearGateMasks();
  check(mse(base.data, pipe.generate(PROMPT, GEN).data) === 0,
        'clearing the bad mask restores baseline');
}

// ── 18. the between-step control schedule ──────────────────────────────────
//
// The conditioning axes are the only surface on this model with real steering
// authority, and until now they landed once, before step 0. A schedule
// re-aims one per step by rebuilding the text rows from the PRIMED embedding
// plus that step's alpha, which is why a flat schedule has to be pixel-exact
// against the static desk: the arithmetic is the same arithmetic.

console.log('\n[18] between-step control schedule');
{
  // The same diff-of-means recipe as [4], rebuilt here so the section stands
  // on its own (a schedule is the first thing a reader will want to copy).
  const SCENES = [
    'a harbour at dusk',
    'a kitchen table with fruit',
    'a city street after rain',
  ];
  const warm = new Float64Array(TH);
  const cold = new Float64Array(TH);
  function accum(prompt, sink) {
    const e = pipe.encodeConditioning(prompt);
    for (let r = 0; r < e.rows; r++) {
      for (let c = 0; c < TH; c++) sink[c] += e.data[r * e.cols + c] / e.rows;
    }
  }
  timed('6 conditioning encodes', () => {
    for (const s of SCENES) {
      accum(s + ', warm golden light', warm);
      accum(s + ', cold blue light', cold);
    }
  });
  const dir = new Float32Array(TH);
  let norm = 0;
  for (let c = 0; c < TH; c++) {
    dir[c] = (warm[c] - cold[c]) / SCENES.length;
    norm += dir[c] * dir[c];
  }
  norm = Math.sqrt(norm);
  for (let c = 0; c < TH; c++) dir[c] /= norm;

  const AXIS = 'schedWarmth';
  const A = 3.0;
  const flat = new Float32Array(GEN.steps);
  for (let i = 0; i < GEN.steps; i++) flat[i] = A;

  // (a) the static reference: the axis applied once, at prime time.
  pipe.setControlVector(AXIS, dir, A, norm);
  const statImg = png('qi21_sched_static.png', timed('static-axis render',
    () => pipe.generate(PROMPT, GEN)));
  const mStat = mse(base.data, statImg.data);
  console.log('  static axis vs base MSE = ' + mStat.toFixed(3));
  check(isFinite(mStat) && mStat > 0, 'the static axis moves the image');

  // The axis stays in the bank at weight 0, so the schedule below resolves it
  // by name while the prime-time seam contributes nothing.
  pipe.setControl(AXIS, 0);
  check(mse(base.data, pipe.generate(PROMPT, GEN).data) === 0,
        'weight 0 is the baseline again');

  // (b) the same axis as a FLAT schedule over every step.
  check(pipe.qwenImage21ControlScheduleCount() === 0, 'no schedule armed yet');
  const slot = pipe.qwenImage21SetControlSchedule(AXIS, flat, 0, GEN.steps);
  check(slot === 0, 'Set returns slot 0');
  check(pipe.qwenImage21ControlScheduleCount() === 1, 'one schedule armed');
  const flatImg = png('qi21_sched_flat.png', timed('flat-schedule render',
    () => pipe.generate(PROMPT, GEN)));
  const mFlat = mse(statImg.data, flatImg.data);
  console.log('  flat schedule vs prime-time setControl MSE = ' + mFlat);
  check(mFlat === 0,
        'alpha = ' + A + ' on every step reproduces the prime-time render');

  // (c) the same alpha on the LAST HALF only is a different picture — which
  //     is the whole point, and the thing the static seam cannot express.
  const late = new Float32Array(GEN.steps);
  for (let i = GEN.steps >> 1; i < GEN.steps; i++) late[i] = A;
  pipe.qwenImage21SetControlSchedule(AXIS, late, 0, GEN.steps);
  const lateImg = png('qi21_sched_late.png', timed('late-only render',
    () => pipe.generate(PROMPT, GEN)));
  const mLate = mse(statImg.data, lateImg.data);
  const mLateBase = mse(base.data, lateImg.data);
  console.log('  late-only vs flat MSE = ' + mLate.toFixed(3) +
              ', vs base MSE = ' + mLateBase.toFixed(3));
  check(isFinite(mLate) && mLate > 0, 'a late-only schedule is not the static render');
  check(isFinite(mLateBase) && mLateBase > 0, 'a late-only schedule still moves the image');

  // (d) an all-zero schedule renders the baseline: the rows go back, they do
  //     not stick at whatever the last nonzero step installed.
  const zero = new Float32Array(GEN.steps);
  pipe.qwenImage21SetControlSchedule(AXIS, zero, 0, GEN.steps);
  check(mse(base.data, pipe.generate(PROMPT, GEN).data) === 0,
        'an all-zero schedule is the baseline');

  // (e) the explicit-direction form: the same Float32Array in place of the
  //     name, with the axis scale as the trailing argument. For the research
  //     that is a diff-of-means axis that is in no bank.
  pipe.qwenImage21SetControlSchedule(dir, flat, 0, GEN.steps, norm);
  const dirImg = pipe.generate(PROMPT, GEN);
  console.log('  explicit direction vs bank name MSE = ' + mse(statImg.data, dirImg.data));
  check(mse(statImg.data, dirImg.data) === 0,
        'an explicit direction schedules identically to the bank axis');

  // (f) schedules compose: two half-amplitude slots sum to the whole.
  const half = new Float32Array(GEN.steps);
  for (let i = 0; i < GEN.steps; i++) half[i] = A / 2;
  pipe.qwenImage21ClearControlSchedules();
  check(pipe.qwenImage21AddControlSchedule(AXIS, half, 0, GEN.steps) === 0,
        'Add returns slot 0');
  check(pipe.qwenImage21AddControlSchedule(AXIS, half, 0, GEN.steps) === 1,
        'Add returns slot 1');
  check(pipe.qwenImage21ControlScheduleCount() === 2, 'two slots armed');
  const sumImg = pipe.generate(PROMPT, GEN);
  console.log('  two half schedules vs one whole MSE = ' + mse(statImg.data, sumImg.data));
  check(mse(statImg.data, sumImg.data) === 0, 'two halves sum to the whole');

  // (g) clearing restores the baseline for good.
  pipe.qwenImage21ClearControlSchedules();
  check(pipe.qwenImage21ControlScheduleCount() === 0, 'the list is empty');
  pipe.removeControl(AXIS);
  check(mse(base.data, pipe.generate(PROMPT, GEN).data) === 0,
        'clearing the schedule restores baseline');

  // (h) what a re-extract costs. A schedule whose alpha MOVES every step
  //     rebuilds the rows and drops the prefix cache every step; a flat one
  //     rebuilds once. The difference in per-step wall time IS the
  //     re-extract, measured rather than asserted.
  function perStepMs(gen, alphas) {
    pipe.qwenImage21ClearControlSchedules();
    if (alphas) pipe.qwenImage21SetControlSchedule(dir, alphas, 0, gen.steps, norm);
    const st = pipe.prime(PROMPT, gen);
    st.stepOnce();                       // step 0 extracts either way
    const t0 = Date.now();
    for (let i = 1; i < gen.steps; i++) st.stepOnce();
    return (Date.now() - t0) / (gen.steps - 1);
  }
  for (const gen of [GEN, Object.assign({}, GEN, { width: 1024, height: 1024 })]) {
    const moving = new Float32Array(gen.steps);
    for (let i = 0; i < gen.steps; i++) moving[i] = A * (i + 1) / gen.steps;
    const flatMs = perStepMs(gen, flat);
    const moveMs = perStepMs(gen, moving);
    console.log('  ' + gen.width + '^2: ' + flatMs.toFixed(1) +
                ' ms/step flat, ' + moveMs.toFixed(1) +
                ' ms/step re-extracting -> ' + (moveMs - flatMs).toFixed(1) +
                ' ms per re-extract');
    check(isFinite(flatMs) && isFinite(moveMs) && moveMs > 0,
          gen.width + '^2 schedule timings are finite');
  }

  pipe.qwenImage21ClearControlSchedules();
  check(mse(base.data, pipe.generate(PROMPT, GEN).data) === 0,
        'the card is back at baseline after the timing runs');
}

console.log('\n' + (failures === 0
  ? 'qi21 surface smoke: OK'
  : 'qi21 surface smoke: ' + failures + ' failure(s)'));
assert(failures === 0, 'qi21 surface smoke had ' + failures + ' failure(s)');
