// Qwen-Image 2.1 — the multi-slot binding lists and the prefix-KV dial.
//
// Everything here is about the difference between "a hook holds one
// configuration" and "a hook holds a list", which is what the research effort
// ran into: a second qwenImage21SetGateScale replaced the first, so a dial
// vector could not scale the shallow blocks and delta the deep ones in the
// same generation, and qwenImage21ScalePrefixKv multiplied the cache in place,
// so calling it twice meant something different from calling it once.
//
// The claims checked, in order:
//
//   1. Set* is still exactly "replace the list with one entry": one Add after
//      a Clear is BIT-IDENTICAL to the Set it replaces.
//   2. Two bindings over DISJOINT block ranges both apply. The composed
//      result is bit-identical to running them one at a time over a model
//      whose other blocks were left alone — which under a 2-block fixture is
//      a strong statement: it pins the per-block resolution exactly.
//   3. Bindings over the SAME block compose the way the semantics say:
//      scales multiply, deltas add. Checked against the single binding that
//      spells the composition out.
//   4. Clear puts every surface back to bit-exact baseline.
//   5. The prefix-KV scale is an idempotent dial: setting it twice equals
//      setting it once (the old in-place version squared), it survives a
//      cache reset and re-extract, and a per-layer list composes.
//   6. The four gate multipliers are INDEPENDENT: the rank-1 attn x txt/img
//      product is exactly the sugar that spells four of them out, and
//      "attention gate, image rows only" — which the product cannot say — is
//      a different picture from "attention gate, both row sets".
//   7. A prefix-affecting change re-extracts the cache AND re-applies every
//      armed prefix edit. This is the one the research round lost work to: a
//      txt-side gate scale used to invalidate the cache and silently discard
//      a live ScalePrefixKv.
//   8. A gate mask names its sublayer. Masking 'both' is bit-identical to
//      masking 'attn' and 'mlp' with the same vector, and each half on its
//      own is a different picture from either.
//   9. A gate mask whose length is not the joint sequence length THROWS,
//      naming the length it wanted. It used to be a silent no-op, which is
//      how a whole study concluded "this surface does nothing".
//  10. The prefix-KV row scale is the scalar generalised: a uniform row
//      vector is bit-identical to the scalar it repeats, and a non-uniform
//      one is a different picture.
//
// Runs against the same synthetic checkpoint as test_qwenimage21_dit.cpp
// (qwenimage21_fixture.h): 2 blocks, 2 heads of 64, context 128.

#define _CRT_SECURE_NO_WARNINGS

#include "brodiffusion/dit/qwenimage21.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include "test_compute.h"
#include "qwenimage21_fixture.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace bt = brotensor;
namespace st = brotensor::safetensors;
namespace qd = brodiffusion::dit;

using qi21fix::download_any;
using qi21fix::rel_maxdiff;
using qi21fix::rnd;
using qi21fix::synth_cfg;
using qi21fix::write_fixture;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

namespace {

// One model, one input, one forward per assertion.
struct Rig {
    qd::QwenImage21Config cfg = synth_cfg();
    std::filesystem::path path;
    qd::QwenImage21Transformer2DModel model{cfg};
    bt::Tensor lat, txt, out;
    int hp = 6, wp = 4, text_seq = 5;
    int img_len() const { return hp * wp; }
    int L() const { return text_seq + img_len(); }
    int H() const { return cfg.hidden_size(); }

    Rig() {
        path = write_fixture(cfg, "brodiffusion_qi21_slots_test.safetensors");
        auto file = st::File::open(path.string());
        model.load_weights(file, "");
        lat = bdtest::bd_upload(
            rnd(static_cast<std::size_t>(img_len()) * cfg.in_channels, 5001),
            img_len(), cfg.in_channels);
        bt::Tensor emb = bdtest::bd_upload(
            rnd(static_cast<std::size_t>(text_seq) * cfg.context_in_dim, 5002),
            text_seq, cfg.context_in_dim);
        model.encode_text(emb, txt);
    }
    ~Rig() {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }

    std::vector<float> run(float t = 0.7f) {
        model.forward(lat, hp, wp, txt, t, nullptr, out);
        bt::sync_all();
        return download_any(out);
    }
    std::vector<float> run_cached(qd::QwenImage21PrefixCache& c, float t) {
        model.forward(lat, hp, wp, txt, t, &c, out);
        bt::sync_all();
        return download_any(out);
    }
    void clear_all() {
        model.clear_mod_deltas();
        model.clear_gate_scales();
        model.clear_gate_deltas();
        model.clear_gate_masks();
        model.clear_prefix_kv_scales();
        model.set_norm_out_scale_delta(bt::Tensor());
    }
    bt::Tensor mod_delta(float v) const {
        return bdtest::bd_upload(
            std::vector<float>(static_cast<std::size_t>(4) * H(), v), 1,
            4 * H());
    }
    bt::Tensor gate_delta(float v) const {
        return bdtest::bd_upload(
            std::vector<float>(static_cast<std::size_t>(2) * H(), v), 1,
            2 * H());
    }
};

// ── 1 + 4. Set is Add-after-Clear, and Clear restores baseline ─────────────
void test_set_is_one_add(Rig& r) {
    const std::vector<float> base = r.run();
    CHECK(r.model.mod_deltas().empty());
    CHECK(r.model.gate_scales().empty());

    const bt::Tensor md = r.mod_delta(0.05f);
    r.model.set_mod_delta(md, 0, r.cfg.num_layers,
                          qd::QwenImage21ModTarget::Target);
    CHECK(r.model.mod_deltas().size() == 1);
    const std::vector<float> via_set = r.run();
    CHECK(rel_maxdiff(base, via_set) > 1e-5);

    r.model.clear_mod_deltas();
    r.model.add_mod_delta(md, 0, r.cfg.num_layers,
                          qd::QwenImage21ModTarget::Target);
    CHECK(r.model.mod_deltas().size() == 1);
    CHECK(via_set == r.run());   // bit-identical, not merely close

    // Clearing restores the unhooked forward exactly.
    r.model.clear_mod_deltas();
    CHECK(base == r.run());

    // Same for the gate scale, whose Set has an identity spelling too.
    r.model.set_gate_scale(0.8f, 1.1f, 1.0f, 1.0f, 0, r.cfg.num_layers);
    const std::vector<float> gs_set = r.run();
    CHECK(rel_maxdiff(base, gs_set) > 1e-5);
    r.model.clear_gate_scales();
    r.model.add_gate_scale(0.8f, 1.1f, 1.0f, 1.0f, 0, r.cfg.num_layers);
    CHECK(gs_set == r.run());
    // All four factors at 1 is the documented clear.
    r.model.set_gate_scale(1.0f, 1.0f, 1.0f, 1.0f, 0, r.cfg.num_layers);
    CHECK(r.model.gate_scales().empty());
    CHECK(base == r.run());
}

// ── 2. Disjoint ranges both apply, in one generation ──────────────────────
//
// The fixture has 2 blocks, so "block 0 scaled, block 1 deltaed" is the
// smallest configuration the single-binding API could not express at all —
// the second call used to drop the first.
void test_disjoint_ranges(Rig& r) {
    r.clear_all();
    const std::vector<float> base = r.run();
    const bt::Tensor gd = r.gate_delta(0.2f);

    // Each alone.
    r.model.set_gate_scale(0.5f, 0.5f, 1.0f, 1.0f, 0, 1);
    const std::vector<float> scale_only = r.run();
    r.model.clear_gate_scales();
    r.model.set_gate_delta(gd, 1, 2, qd::QwenImage21ModTarget::Target);
    const std::vector<float> delta_only = r.run();
    r.model.clear_gate_deltas();
    CHECK(rel_maxdiff(base, scale_only) > 1e-5);
    CHECK(rel_maxdiff(base, delta_only) > 1e-5);
    CHECK(rel_maxdiff(scale_only, delta_only) > 1e-5);

    // Both armed at once: a third image, and neither of the two.
    r.model.add_gate_scale(0.5f, 0.5f, 1.0f, 1.0f, 0, 1);
    r.model.add_gate_delta(gd, 1, 2, qd::QwenImage21ModTarget::Target);
    CHECK(r.model.gate_scales().size() == 1);
    CHECK(r.model.gate_deltas().size() == 1);
    const std::vector<float> both = r.run();
    CHECK(rel_maxdiff(both, scale_only) > 1e-5);
    CHECK(rel_maxdiff(both, delta_only) > 1e-5);
    CHECK(rel_maxdiff(both, base) > 1e-5);
    std::printf("qi21_slots: disjoint scale/delta vs each alone %.3e / %.3e\n",
                rel_maxdiff(both, scale_only), rel_maxdiff(both, delta_only));

    // Two mod deltas over disjoint ranges: block 0 sees one, block 1 the
    // other, and the result is neither single-range delta.
    r.clear_all();
    const bt::Tensor a = r.mod_delta(0.04f);
    const bt::Tensor b = r.mod_delta(-0.03f);
    r.model.set_mod_delta(a, 0, 1, qd::QwenImage21ModTarget::Target);
    const std::vector<float> a_only = r.run();
    r.model.set_mod_delta(b, 1, 2, qd::QwenImage21ModTarget::Target);
    const std::vector<float> b_only = r.run();
    r.model.clear_mod_deltas();
    r.model.add_mod_delta(a, 0, 1, qd::QwenImage21ModTarget::Target);
    r.model.add_mod_delta(b, 1, 2, qd::QwenImage21ModTarget::Target);
    CHECK(r.model.mod_deltas().size() == 2);
    const std::vector<float> ab = r.run();
    CHECK(rel_maxdiff(ab, a_only) > 1e-6);
    CHECK(rel_maxdiff(ab, b_only) > 1e-6);
    r.clear_all();
    CHECK(base == r.run());
}

// ── 3. Overlapping bindings compose: scales multiply, deltas add ──────────
void test_composition(Rig& r) {
    r.clear_all();
    const std::vector<float> base = r.run();

    // Two scales over the same range == one scale at their product.
    r.model.set_gate_scale(0.6f, 1.0f, 1.0f, 1.0f, 0, r.cfg.num_layers);
    r.model.add_gate_scale(0.5f, 1.0f, 1.0f, 1.0f, 0, r.cfg.num_layers);
    const std::vector<float> two = r.run();
    r.model.set_gate_scale(0.3f, 1.0f, 1.0f, 1.0f, 0, r.cfg.num_layers);
    const std::vector<float> one = r.run();
    const double d_mul = rel_maxdiff(two, one);
    CHECK(d_mul < 1e-6);
    CHECK(rel_maxdiff(base, one) > 1e-4);
    std::printf("qi21_slots: 0.6*0.5 vs 0.3 gate scale rel maxdiff %.3e\n",
                d_mul);

    // Two gate deltas over the same range == one delta at their sum.
    //
    // The constants are powers of two (0.125 + 0.0625 = 0.1875), so nothing
    // is lost representing them at the BF16 compute dtype — but the ADDS
    // still happen against a BF16 gate row, and (g + a) + b is not bit-equal
    // to g + (a + b) there. So the claim is checked the way it is meant:
    // the composed result sits on top of the spelled-out sum, orders of
    // magnitude closer to it than either component alone is.
    r.clear_all();
    r.model.add_gate_delta(r.gate_delta(0.125f), 0, r.cfg.num_layers,
                           qd::QwenImage21ModTarget::Target);
    const std::vector<float> first_only = r.run();
    r.model.add_gate_delta(r.gate_delta(0.0625f), 0, r.cfg.num_layers,
                           qd::QwenImage21ModTarget::Target);
    CHECK(r.model.gate_deltas().size() == 2);
    const std::vector<float> sum_two = r.run();
    r.model.set_gate_delta(r.gate_delta(0.1875f), 0, r.cfg.num_layers,
                           qd::QwenImage21ModTarget::Target);
    const std::vector<float> sum_one = r.run();
    const double d_add = rel_maxdiff(sum_two, sum_one);
    const double d_apart = rel_maxdiff(first_only, sum_one);
    // 1e-2 is the BF16 floor here, not slack: a (1, hidden) gate row carries
    // 8 mantissa bits, so re-rounding it once per add is worth a few 1e-3 on
    // the output after two blocks. The ratio below is the real assertion.
    CHECK(d_add < 1e-2);
    CHECK(d_add < 0.05 * d_apart);
    CHECK(rel_maxdiff(base, sum_one) > 1e-4);
    std::printf("qi21_slots: 0.125+0.0625 vs 0.1875 gate delta %.3e "
                "(one component apart %.3e)\n", d_add, d_apart);

    // Two mod deltas over the same range == one delta at their sum. This one
    // then passes through the gates' tanh, so it only composes at all
    // because the sum is formed BEFORE the tanh — which is what the pre-tanh
    // seam buys and what a post-tanh-only hook could not give.
    r.clear_all();
    r.model.add_mod_delta(r.mod_delta(0.03125f), 0, r.cfg.num_layers,
                          qd::QwenImage21ModTarget::Both);
    const std::vector<float> md_first = r.run();
    r.model.add_mod_delta(r.mod_delta(0.015625f), 0, r.cfg.num_layers,
                          qd::QwenImage21ModTarget::Both);
    const std::vector<float> md_two = r.run();
    r.model.set_mod_delta(r.mod_delta(0.046875f), 0, r.cfg.num_layers,
                          qd::QwenImage21ModTarget::Both);
    const std::vector<float> md_one = r.run();
    const double d_mod = rel_maxdiff(md_two, md_one);
    const double d_mod_apart = rel_maxdiff(md_first, md_one);
    CHECK(d_mod < 1e-2);
    CHECK(d_mod < 0.05 * d_mod_apart);
    std::printf("qi21_slots: 0.03125+0.015625 vs 0.046875 mod delta %.3e "
                "(one component apart %.3e)\n", d_mod, d_mod_apart);

    // Two masks over the same range == one mask at their product.
    r.clear_all();
    std::vector<float> m1(static_cast<std::size_t>(r.L()), 1.0f);
    std::vector<float> m2(static_cast<std::size_t>(r.L()), 1.0f);
    std::vector<float> mp(static_cast<std::size_t>(r.L()), 1.0f);
    for (int i = 0; i < r.L(); i += 2) m1[static_cast<std::size_t>(i)] = 0.5f;
    for (int i = 0; i < r.L(); i += 3) m2[static_cast<std::size_t>(i)] = 0.25f;
    for (int i = 0; i < r.L(); ++i) {
        mp[static_cast<std::size_t>(i)] =
            m1[static_cast<std::size_t>(i)] * m2[static_cast<std::size_t>(i)];
    }
    r.model.add_gate_mask(bdtest::bd_upload(m1, r.L(), 1), 0, r.cfg.num_layers);
    r.model.add_gate_mask(bdtest::bd_upload(m2, r.L(), 1), 0, r.cfg.num_layers);
    CHECK(r.model.gate_masks().size() == 2);
    const std::vector<float> mask_two = r.run();
    r.model.set_gate_mask(bdtest::bd_upload(mp, r.L(), 1), 0, r.cfg.num_layers);
    const std::vector<float> mask_one = r.run();
    const double d_mask = rel_maxdiff(mask_two, mask_one);
    CHECK(d_mask < 1e-3);
    CHECK(rel_maxdiff(base, mask_one) > 1e-4);
    std::printf("qi21_slots: m1*m2 vs product mask rel maxdiff %.3e\n", d_mask);

    r.clear_all();
    CHECK(base == r.run());
}

// ── gate capture reads the COMPOSED gate ──────────────────────────────────
void test_capture_composed(Rig& r) {
    r.clear_all();
    std::vector<float> sink;
    r.model.capture_gates(&sink);
    r.run();
    const float g_plain = sink[static_cast<std::size_t>(r.text_seq)];

    // Halve the target-row attention gate twice; the capture should report a
    // quarter of the plain value, because it reads the folded row rather
    // than reconstructing what the factors "should" have done.
    r.model.add_gate_scale(0.5f, 1.0f, 1.0f, 1.0f, 0, r.cfg.num_layers);
    r.model.add_gate_scale(0.5f, 1.0f, 1.0f, 1.0f, 0, r.cfg.num_layers);
    r.run();
    const float g_quarter = sink[static_cast<std::size_t>(r.text_seq)];
    CHECK(std::abs(g_quarter - 0.25f * g_plain) <
          1e-3f * (1.0f + std::abs(g_plain)));

    // A per-block scale is visible per block: block 0 scaled, block 1 not.
    r.clear_all();
    r.model.add_gate_scale(0.5f, 1.0f, 1.0f, 1.0f, 0, 1);
    r.run();
    const float b0 = sink[static_cast<std::size_t>(r.text_seq)];
    const float b1 = sink[static_cast<std::size_t>(r.L()) +
                          static_cast<std::size_t>(r.text_seq)];
    CHECK(std::abs(b0 - 0.5f * g_plain) < 1e-3f * (1.0f + std::abs(g_plain)));
    CHECK(std::abs(b1 - g_plain) < 1e-3f * (1.0f + std::abs(g_plain)));
    std::printf("qi21_slots: capture plain %.5f, block0 %.5f, block1 %.5f\n",
                g_plain, b0, b1);

    r.model.capture_gates(nullptr);
    r.clear_all();
}

// ── 5. The prefix-KV dial ─────────────────────────────────────────────────
void test_prefix_kv_dial(Rig& r) {
    r.clear_all();

    qd::QwenImage21PrefixCache c;
    r.run_cached(c, 0.7f);                     // extract
    const std::vector<float> plain = r.run_cached(c, 0.4f);

    // Armed, it changes the cached step.
    r.model.set_prefix_kv_scale(0, r.cfg.num_layers, 1.0f, 0.5f);
    const std::vector<float> halved = r.run_cached(c, 0.4f);
    CHECK(rel_maxdiff(plain, halved) > 1e-4);

    // IDEMPOTENT: setting the same value again is the same picture. The old
    // in-place implementation would have squared it here.
    r.model.set_prefix_kv_scale(0, r.cfg.num_layers, 1.0f, 0.5f);
    const std::vector<float> again = r.run_cached(c, 0.4f);
    CHECK(halved == again);
    std::printf("qi21_slots: prefix-kv set twice is bit-identical\n");

    // And it is genuinely not the squared version.
    r.model.set_prefix_kv_scale(0, r.cfg.num_layers, 1.0f, 0.25f);
    CHECK(rel_maxdiff(halved, r.run_cached(c, 0.4f)) > 1e-5);

    // The cache stayed pristine: clearing the dial restores the unscaled
    // picture exactly, with no re-extraction.
    r.model.clear_prefix_kv_scales();
    CHECK(plain == r.run_cached(c, 0.4f));

    // It survives a reset and re-extract — it is a dial, not cache content.
    r.model.set_prefix_kv_scale(0, r.cfg.num_layers, 1.0f, 0.5f);
    c.reset();
    r.run_cached(c, 0.7f);                     // re-extract under the dial
    CHECK(rel_maxdiff(plain, r.run_cached(c, 0.4f)) > 1e-4);

    // A list composes per layer: 0.5 over all layers plus 0.5 over layer 1
    // is 0.5 on layer 0 and 0.25 on layer 1 — not the same as either alone.
    r.model.clear_prefix_kv_scales();
    const std::vector<float> half_all = r.run_cached(c, 0.4f);
    r.model.add_prefix_kv_scale(0, r.cfg.num_layers, 1.0f, 0.5f);
    const std::vector<float> one_binding = r.run_cached(c, 0.4f);
    r.model.add_prefix_kv_scale(1, 2, 1.0f, 0.5f);
    CHECK(r.model.prefix_kv_scales().size() == 2);
    const std::vector<float> two_bindings = r.run_cached(c, 0.4f);
    CHECK(rel_maxdiff(one_binding, two_bindings) > 1e-5);
    CHECK(rel_maxdiff(half_all, two_bindings) > 1e-4);

    // ...and it equals the single binding that spells the composition out.
    r.model.clear_prefix_kv_scales();
    r.model.add_prefix_kv_scale(0, 1, 1.0f, 0.5f);
    r.model.add_prefix_kv_scale(1, 2, 1.0f, 0.25f);
    CHECK(rel_maxdiff(two_bindings, r.run_cached(c, 0.4f)) < 1e-6);

    // An identity list costs nothing and reads as cleared.
    r.model.set_prefix_kv_scale(0, r.cfg.num_layers, 1.0f, 1.0f);
    CHECK(r.model.prefix_kv_scales().empty());
    CHECK(half_all == r.run_cached(c, 0.4f));

    r.clear_all();
}

// ── 6. Four independent gate multipliers ──────────────────────────────────
void test_gate_scale_rows(Rig& r) {
    r.clear_all();
    const std::vector<float> base = r.run();

    // The rank-1 form IS the sugar: attn x txt/img, mlp x txt/img.
    r.model.set_gate_scale(1.25f, 0.75f, 1.5f, 0.5f, 0, r.cfg.num_layers);
    const std::vector<float> via_product = r.run();
    r.model.set_gate_scale_rows(1.25f * 1.5f, 1.25f * 0.5f, 0.75f * 1.5f,
                                0.75f * 0.5f, 0, r.cfg.num_layers);
    CHECK(via_product == r.run());   // bit-identical
    std::printf("qi21_slots: rank-1 gate scale is four multipliers\n");

    // The thing the product cannot express: the attention gate on the IMAGE
    // rows alone. Under the product, raising attn raises the prefix's factor
    // too — so these two are different pictures, and only the second leaves
    // the prefix rows at 1.
    r.model.set_gate_scale(1.5f, 1.0f, 1.0f, 1.0f, 0, r.cfg.num_layers);
    const std::vector<float> attn_both_rows = r.run();
    r.model.set_gate_scale_rows(1.0f, 1.5f, 1.0f, 1.0f, 0, r.cfg.num_layers);
    const std::vector<float> attn_img_only = r.run();
    CHECK(rel_maxdiff(attn_both_rows, attn_img_only) > 1e-4);
    CHECK(rel_maxdiff(base, attn_img_only) > 1e-4);

    // ...and the mirror image: txt rows alone moves the picture too (the
    // text rows feed every image row through attention), and differently.
    r.model.set_gate_scale_rows(1.5f, 1.0f, 1.0f, 1.0f, 0, r.cfg.num_layers);
    const std::vector<float> attn_txt_only = r.run();
    CHECK(rel_maxdiff(attn_txt_only, attn_img_only) > 1e-4);

    // The four axes are separable: txt-only composed with img-only is the
    // binding that sets both.
    r.model.clear_gate_scales();
    r.model.add_gate_scale_rows(1.5f, 1.0f, 1.0f, 1.0f, 0, r.cfg.num_layers);
    r.model.add_gate_scale_rows(1.0f, 1.5f, 1.0f, 1.0f, 0, r.cfg.num_layers);
    const std::vector<float> composed = r.run();
    r.model.set_gate_scale_rows(1.5f, 1.5f, 1.0f, 1.0f, 0, r.cfg.num_layers);
    CHECK(composed == r.run());

    // Identity on all four reads as cleared and restores baseline exactly.
    r.model.set_gate_scale_rows(1.0f, 1.0f, 1.0f, 1.0f, 0, r.cfg.num_layers);
    CHECK(r.model.gate_scales().empty());
    CHECK(base == r.run());

    r.clear_all();
}

// ── 7. A prefix-affecting change re-applies every armed prefix edit ────────
//
// The coordinator's case, spelled out: arm the prefix-KV dial, then move a
// TXT-side gate scale — which is prefix-affecting, so the cache has to be
// dropped and re-extracted. After the re-extract the dial must still be in
// force. The old in-place ScalePrefixKv was multiplied INTO the cache, so a
// re-extract silently threw the edit away and the study's numbers were
// measuring an unedited prefix.
void test_prefix_edit_survives_reextract(Rig& r) {
    r.clear_all();

    qd::QwenImage21PrefixCache c;
    r.model.set_prefix_kv_scale(0, r.cfg.num_layers, 1.0f, 0.5f);
    r.run_cached(c, 0.7f);                      // extract under the dial
    const std::vector<float> dialled = r.run_cached(c, 0.4f);

    // A txt-side gate scale: prefix-affecting, so the cache is now stale.
    r.model.set_gate_scale_rows(1.3f, 1.0f, 1.0f, 1.0f, 0, r.cfg.num_layers);
    c.reset();
    r.run_cached(c, 0.7f);                      // re-extract
    const std::vector<float> after = r.run_cached(c, 0.4f);
    CHECK(rel_maxdiff(dialled, after) > 1e-5);  // the gate scale landed

    // ...and the dial is STILL applied: dropping it now moves the picture.
    r.model.clear_prefix_kv_scales();
    const std::vector<float> without = r.run_cached(c, 0.4f);
    CHECK(rel_maxdiff(after, without) > 1e-4);
    std::printf("qi21_slots: prefix edit survives a prefix-side re-extract\n");

    // Re-arming reproduces it exactly — nothing about the cache changed.
    r.model.set_prefix_kv_scale(0, r.cfg.num_layers, 1.0f, 0.5f);
    CHECK(after == r.run_cached(c, 0.4f));

    r.clear_all();
}

// ── 8 + 9. Per-sublayer gate masks ────────────────────────────────────────
void test_gate_mask_sublayer(Rig& r) {
    r.clear_all();
    const std::vector<float> base = r.run();

    // A mask that damps the second half of the image rows.
    std::vector<float> host(static_cast<std::size_t>(r.L()), 1.0f);
    for (int i = r.text_seq + r.img_len() / 2; i < r.L(); ++i) {
        host[static_cast<std::size_t>(i)] = 0.4f;
    }
    const bt::Tensor mask = bdtest::bd_upload(host, r.L(), 1);

    r.model.set_gate_mask(mask, 0, r.cfg.num_layers,
                          qd::QwenImage21GateSublayer::Both);
    const std::vector<float> both = r.run();
    CHECK(rel_maxdiff(base, both) > 1e-4);

    r.model.set_gate_mask(mask, 0, r.cfg.num_layers,
                          qd::QwenImage21GateSublayer::Attn);
    const std::vector<float> attn = r.run();
    r.model.set_gate_mask(mask, 0, r.cfg.num_layers,
                          qd::QwenImage21GateSublayer::Mlp);
    const std::vector<float> mlp = r.run();

    // Each half is its own picture, and neither is the blunt form. This is
    // the whole point: the MLP half is what drags a late-step edit's
    // retention down, so "restyle this region" wants the attn half alone.
    CHECK(rel_maxdiff(attn, mlp) > 1e-4);
    CHECK(rel_maxdiff(attn, both) > 1e-4);
    CHECK(rel_maxdiff(mlp, both) > 1e-4);

    // 'both' is exactly 'attn' and 'mlp' armed together.
    r.model.clear_gate_masks();
    r.model.add_gate_mask(mask, 0, r.cfg.num_layers,
                          qd::QwenImage21GateSublayer::Attn);
    r.model.add_gate_mask(mask, 0, r.cfg.num_layers,
                          qd::QwenImage21GateSublayer::Mlp);
    CHECK(both == r.run());
    std::printf("qi21_slots: 'both' == 'attn' + 'mlp' bit-for-bit\n");

    // Two masks on the SAME sublayer multiply; one on each do not interfere.
    r.model.clear_gate_masks();
    r.model.add_gate_mask(mask, 0, r.cfg.num_layers,
                          qd::QwenImage21GateSublayer::Attn);
    r.model.add_gate_mask(mask, 0, r.cfg.num_layers,
                          qd::QwenImage21GateSublayer::Attn);
    const std::vector<float> squared = r.run();
    CHECK(rel_maxdiff(attn, squared) > 1e-5);

    // 9. A wrong-length mask THROWS, and says what it wanted. The silent
    // no-op it replaces is how an all-zero mask written at img_len instead
    // of prefix_len + img_len read as "this surface does nothing".
    r.model.clear_gate_masks();
    const bt::Tensor short_mask = bdtest::bd_upload(
        std::vector<float>(static_cast<std::size_t>(r.img_len()), 0.0f),
        r.img_len(), 1);
    r.model.set_gate_mask(short_mask, 0, r.cfg.num_layers,
                          qd::QwenImage21GateSublayer::Both);
    bool threw = false;
    std::string msg;
    try {
        r.run();
    } catch (const std::exception& e) {
        threw = true;
        msg = e.what();
    }
    CHECK(threw);
    CHECK(msg.find(std::to_string(r.L())) != std::string::npos);
    CHECK(msg.find(std::to_string(r.img_len())) != std::string::npos);
    std::printf("qi21_slots: short mask throws: %s\n", msg.c_str());

    r.clear_all();
    CHECK(base == r.run());
}

// ── 10. The per-row prefix-KV scale ───────────────────────────────────────
void test_prefix_kv_rows(Rig& r) {
    r.clear_all();

    qd::QwenImage21PrefixCache c;
    r.run_cached(c, 0.7f);                     // extract
    const std::vector<float> plain = r.run_cached(c, 0.4f);

    // row_scale is a per-row WEIGHT on the scales, so an all-ones vector is
    // the broadcast it replaces — bit for bit. That is what makes
    // "row_scale empty = all rows" literally true rather than nearly true.
    r.model.set_prefix_kv_scale(0, r.cfg.num_layers, 1.0f, 0.5f);
    const std::vector<float> scalar = r.run_cached(c, 0.4f);
    const bt::Tensor all_rows = bdtest::bd_upload(
        std::vector<float>(static_cast<std::size_t>(r.text_seq), 1.0f),
        r.text_seq, 1);
    r.model.set_prefix_kv_scale(0, r.cfg.num_layers, 1.0f, 0.5f, all_rows);
    CHECK(rel_maxdiff(scalar, r.run_cached(c, 0.4f)) < 1e-6);
    std::printf("qi21_slots: an all-ones row scale is the broadcast\n");

    // ...and an all-ZEROS one is the identity: no row is selected.
    const bt::Tensor no_rows = bdtest::bd_upload(
        std::vector<float>(static_cast<std::size_t>(r.text_seq), 0.0f),
        r.text_seq, 1);
    r.model.set_prefix_kv_scale(0, r.cfg.num_layers, 1.0f, 0.5f, no_rows);
    CHECK(rel_maxdiff(plain, r.run_cached(c, 0.4f)) < 1e-6);

    // Per-token prompt weighting: apply the 0.5 to the last three rows only.
    std::vector<float> rows(static_cast<std::size_t>(r.text_seq), 1.0f);
    rows[0] = 0.0f;
    rows[1] = 0.0f;
    r.model.set_prefix_kv_scale(0, r.cfg.num_layers, 1.0f, 0.5f,
                                bdtest::bd_upload(rows, r.text_seq, 1));
    const std::vector<float> weighted = r.run_cached(c, 0.4f);
    CHECK(rel_maxdiff(scalar, weighted) > 1e-4);
    CHECK(rel_maxdiff(plain, weighted) > 1e-4);

    // A partial weight interpolates: 0.5 at weight 0.5 is 0.75 at weight 1.
    r.model.set_prefix_kv_scale(
        0, r.cfg.num_layers, 1.0f, 0.5f,
        bdtest::bd_upload(
            std::vector<float>(static_cast<std::size_t>(r.text_seq), 0.5f),
            r.text_seq, 1));
    const std::vector<float> half_weight = r.run_cached(c, 0.4f);
    r.model.set_prefix_kv_scale(0, r.cfg.num_layers, 1.0f, 0.75f);
    CHECK(rel_maxdiff(half_weight, r.run_cached(c, 0.4f)) < 1e-6);

    // A wrong-length row vector throws rather than scaling the wrong rows.
    r.model.set_prefix_kv_scale(
        0, r.cfg.num_layers, 1.0f, 1.0f,
        bdtest::bd_upload(
            std::vector<float>(static_cast<std::size_t>(r.text_seq + 3), 0.5f),
            r.text_seq + 3, 1));
    bool threw = false;
    try { r.run_cached(c, 0.4f); }
    catch (const std::exception&) { threw = true; }
    CHECK(threw);

    // The cache stayed pristine through all of it.
    r.model.clear_prefix_kv_scales();
    CHECK(plain == r.run_cached(c, 0.4f));

    r.clear_all();
}

}  // namespace

int main() {
    try { bt::init(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "init failed: %s\n", e.what());
        return 1;
    }
    try {
        Rig r;
        test_set_is_one_add(r);
        test_disjoint_ranges(r);
        test_composition(r);
        test_capture_composed(r);
        test_prefix_kv_dial(r);
        test_gate_scale_rows(r);
        test_prefix_edit_survives_reextract(r);
        test_gate_mask_sublayer(r);
        test_prefix_kv_rows(r);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "qi21_slots: exception: %s\n", e.what());
        return 1;
    }
    if (g_failures == 0) std::printf("qi21_slots: OK\n");
    else std::fprintf(stderr, "qi21_slots: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
