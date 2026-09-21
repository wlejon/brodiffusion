// Qwen-Image 2.1 image DiT (dit::QwenImage21Transformer2DModel) smoke +
// real-weights test.
//
// Part 1 (synthetic, always runs): a scaled-down but architecturally complete
// checkpoint — 2 blocks, 2 heads of 64, context 128, axes_dims_rope {8,28,28} —
// small enough for an in-memory fixture. Checks output shape / all-finite /
// determinism, that a CACHED second step reproduces a cache-free full prefill
// of the same step (the prefix KV cache's whole contract), and that the
// general multi-segment entry point agrees with the text-only one.
//
// Part 2 (synthetic): the Denoiser wrapper's NCHW <-> token transpose and its
// per-branch prefix caches.
//
// Part 3 (cooperative cancel): the sharded loader polls should_cancel once per
// block and throws LoadCancelled.
//
// Part 4 (gated on BRODIFFUSION_QI21_DIT_REAL=1 AND the weights): loads the
// real 2-shard 7.1B transformer with INT8 weight-only quantisation (so it fits
// a 24 GB card) and runs one forward. Numerical parity vs diffusers is checked
// separately by scripts/qwenimage21_dit_parity.sh, not by ctest.

#define _CRT_SECURE_NO_WARNINGS

#include "brodiffusion/dit/qwenimage21.h"
#include "brodiffusion/detail/compute.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include "test_compute.h"
#include "qwenimage21_fixture.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace bt = brotensor;
namespace st = brotensor::safetensors;
namespace qd = brodiffusion::dit;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

// The synthetic checkpoint, its RNG and the comparison helpers live in
// qwenimage21_fixture.h so the multi-slot test runs against the same model.
using qi21fix::download_any;
using qi21fix::rel_maxdiff;
using qi21fix::rnd;
using qi21fix::synth_cfg;
using qi21fix::write_fixture;

// ── shape / finite / determinism, and the prefix cache's contract ──────────
static void test_synthetic() {
    qd::QwenImage21Config cfg = synth_cfg();
    auto path = write_fixture(cfg, "brodiffusion_qi21_dit_test.safetensors");
    auto file = st::File::open(path.string());
    qd::QwenImage21Transformer2DModel model(cfg);
    model.load_weights(file, "");

    const int hp = 6, wp = 4, text_seq = 5;
    const int img_len = hp * wp;
    std::vector<float> lat_h =
        rnd(static_cast<std::size_t>(img_len) * cfg.in_channels, 1001);
    std::vector<float> emb_h =
        rnd(static_cast<std::size_t>(text_seq) * cfg.context_in_dim, 1002);

    bt::Tensor lat = bdtest::bd_upload(lat_h, img_len, cfg.in_channels);
    bt::Tensor emb = bdtest::bd_upload(emb_h, text_seq, cfg.context_in_dim);

    bt::Tensor txt;
    model.encode_text(emb, txt);
    bt::sync_all();
    CHECK(txt.rows == text_seq);
    CHECK(txt.cols == cfg.hidden_size());
    CHECK(txt.dtype == model.compute_dtype());

    // Step 0 with no cache at all.
    bt::Tensor out;
    model.forward(lat, hp, wp, txt, 0.7f, nullptr, out);
    bt::sync_all();
    CHECK(out.rows == img_len);
    CHECK(out.cols == cfg.out_channels);
    CHECK(out.dtype == model.compute_dtype());
    std::vector<float> v1 = download_any(out);
    int nonfinite = 0;
    for (float v : v1) if (!std::isfinite(v)) ++nonfinite;
    CHECK(nonfinite == 0);

    // Determinism.
    model.forward(lat, hp, wp, txt, 0.7f, nullptr, out);
    bt::sync_all();
    CHECK(v1 == download_any(out));

    // An "extract" step must equal the cache-free step exactly (same code
    // path plus a snapshot).
    qd::QwenImage21PrefixCache cache;
    model.forward(lat, hp, wp, txt, 0.7f, &cache, out);
    bt::sync_all();
    CHECK(cache.valid());
    CHECK(cache.prefix_len() == text_seq);
    CHECK(cache.target_hp() == hp && cache.target_wp() == wp);
    CHECK(v1 == download_any(out));

    // A "cached" step must reproduce the cache-free full prefill of the same
    // timestep — the whole point of the cache.
    model.forward(lat, hp, wp, txt, 0.4f, &cache, out);
    bt::sync_all();
    std::vector<float> cached = download_any(out);
    bt::Tensor out_ref;
    model.forward(lat, hp, wp, txt, 0.4f, nullptr, out_ref);
    bt::sync_all();
    std::vector<float> full = download_any(out_ref);
    const double d = rel_maxdiff(full, cached);
    CHECK(d < 2e-3);
    std::printf("qi21_dit: cached vs full-prefill rel maxdiff %.3e\n", d);

    // A grid change on a live cache is a caller error, not silent garbage.
    bool threw = false;
    try {
        model.forward(lat, wp, hp, txt, 0.4f, &cache, out);
    } catch (const std::exception&) { threw = true; }
    CHECK(threw);
    cache.reset();
    CHECK(!cache.valid());

    // forward() is forward_joint() with a single Text segment — check the
    // general entry point agrees bit-for-bit.
    std::vector<qd::QwenImage21Segment> prefix(1);
    prefix[0].kind = qd::QwenImage21Segment::Kind::Text;
    prefix[0].n_tokens = text_seq;
    model.forward_joint(lat, hp, wp, txt, nullptr, prefix, 0.7f, nullptr, out);
    bt::sync_all();
    CHECK(v1 == download_any(out));

    // Multi-segment prefix (text / condition image / text) — the layout edit
    // mode uses. Only structural checks here; numerical parity for the T2I
    // layout is scripts/qwenimage21_dit_parity.sh.
    {
        const int ch = 2, cw = 2;
        std::vector<qd::QwenImage21Segment> segs(3);
        segs[0].kind = qd::QwenImage21Segment::Kind::Text;
        segs[0].n_tokens = 3;
        segs[1].kind = qd::QwenImage21Segment::Kind::Image;
        segs[1].n_tokens = ch * cw;
        segs[1].h = ch;
        segs[1].w = cw;
        segs[2].kind = qd::QwenImage21Segment::Kind::Text;
        segs[2].n_tokens = text_seq - 3;
        std::vector<float> cond_h =
            rnd(static_cast<std::size_t>(ch) * cw * cfg.in_channels, 1003);
        bt::Tensor cond = bdtest::bd_upload(cond_h, ch * cw, cfg.in_channels);

        bt::Tensor joint_out;
        model.forward_joint(lat, hp, wp, txt, &cond, segs, 0.7f, nullptr,
                            joint_out);
        bt::sync_all();
        CHECK(joint_out.rows == img_len);
        CHECK(joint_out.cols == cfg.out_channels);
        std::vector<float> jv = download_any(joint_out);
        int nf = 0;
        for (float v : jv) if (!std::isfinite(v)) ++nf;
        CHECK(nf == 0);
        // The condition image genuinely participates.
        CHECK(rel_maxdiff(v1, jv) > 1e-4);

        // ... and the cache reproduces it for the multi-segment layout too.
        qd::QwenImage21PrefixCache c2;
        model.forward_joint(lat, hp, wp, txt, &cond, segs, 0.7f, &c2, joint_out);
        bt::sync_all();
        CHECK(c2.prefix_len() == text_seq + ch * cw);
        CHECK(jv == download_any(joint_out));
        model.forward_joint(lat, hp, wp, txt, &cond, segs, 0.4f, &c2, joint_out);
        bt::sync_all();
        std::vector<float> jc = download_any(joint_out);
        model.forward_joint(lat, hp, wp, txt, &cond, segs, 0.4f, nullptr,
                            joint_out);
        bt::sync_all();
        const double d2 = rel_maxdiff(download_any(joint_out), jc);
        CHECK(d2 < 2e-3);
        std::printf("qi21_dit: multi-segment cached rel maxdiff %.3e\n", d2);
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

// ── Denoiser wrapper: NCHW <-> token transpose, per-branch caches ──────────
static void test_denoiser() {
    qd::QwenImage21Config cfg = synth_cfg();
    auto path = write_fixture(cfg, "brodiffusion_qi21_den_test.safetensors");
    auto file = st::File::open(path.string());
    qd::QwenImage21Denoiser den(cfg);
    den.load_weights(file, "");
    CHECK(den.latent_channels() == cfg.in_channels);
    CHECK(den.prediction_type() == brodiffusion::PredictionType::Velocity);
    CHECK(den.uses_cfg());

    const int H_lat = 6, W_lat = 4, text_seq = 5;
    const std::size_t n_lat =
        static_cast<std::size_t>(cfg.in_channels) * H_lat * W_lat;

    brodiffusion::Conditioning cond;
    cond.text_embeddings = bdtest::bd_upload(
        rnd(static_cast<std::size_t>(text_seq) * cfg.context_in_dim, 2001),
        text_seq, cfg.context_in_dim);
    cond.uncond_embeddings = bdtest::bd_upload(
        rnd(static_cast<std::size_t>(text_seq) * cfg.context_in_dim, 2002),
        text_seq, cfg.context_in_dim);
    cond.has_uncond = true;
    brodiffusion::PreparedConditioning prep = den.prepare(cond);
    CHECK(static_cast<bool>(prep));

    bt::Tensor latent = bdtest::bd_upload(rnd(n_lat, 2003), 1,
                                          static_cast<int>(n_lat));
    bt::Tensor out;
    den.forward(latent, H_lat, W_lat, 700.0f, prep,
                brodiffusion::Branch::Cond, out);
    bt::sync_all();
    CHECK(out.rows == 1);
    CHECK(out.cols == static_cast<int>(n_lat));
    std::vector<float> v0 = bdtest::bd_download(out);
    int nonfinite = 0;
    for (float v : v0) if (!std::isfinite(v)) ++nonfinite;
    CHECK(nonfinite == 0);

    // Second step decodes from the cache the first one extracted; it must
    // match a forward off a freshly reset cache (a full prefill).
    den.forward(latent, H_lat, W_lat, 400.0f, prep,
                brodiffusion::Branch::Cond, out);
    bt::sync_all();
    std::vector<float> cached = bdtest::bd_download(out);
    den.reset_cache(prep);
    den.forward(latent, H_lat, W_lat, 400.0f, prep,
                brodiffusion::Branch::Cond, out);
    bt::sync_all();
    CHECK(rel_maxdiff(bdtest::bd_download(out), cached) < 2e-3);

    // The uncond branch has its own prompt and its own cache.
    bt::Tensor uout;
    den.forward(latent, H_lat, W_lat, 700.0f, prep,
                brodiffusion::Branch::Uncond, uout);
    bt::sync_all();
    CHECK(rel_maxdiff(v0, bdtest::bd_download(uout)) > 1e-4);

    // A grid change resets the cache rather than failing.
    den.forward(latent, W_lat, H_lat, 400.0f, prep,
                brodiffusion::Branch::Cond, out);
    bt::sync_all();
    CHECK(out.cols == static_cast<int>(n_lat));

    // ── image-conditioned prefix ──────────────────────────────────────────
    //
    // prepare_edit() routes forward() through the same forward_joint() an
    // edit-mode generation uses. Two things have to hold: the Denoiser's
    // answer is what a direct forward_joint() on the same inputs gives (so
    // the wrapper adds a transpose and nothing else), and the prefix cache
    // still reproduces a full prefill when the prefix contains image rows.
    {
        const int ch = 2, cw = 2;
        std::vector<qd::QwenImage21Segment> segs(3);
        segs[0].kind = qd::QwenImage21Segment::Kind::Text;
        segs[0].n_tokens = 2;
        segs[1].kind = qd::QwenImage21Segment::Kind::Image;
        segs[1].n_tokens = ch * cw;
        segs[1].h = ch;
        segs[1].w = cw;
        segs[2].kind = qd::QwenImage21Segment::Kind::Text;
        segs[2].n_tokens = text_seq - 2;

        qd::QwenImage21EditPrefix edit;
        edit.segments = segs;
        edit.cond_latents = bdtest::bd_upload(
            rnd(static_cast<std::size_t>(ch) * cw * cfg.in_channels, 2010),
            ch * cw, cfg.in_channels);

        brodiffusion::PreparedConditioning eprep =
            den.prepare_edit(cond, edit, nullptr);
        bt::Tensor eout;
        den.forward(latent, H_lat, W_lat, 700.0f, eprep,
                    brodiffusion::Branch::Cond, eout);
        bt::sync_all();
        CHECK(eout.rows == 1);
        CHECK(eout.cols == static_cast<int>(n_lat));
        std::vector<float> ev = bdtest::bd_download(eout);
        int nf = 0;
        for (float v : ev) if (!std::isfinite(v)) ++nf;
        CHECK(nf == 0);
        // The condition image is actually read: the same prompt without it
        // gives a different velocity.
        CHECK(rel_maxdiff(v0, ev) > 1e-4);

        // Bit-exact against forward_joint() driven by hand. The Denoiser owns
        // the (C,HW) <-> (HW,C) transpose and the timestep's /1000, so this
        // reproduces both and compares the raw token tensor.
        {
            bt::Tensor packed;
            bt::nchw_to_sequence(latent, 1, cfg.in_channels, H_lat, W_lat,
                                 packed);
            bt::Tensor& txt_rows = den.text_rows(eprep, /*uncond=*/false);
            bt::Tensor ref_tok;
            den.model().forward_joint(packed, H_lat, W_lat, txt_rows,
                                      &edit.cond_latents, segs, 0.7f, nullptr,
                                      ref_tok);
            bt::sync_all();
            bt::Tensor ref_nchw;
            bt::sequence_to_nchw(ref_tok, 1, cfg.in_channels, H_lat, W_lat,
                                 ref_nchw);
            bt::sync_all();
            // eout came off a cache-extracting call, which is a full prefill
            // like this one, so the two are the same arithmetic in the same
            // order.
            CHECK(download_any(ref_nchw) == ev);
        }

        // A cached step over an image-bearing prefix still equals a prefill.
        den.forward(latent, H_lat, W_lat, 400.0f, eprep,
                    brodiffusion::Branch::Cond, eout);
        bt::sync_all();
        std::vector<float> ecached = bdtest::bd_download(eout);
        den.reset_cache(eprep);
        den.forward(latent, H_lat, W_lat, 400.0f, eprep,
                    brodiffusion::Branch::Cond, eout);
        bt::sync_all();
        const double de = rel_maxdiff(bdtest::bd_download(eout), ecached);
        CHECK(de < 2e-3);
        std::printf("qi21_dit: edit-prefix cached rel maxdiff %.3e\n", de);

        // A prefix whose text segments do not cover the encoded rows is
        // rejected at prepare time, where the caller can still see which
        // count disagreed.
        bool threw = false;
        try {
            qd::QwenImage21EditPrefix bad = edit;
            bad.segments[2].n_tokens += 1;
            (void)den.prepare_edit(cond, bad, nullptr);
        } catch (const std::exception&) {
            threw = true;
        }
        CHECK(threw);
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

// ── research hooks ─────────────────────────────────────────────────────────
//
// The contract every hook shares: at its identity setting it is BIT-EXACT
// with the unhooked forward (research must be able to leave the plumbing in
// place and dial it to zero), and off-identity it changes exactly what its
// doc comment says it changes.
static void test_hooks() {
    qd::QwenImage21Config cfg = synth_cfg();
    auto path = write_fixture(cfg, "brodiffusion_qi21_hooks_test.safetensors");
    auto file = st::File::open(path.string());
    qd::QwenImage21Transformer2DModel model(cfg);
    model.load_weights(file, "");

    const int hp = 6, wp = 4, text_seq = 5;
    const int img_len = hp * wp;
    const int H = cfg.hidden_size();
    const int L = text_seq + img_len;

    bt::Tensor lat = bdtest::bd_upload(
        rnd(static_cast<std::size_t>(img_len) * cfg.in_channels, 3001),
        img_len, cfg.in_channels);
    bt::Tensor emb = bdtest::bd_upload(
        rnd(static_cast<std::size_t>(text_seq) * cfg.context_in_dim, 3002),
        text_seq, cfg.context_in_dim);
    bt::Tensor txt;
    model.encode_text(emb, txt);

    bt::Tensor out;
    model.forward(lat, hp, wp, txt, 0.7f, nullptr, out);
    bt::sync_all();
    const std::vector<float> base = download_any(out);

    // ── identity settings are no-ops, bit for bit ─────────────────────────
    model.set_mod_delta(bt::Tensor(), 0, 0);
    model.set_gate_scale(1.0f, 1.0f, 1.0f, 1.0f, 0, cfg.num_layers);
    model.set_gate_mask(bt::Tensor(), 0, 0);
    model.set_norm_out_scale_delta(bt::Tensor());
    model.forward(lat, hp, wp, txt, 0.7f, nullptr, out);
    bt::sync_all();
    CHECK(base == download_any(out));

    // A zero delta over every block is also exactly a no-op.
    {
        std::vector<float> z(static_cast<std::size_t>(4) * H, 0.0f);
        bt::Tensor zd = bdtest::bd_upload(z, 1, 4 * H);
        model.set_mod_delta(zd, 0, cfg.num_layers, qd::QwenImage21ModTarget::Both);
        model.forward(lat, hp, wp, txt, 0.7f, nullptr, out);
        bt::sync_all();
        CHECK(base == download_any(out));
        model.set_mod_delta(bt::Tensor(), 0, 0);
    }
    // An all-ones mask over every block, likewise.
    {
        std::vector<float> ones(static_cast<std::size_t>(L), 1.0f);
        bt::Tensor mt = bdtest::bd_upload(ones, L, 1);
        model.set_gate_mask(mt, 0, cfg.num_layers);
        model.forward(lat, hp, wp, txt, 0.7f, nullptr, out);
        bt::sync_all();
        CHECK(base == download_any(out));
        model.set_gate_mask(bt::Tensor(), 0, 0);
    }

    // ── time-mod readout ──────────────────────────────────────────────────
    {
        bt::Tensor temb, mods;
        model.compute_time_mod(0.7f, temb, mods);
        CHECK(temb.rows == 2 && temb.cols == H);
        CHECK(mods.rows == 2 && mods.cols == 4 * H);
        CHECK(temb.dtype == bt::Dtype::FP32 && mods.dtype == bt::Dtype::FP32);
        std::vector<float> mh = bdtest::bd_download(mods);
        int nf = 0;
        for (float v : mh) if (!std::isfinite(v)) ++nf;
        CHECK(nf == 0);
        // The t = 0 row differs from the sampled-t row (that is the whole
        // point of causal_condition).
        double diff = 0.0;
        for (int i = 0; i < 4 * H; ++i) {
            diff = std::max(diff, std::abs(static_cast<double>(mh[i]) -
                                           mh[static_cast<std::size_t>(4 * H + i)]));
        }
        CHECK(diff > 1e-6);
        // t = 0 must be the same row whatever t was sampled.
        bt::Tensor temb2, mods2;
        model.compute_time_mod(0.2f, temb2, mods2);
        std::vector<float> mh2 = bdtest::bd_download(mods2);
        for (int i = 0; i < 4 * H; ++i) {
            CHECK(mh[static_cast<std::size_t>(4 * H + i)] ==
                  mh2[static_cast<std::size_t>(4 * H + i)]);
        }
    }

    // ── a Target delta leaves the extract step's prefix K/V untouched ─────
    //
    // The prefix rows are modulated from t = 0, so steering the sampled-t row
    // cannot move them — which is what makes the prefix KV cache survive a
    // Target-side experiment. Compare the caches an extract step produces.
    {
        std::vector<float> d(static_cast<std::size_t>(4) * H);
        for (std::size_t i = 0; i < d.size(); ++i) {
            d[i] = 0.05f * static_cast<float>(static_cast<int>(i % 7) - 3);
        }
        bt::Tensor dt_ = bdtest::bd_upload(d, 1, 4 * H);

        // Extract a reference cache with no hook armed.
        qd::QwenImage21PrefixCache c_ref;
        model.forward(lat, hp, wp, txt, 0.7f, &c_ref, out);
        bt::sync_all();

        // Extract a second one with a Target-only delta armed.
        model.set_mod_delta(dt_, 0, cfg.num_layers,
                            qd::QwenImage21ModTarget::Target);
        qd::QwenImage21PrefixCache c_tgt;
        bt::Tensor out_t;
        model.forward(lat, hp, wp, txt, 0.7f, &c_tgt, out_t);
        bt::sync_all();
        // The velocity moved...
        CHECK(rel_maxdiff(base, download_any(out_t)) > 1e-4);
        model.set_mod_delta(bt::Tensor(), 0, 0);

        // ...but the two caches hold the SAME prefix K/V: with the hook
        // cleared again, a cached step off either one is bit-identical. Only
        // the t = 0 row feeds the prefix, and a Target delta does not touch
        // it — which is exactly why a Target-side experiment survives the
        // cache.
        bt::Tensor out_a, out_b;
        model.forward(lat, hp, wp, txt, 0.4f, &c_ref, out_a);
        bt::sync_all();
        const std::vector<float> from_ref = download_any(out_a);
        model.forward(lat, hp, wp, txt, 0.4f, &c_tgt, out_b);
        bt::sync_all();
        CHECK(from_ref == download_any(out_b));

        // A Prefix delta, by contrast, changes the cache it extracts — the
        // same cached step off it lands somewhere else.
        model.set_mod_delta(dt_, 0, cfg.num_layers,
                            qd::QwenImage21ModTarget::Prefix);
        qd::QwenImage21PrefixCache c_pre;
        bt::Tensor tmp_pre;
        model.forward(lat, hp, wp, txt, 0.7f, &c_pre, tmp_pre);
        bt::sync_all();
        model.set_mod_delta(bt::Tensor(), 0, 0);
        bt::Tensor out_pc;
        model.forward(lat, hp, wp, txt, 0.4f, &c_pre, out_pc);
        bt::sync_all();
        CHECK(rel_maxdiff(from_ref, download_any(out_pc)) > 1e-4);

        // A Prefix delta also moves the extract step's own output.
        model.set_mod_delta(dt_, 0, cfg.num_layers,
                            qd::QwenImage21ModTarget::Prefix);
        model.set_mod_delta(dt_, 0, cfg.num_layers,
                            qd::QwenImage21ModTarget::Prefix);
        bt::Tensor out_p;
        model.forward(lat, hp, wp, txt, 0.7f, nullptr, out_p);
        bt::sync_all();
        std::vector<float> vp = download_any(out_p);
        CHECK(rel_maxdiff(base, vp) > 1e-4);
        CHECK(rel_maxdiff(download_any(out_t), vp) > 1e-6);
        model.set_mod_delta(bt::Tensor(), 0, 0);

        // A block-range delta is not a whole-model delta.
        model.set_mod_delta(dt_, cfg.num_layers - 1, cfg.num_layers,
                            qd::QwenImage21ModTarget::Target);
        bt::Tensor out_r;
        model.forward(lat, hp, wp, txt, 0.7f, nullptr, out_r);
        bt::sync_all();
        std::vector<float> vr = download_any(out_r);
        CHECK(rel_maxdiff(base, vr) > 1e-6);
        CHECK(rel_maxdiff(download_any(out_t), vr) > 1e-6);
        model.set_mod_delta(bt::Tensor(), 0, 0);
    }

    // ── gate scale ────────────────────────────────────────────────────────
    {
        model.set_gate_scale(0.5f, 1.0f, 1.0f, 1.0f, 0, cfg.num_layers);
        model.forward(lat, hp, wp, txt, 0.7f, nullptr, out);
        bt::sync_all();
        CHECK(rel_maxdiff(base, download_any(out)) > 1e-4);
        // Scaling only the prefix side is a different move from only the
        // target side.
        model.set_gate_scale(1.0f, 1.0f, 0.5f, 1.0f, 0, cfg.num_layers);
        model.forward(lat, hp, wp, txt, 0.7f, nullptr, out);
        bt::sync_all();
        std::vector<float> v_txt = download_any(out);
        model.set_gate_scale(1.0f, 1.0f, 1.0f, 0.5f, 0, cfg.num_layers);
        model.forward(lat, hp, wp, txt, 0.7f, nullptr, out);
        bt::sync_all();
        CHECK(rel_maxdiff(v_txt, download_any(out)) > 1e-5);
        model.set_gate_scale(1.0f, 1.0f, 1.0f, 1.0f, 0, 0);
    }

    // ── gate mask: zeroing a block range zeroes its contribution ──────────
    //
    // With every row of the mask at 0 over blocks [lo, hi), those blocks add
    // nothing to the residual — so the whole model collapses to the one built
    // from the remaining blocks. Check that against a zero gate scale, which
    // is the same statement through the other hook.
    {
        std::vector<float> zeros(static_cast<std::size_t>(L), 0.0f);
        bt::Tensor mz = bdtest::bd_upload(zeros, L, 1);
        model.set_gate_mask(mz, 1, cfg.num_layers);
        model.forward(lat, hp, wp, txt, 0.7f, nullptr, out);
        bt::sync_all();
        std::vector<float> v_masked = download_any(out);
        model.set_gate_mask(bt::Tensor(), 0, 0);

        model.set_gate_scale(0.0f, 0.0f, 1.0f, 1.0f, 1, cfg.num_layers);
        model.forward(lat, hp, wp, txt, 0.7f, nullptr, out);
        bt::sync_all();
        CHECK(rel_maxdiff(v_masked, download_any(out)) < 1e-6);
        model.set_gate_scale(1.0f, 1.0f, 1.0f, 1.0f, 0, 0);
        CHECK(rel_maxdiff(base, v_masked) > 1e-4);
    }

    // ── gate delta (post-tanh) ────────────────────────────────────────────
    //
    // The hook adds to the EFFECTIVE gate, after the tanh and after any
    // set_gate_scale factor, which makes the two composable exactly:
    // zeroing the target rows' gate and then adding back tanh(gate) as a
    // delta has to land on the unhooked forward.
    {
        // An empty delta, and a zero one over every block, are no-ops.
        model.set_gate_delta(bt::Tensor(), 0, 0);
        model.forward(lat, hp, wp, txt, 0.7f, nullptr, out);
        bt::sync_all();
        CHECK(base == download_any(out));
        {
            std::vector<float> z(static_cast<std::size_t>(2) * H, 0.0f);
            bt::Tensor zd = bdtest::bd_upload(z, 1, 2 * H);
            model.set_gate_delta(zd, 0, cfg.num_layers,
                                 qd::QwenImage21ModTarget::Both);
            model.forward(lat, hp, wp, txt, 0.7f, nullptr, out);
            bt::sync_all();
            CHECK(base == download_any(out));
            model.set_gate_delta(bt::Tensor(), 0, 0);
        }

        // Zero gate + delta == the pure gated residual it replaces.
        // img_scale = 0 zeroes the TARGET rows' gates only, leaving the
        // prefix (t = 0) side alone; the delta then restores exactly
        // tanh(gate1) / tanh(gate2) for those rows, read out of
        // compute_time_mod's raw pre-tanh modulation.
        bt::Tensor temb, mods;
        model.compute_time_mod(0.7f, temb, mods);
        std::vector<float> mh = bdtest::bd_download(mods);
        std::vector<float> eff(static_cast<std::size_t>(2) * H);
        for (int j = 0; j < H; ++j) {
            eff[static_cast<std::size_t>(j)] =
                std::tanh(mh[static_cast<std::size_t>(H + j)]);          // gate1
            eff[static_cast<std::size_t>(H + j)] =
                std::tanh(mh[static_cast<std::size_t>(3 * H + j)]);      // gate2
        }
        bt::Tensor effd = bdtest::bd_upload(eff, 1, 2 * H);

        // Without the delta, a zeroed target gate is a very different image.
        model.set_gate_scale(1.0f, 1.0f, 1.0f, 0.0f, 0, cfg.num_layers);
        model.forward(lat, hp, wp, txt, 0.7f, nullptr, out);
        bt::sync_all();
        const std::vector<float> v_zero = download_any(out);
        const double d_zero = rel_maxdiff(base, v_zero);
        CHECK(d_zero > 1e-3);

        model.set_gate_delta(effd, 0, cfg.num_layers,
                             qd::QwenImage21ModTarget::Target);
        model.forward(lat, hp, wp, txt, 0.7f, nullptr, out);
        bt::sync_all();
        const double d_restored = rel_maxdiff(base, download_any(out));
        // The restored gate takes a detour the base one does not: device
        // BF16 -> host FP32 tanh -> device BF16 delta -> added to zero. One
        // BF16 ulp is ~0.4% and two blocks compound it, so the bar is the
        // ratio — the restore has to put the output back essentially where
        // the base forward had it, not reproduce it bit for bit.
        CHECK(d_restored < 1e-2);
        CHECK(d_restored < d_zero * 0.05);
        std::printf("qi21_dit: gate delta restore rel maxdiff %.3e "
                    "(zero-gate %.3e)\n", d_restored, d_zero);
        model.set_gate_scale(1.0f, 1.0f, 1.0f, 1.0f, 0, 0);
        model.set_gate_delta(bt::Tensor(), 0, 0);

        // A nonzero delta on its own moves the output, and a block-range
        // delta is not a whole-model one.
        std::vector<float> d(static_cast<std::size_t>(2) * H, 0.2f);
        bt::Tensor dd = bdtest::bd_upload(d, 1, 2 * H);
        model.set_gate_delta(dd, 0, cfg.num_layers,
                             qd::QwenImage21ModTarget::Target);
        model.forward(lat, hp, wp, txt, 0.7f, nullptr, out);
        bt::sync_all();
        const std::vector<float> v_all = download_any(out);
        CHECK(rel_maxdiff(base, v_all) > 1e-4);
        model.set_gate_delta(dd, cfg.num_layers - 1, cfg.num_layers,
                             qd::QwenImage21ModTarget::Target);
        model.forward(lat, hp, wp, txt, 0.7f, nullptr, out);
        bt::sync_all();
        const std::vector<float> v_last = download_any(out);
        CHECK(rel_maxdiff(base, v_last) > 1e-6);
        CHECK(rel_maxdiff(v_all, v_last) > 1e-6);

        // Target and Prefix address different rows, so they are different
        // moves — and a Prefix-side delta changes the prefix K/V an extract
        // step writes, exactly as a Prefix mod delta does.
        model.set_gate_delta(dd, 0, cfg.num_layers,
                             qd::QwenImage21ModTarget::Prefix);
        model.forward(lat, hp, wp, txt, 0.7f, nullptr, out);
        bt::sync_all();
        CHECK(rel_maxdiff(v_all, download_any(out)) > 1e-6);
        qd::QwenImage21PrefixCache c_pd;
        bt::Tensor tmp_pd;
        model.forward(lat, hp, wp, txt, 0.7f, &c_pd, tmp_pd);
        bt::sync_all();
        model.set_gate_delta(bt::Tensor(), 0, 0);
        qd::QwenImage21PrefixCache c_pref;
        model.forward(lat, hp, wp, txt, 0.7f, &c_pref, tmp_pd);
        bt::sync_all();
        bt::Tensor o_a, o_b;
        model.forward(lat, hp, wp, txt, 0.4f, &c_pd, o_a);
        model.forward(lat, hp, wp, txt, 0.4f, &c_pref, o_b);
        bt::sync_all();
        CHECK(rel_maxdiff(download_any(o_a), download_any(o_b)) > 1e-4);

        // ...and it composes with set_gate_scale rather than replacing it:
        // scale 0.5 plus delta 0.2 is neither one alone.
        model.set_gate_delta(dd, 0, cfg.num_layers,
                             qd::QwenImage21ModTarget::Target);
        model.set_gate_scale(0.5f, 0.5f, 1.0f, 1.0f, 0, cfg.num_layers);
        model.forward(lat, hp, wp, txt, 0.7f, nullptr, out);
        bt::sync_all();
        const std::vector<float> v_both = download_any(out);
        CHECK(rel_maxdiff(v_all, v_both) > 1e-6);
        model.set_gate_delta(bt::Tensor(), 0, 0);
        model.forward(lat, hp, wp, txt, 0.7f, nullptr, out);
        bt::sync_all();
        CHECK(rel_maxdiff(v_both, download_any(out)) > 1e-6);
        model.set_gate_scale(1.0f, 1.0f, 1.0f, 1.0f, 0, 0);
        model.forward(lat, hp, wp, txt, 0.7f, nullptr, out);
        bt::sync_all();
        CHECK(base == download_any(out));
    }

    // ── gate capture shapes ───────────────────────────────────────────────
    {
        std::vector<float> sink;
        model.capture_gates(&sink);
        model.forward(lat, hp, wp, txt, 0.7f, nullptr, out);
        bt::sync_all();
        CHECK(sink.size() ==
              static_cast<std::size_t>(cfg.num_layers) * static_cast<std::size_t>(L));
        int nf = 0;
        for (float v : sink) if (!std::isfinite(v)) ++nf;
        CHECK(nf == 0);
        // The prefix columns carry the t = 0 gate, the target columns the
        // sampled-t gate — one shared modulation, so the two are constant
        // within their own span.
        CHECK(sink[0] == sink[static_cast<std::size_t>(text_seq - 1)]);
        CHECK(sink[static_cast<std::size_t>(text_seq)] ==
              sink[static_cast<std::size_t>(L - 1)]);

        // A half-range mask shows up in the capture where it was aimed.
        std::vector<float> m(static_cast<std::size_t>(L), 1.0f);
        m[static_cast<std::size_t>(text_seq)] = 0.0f;
        bt::Tensor mt = bdtest::bd_upload(m, L, 1);
        model.set_gate_mask(mt, 0, 1);
        model.forward(lat, hp, wp, txt, 0.7f, nullptr, out);
        bt::sync_all();
        CHECK(sink[static_cast<std::size_t>(text_seq)] == 0.0f);
        CHECK(sink[static_cast<std::size_t>(L) +
                   static_cast<std::size_t>(text_seq)] != 0.0f);
        model.set_gate_mask(bt::Tensor(), 0, 0);
        model.capture_gates(nullptr);
    }

    // ── norm_out scale delta ──────────────────────────────────────────────
    {
        std::vector<float> d(static_cast<std::size_t>(H), 0.25f);
        bt::Tensor dt_ = bdtest::bd_upload(d, 1, H);
        model.set_norm_out_scale_delta(dt_);
        model.forward(lat, hp, wp, txt, 0.7f, nullptr, out);
        bt::sync_all();
        CHECK(rel_maxdiff(base, download_any(out)) > 1e-4);
        model.set_norm_out_scale_delta(bt::Tensor());
        model.forward(lat, hp, wp, txt, 0.7f, nullptr, out);
        bt::sync_all();
        CHECK(base == download_any(out));
    }

    // ── prefix KV cache surface ───────────────────────────────────────────
    {
        qd::QwenImage21PrefixCache c1;
        model.forward(lat, hp, wp, txt, 0.7f, &c1, out);
        bt::sync_all();
        CHECK(c1.num_layers() == cfg.num_layers);

        bt::Tensor cached;
        model.forward(lat, hp, wp, txt, 0.4f, &c1, cached);
        bt::sync_all();
        std::vector<float> v_plain = download_any(cached);

        // Scaling V attenuates the prefix's contribution.
        c1.scale_kv(0, cfg.num_layers, 1.0f, 0.5f);
        model.forward(lat, hp, wp, txt, 0.4f, &c1, cached);
        bt::sync_all();
        CHECK(rel_maxdiff(v_plain, download_any(cached)) > 1e-4);
        // ... and 1/1 is exactly a no-op.
        c1.scale_kv(0, cfg.num_layers, 1.0f, 2.0f);
        model.forward(lat, hp, wp, txt, 0.4f, &c1, cached);
        bt::sync_all();
        CHECK(rel_maxdiff(v_plain, download_any(cached)) < 2e-3);

        // Blending towards a second prompt's cache.
        bt::Tensor emb2 = bdtest::bd_upload(
            rnd(static_cast<std::size_t>(text_seq) * cfg.context_in_dim, 3003),
            text_seq, cfg.context_in_dim);
        bt::Tensor txt2;
        model.encode_text(emb2, txt2);
        qd::QwenImage21PrefixCache c2;
        bt::Tensor tmp;
        model.forward(lat, hp, wp, txt2, 0.7f, &c2, tmp);
        bt::sync_all();

        qd::QwenImage21PrefixCache c3;
        model.forward(lat, hp, wp, txt, 0.7f, &c3, tmp);
        bt::sync_all();
        c3.blend_from(c2, 0.0f);      // no-op
        model.forward(lat, hp, wp, txt, 0.4f, &c3, cached);
        bt::sync_all();
        CHECK(rel_maxdiff(v_plain, download_any(cached)) < 2e-3);
        c3.blend_from(c2, 0.5f);
        model.forward(lat, hp, wp, txt, 0.4f, &c3, cached);
        bt::sync_all();
        CHECK(rel_maxdiff(v_plain, download_any(cached)) > 1e-4);

        // A layout mismatch is an error, not silent garbage.
        qd::QwenImage21PrefixCache c4;
        bool threw = false;
        try { c4.blend_from(c2, 0.5f); }
        catch (const std::exception&) { threw = true; }
        CHECK(threw);
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

// ── cooperative cancel ─────────────────────────────────────────────────────
static void test_load_cancel() {
    qd::QwenImage21Config cfg = synth_cfg();   // num_layers == 2
    auto path = write_fixture(cfg, "brodiffusion_qi21_cancel_test.safetensors");
    auto file = st::File::open(path.string());
    std::vector<const st::File*> shards{ &file };

    {
        int polls = 0;
        qd::QwenImage21Transformer2DModel model(cfg);
        bool threw = false;
        try {
            model.load_weights(shards, "", [&]() { ++polls; return true; });
        } catch (const brodiffusion::LoadCancelled&) { threw = true; }
        CHECK(threw);
        CHECK(polls == 1);
    }
    {
        int polls = 0;
        qd::QwenImage21Transformer2DModel model(cfg);
        bool threw = false;
        try {
            model.load_weights(shards, "", [&]() { ++polls; return false; });
        } catch (const brodiffusion::LoadCancelled&) { threw = true; }
        CHECK(!threw);
        CHECK(polls == cfg.num_layers);
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

#ifndef BRODIFFUSION_WEIGHTS_DIR
#define BRODIFFUSION_WEIGHTS_DIR ""
#endif

static std::string weights_dir() {
    if (const char* e = std::getenv("BRODIFFUSION_WEIGHTS_DIR")) {
        if (e[0]) return e;
    }
    return BRODIFFUSION_WEIGHTS_DIR;
}

static void test_real_weights() {
    const char* opt = std::getenv("BRODIFFUSION_QI21_DIT_REAL");
    if (!opt || !opt[0] || opt[0] == '0') {
        std::printf("qi21_dit: real-weights skipped "
                    "(set BRODIFFUSION_QI21_DIT_REAL=1)\n");
        return;
    }
    const std::string tdir = weights_dir() + "/qwen-image-2.1/transformer";
    const std::string s1 =
        tdir + "/diffusion_pytorch_model-00001-of-00002.safetensors";
    if (!std::filesystem::exists(s1)) {
        std::printf("qi21_dit: real-weights skipped (no weights)\n");
        return;
    }
    std::vector<st::File> files;
    files.push_back(st::File::open(s1));
    files.push_back(st::File::open(
        tdir + "/diffusion_pytorch_model-00002-of-00002.safetensors"));
    std::vector<const st::File*> shards;
    for (const st::File& f : files) shards.push_back(&f);

    qd::QwenImage21Config cfg;   // real 2.1 config (header defaults)
    // INT8 weight-only: 7.2 GB instead of 14.2, so this fits a 24 GB card.
    cfg.quantize_weights = true;
    qd::QwenImage21Transformer2DModel model(cfg);
    model.load_weights(shards, "");

    const int hp = 16, wp = 16, text_seq = 24;
    const int img_len = hp * wp;
    std::vector<float> lat_h(
        static_cast<std::size_t>(img_len) * cfg.in_channels);
    for (std::size_t i = 0; i < lat_h.size(); ++i) {
        lat_h[i] = std::sin(0.01f * static_cast<float>(i)) * 0.5f;
    }
    std::vector<float> emb_h(
        static_cast<std::size_t>(text_seq) * cfg.context_in_dim);
    for (std::size_t i = 0; i < emb_h.size(); ++i) {
        emb_h[i] = std::sin(0.001f * static_cast<float>(i)) * 0.5f;
    }
    bt::Tensor lat = bdtest::bd_upload(lat_h, img_len, cfg.in_channels);
    bt::Tensor emb = bdtest::bd_upload(emb_h, text_seq, cfg.context_in_dim);

    bt::Tensor txt, out;
    model.encode_text(emb, txt);
    qd::QwenImage21PrefixCache cache;
    model.forward(lat, hp, wp, txt, 0.7f, &cache, out);
    bt::sync_all();
    CHECK(out.rows == img_len);
    CHECK(out.cols == cfg.out_channels);
    std::vector<float> v = download_any(out);
    int nonfinite = 0;
    double mean = 0.0;
    for (float f : v) { if (!std::isfinite(f)) ++nonfinite; else mean += f; }
    CHECK(nonfinite == 0);
    mean /= static_cast<double>(v.size());
    double var = 0.0;
    for (float f : v) var += (f - mean) * (f - mean);
    std::printf("qi21_dit: real-weights forward OK (mean %.5f std %.5f)\n",
                mean, std::sqrt(var / static_cast<double>(v.size())));

    // And the cached step runs on the real weights too.
    model.forward(lat, hp, wp, txt, 0.4f, &cache, out);
    bt::sync_all();
    std::vector<float> v2 = download_any(out);
    nonfinite = 0;
    for (float f : v2) if (!std::isfinite(f)) ++nonfinite;
    CHECK(nonfinite == 0);
}

int main() {
    try { bt::init(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "init failed: %s\n", e.what()); return 1;
    }
    try { test_synthetic(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "qi21_dit: synthetic exception: %s\n", e.what());
        return 1;
    }
    try { test_denoiser(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "qi21_dit: denoiser exception: %s\n", e.what());
        return 1;
    }
    try { test_hooks(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "qi21_dit: hooks exception: %s\n", e.what());
        return 1;
    }
    try { test_load_cancel(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "qi21_dit: load-cancel exception: %s\n", e.what());
        return 1;
    }
    try { test_real_weights(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "qi21_dit: real-weights exception: %s\n", e.what());
        return 1;
    }
    if (g_failures == 0) std::printf("qi21_dit: OK\n");
    else std::fprintf(stderr, "qi21_dit: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
