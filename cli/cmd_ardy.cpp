// ARDY subcommands — the text-to-motion stack's parity / rollout drivers.
//
// Motion rep codec, FSQ detokenizer, denoiser backbone, single-window sampler,
// autoregressive generator, and the LLM2Vec text conditioner. Each reads fixed
// inputs (raw f32/f64 or 1-D .npy) and dumps its result for the matching
// scripts/ardy_*_parity.sh to diff against the PyTorch reference.

#include "commands.h"

#include "brodiffusion/ardy/motion_rep.h"
#include "brodiffusion/ardy/fsq_decoder.h"
#include "brodiffusion/ardy/denoiser_backbone.h"
#include "brodiffusion/ardy/denoiser.h"
#include "brodiffusion/ardy/sampler.h"
#include "brodiffusion/ardy/text_conditioner.h"
#include "brodiffusion/detail/compute.h"

#include "brolm/llm2vec.h"
#include "brolm/llama3_tokenizer.h"

#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace brodiffusion::cli {

namespace st = brotensor::safetensors;

namespace {

std::vector<double> load_f64(const char* path, size_t n) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error(std::string("cannot open ") + path);
    std::vector<double> v(n);
    in.read(reinterpret_cast<char*>(v.data()),
            static_cast<std::streamsize>(n * sizeof(double)));
    if (!in) throw std::runtime_error(std::string("short read from ") + path);
    return v;
}

void dump_f64(const char* path, const std::vector<double>& v) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error(std::string("cannot write ") + path);
    out.write(reinterpret_cast<const char*>(v.data()),
              static_cast<std::streamsize>(v.size() * sizeof(double)));
}

// Read a 1-D float32 numpy .npy array (n elements) into host floats.
std::vector<float> load_npy_f32(const char* path, int n) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error(std::string("cannot open ") + path);
    char magic[8];
    in.read(magic, 8);  // \x93NUMPY + version(2)
    unsigned char hl[2];
    in.read(reinterpret_cast<char*>(hl), 2);
    const int hlen = hl[0] | (hl[1] << 8);
    in.seekg(10 + hlen, std::ios::beg);  // data starts after the header
    std::vector<float> v(n);
    in.read(reinterpret_cast<char*>(v.data()),
            static_cast<std::streamsize>(n * sizeof(float)));
    if (!in) throw std::runtime_error(std::string("short read from ") + path);
    return v;
}

// Read a 1-D float64 numpy .npy array (n elements) into host floats.
std::vector<float> load_npy_f64_as_f32(const char* path, int n) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error(std::string("cannot open ") + path);
    char magic[8];
    in.read(magic, 8);
    unsigned char hl[2];
    in.read(reinterpret_cast<char*>(hl), 2);
    const int hlen = hl[0] | (hl[1] << 8);
    in.seekg(10 + hlen, std::ios::beg);
    std::vector<double> d(n);
    in.read(reinterpret_cast<char*>(d.data()),
            static_cast<std::streamsize>(n * sizeof(double)));
    if (!in) throw std::runtime_error(std::string("short read from ") + path);
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) v[i] = static_cast<float>(d[i]);
    return v;
}

std::vector<float> load_raw_f32(const char* path, size_t n) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error(std::string("cannot open ") + path);
    std::vector<float> v(n);
    in.read(reinterpret_cast<char*>(v.data()),
            static_cast<std::streamsize>(n * sizeof(float)));
    if (!in) throw std::runtime_error(std::string("short read from ") + path);
    return v;
}

}  // namespace

// Reads float64 local rotation matrices + root positions, runs the ARDY motion
// codec forward (features) and inverse (posed joints + recovered local rots),
// and dumps all three as raw float64 to diff against the PyTorch reference
// (scripts/ardy_motionrep_ref.py).
int run_ardy_motionrep_fwd(int argc, char** argv) {
    const char* lr = arg_after(argc, argv, "--local-rots");
    const char* rp = arg_after(argc, argv, "--root-pos");
    const char* Ts = arg_after(argc, argv, "--T");
    const char* of = arg_after(argc, argv, "--out-features");
    const char* op = arg_after(argc, argv, "--out-posed");
    const char* ol = arg_after(argc, argv, "--out-local");
    if (!lr || !rp || !Ts || !of || !op || !ol) {
        std::fprintf(stderr, "ardy-motionrep-fwd: need --local-rots --root-pos "
                             "--T --out-features --out-posed --out-local\n");
        return 2;
    }
    const int T = std::atoi(Ts);
    constexpr int Jn = brodiffusion::ardy::ArdyMotionRep::kNumJoints;   // 34
    constexpr int Fd = brodiffusion::ardy::ArdyMotionRep::kFeatureDim;  // 414

    auto local = load_f64(lr, static_cast<size_t>(T) * Jn * 9);
    auto root  = load_f64(rp, static_cast<size_t>(T) * 3);

    brodiffusion::ardy::ArdyMotionRep rep(/*fps=*/25.0);
    std::vector<double> feats;
    rep.forward(local.data(), root.data(), T, feats);

    auto dec = rep.inverse(feats.data(), T, /*is_normalized=*/false,
                           /*posed_from_rot=*/true);

    dump_f64(of, feats);
    dump_f64(op, dec.posed_joints);
    dump_f64(ol, dec.local_rot_mats);
    std::printf("ardy-motionrep-fwd: T=%d features(%d,%d) posed(%d,%d,3) "
                "local(%d,%d,3,3)\n", T, T, Fd, T, Jn, T, Jn);
    return 0;
}

int run_ardy_fsq_detok(int argc, char** argv) {
    const char* w  = arg_after(argc, argv, "--weights");
    const char* mp = arg_after(argc, argv, "--pq-mean");
    const char* sp = arg_after(argc, argv, "--pq-std");
    const char* tp = arg_after(argc, argv, "--tokens");
    const char* rp = arg_after(argc, argv, "--local-root");
    const char* Ts = arg_after(argc, argv, "--T-tok");
    const char* op = arg_after(argc, argv, "--out");
    if (!w || !mp || !sp || !tp || !rp || !Ts || !op) {
        std::fprintf(stderr, "ardy-fsq-detok: need --weights --pq-mean --pq-std "
                             "--tokens --local-root --T-tok --out\n");
        return 2;
    }
    const int T_tok = std::atoi(Ts);
    brotensor::init();
    brodiffusion::ardy::FsqMotionDecoder dec;
    const auto& cfg = dec.config();

    auto f = st::File::open(w);
    dec.load_weights(f);
    auto mean = load_npy_f32(mp, cfg.token_dim);
    auto std_ = load_npy_f32(sp, cfg.token_dim);
    dec.set_post_quant_stats(mean.data(), std_.data(), cfg.token_dim);

    auto tokens = load_raw_f32(tp, static_cast<size_t>(T_tok) * cfg.token_dim);
    auto lroot  = load_raw_f32(rp, static_cast<size_t>(T_tok) *
                                       cfg.num_frames_per_token * cfg.external_cond_dim);

    brotensor::Tensor out;
    dec.detokenize(tokens.data(), lroot.data(), T_tok, out);
    brotensor::sync_all();
    dump_latent_f32(op, out);
    std::printf("ardy-fsq-detok: T_tok=%d out(%d,%d) -> %d frames x %d\n",
                T_tok, out.rows, out.cols, T_tok * cfg.num_frames_per_token,
                cfg.output_dim);
    return 0;
}

int run_ardy_denoiser_fwd(int argc, char** argv) {
    const char* w  = arg_after(argc, argv, "--weights");
    const char* mp = arg_after(argc, argv, "--mean");     // motion mean.npy (414)
    const char* sp = arg_after(argc, argv, "--std");      // motion std.npy (414)
    const char* xp = arg_after(argc, argv, "--hybrid");   // (T_tok, 148) raw f32
    const char* tp = arg_after(argc, argv, "--text");     // (1, 4096) raw f32
    const char* Ts = arg_after(argc, argv, "--T-tok");
    const char* ts = arg_after(argc, argv, "--timestep");
    const char* hs = arg_after(argc, argv, "--heading");
    const char* hts = arg_after(argc, argv, "--history-tok");  // optional, default 0
    const char* op = arg_after(argc, argv, "--out");
    if (!w || !mp || !sp || !xp || !tp || !Ts || !ts || !hs || !op) {
        std::fprintf(stderr, "ardy-denoiser-fwd: need --weights --mean --std "
                             "--hybrid --text --T-tok --timestep --heading --out "
                             "[--history-tok N]\n");
        return 2;
    }
    const int T_tok = std::atoi(Ts);
    const int timestep = std::atoi(ts);
    const float heading = static_cast<float>(std::atof(hs));
    const int history_tok = hts ? std::atoi(hts) : 0;

    brotensor::init();
    brodiffusion::ardy::ArdyDenoiser dn;

    auto f = st::File::open(w);
    dn.load_weights(f);
    const int sdim = dn.stats_dim();  // 418 (float64 stats)
    auto mean = load_npy_f64_as_f32(mp, sdim);
    auto std_ = load_npy_f64_as_f32(sp, sdim);
    dn.set_motion_stats(mean.data(), std_.data(), sdim);

    auto hyb  = load_raw_f32(xp, static_cast<size_t>(T_tok) * dn.hybrid_dim());
    auto text = load_raw_f32(tp, 4096);

    brotensor::Tensor out;
    dn.forward(hyb.data(), text.data(), timestep, heading, T_tok, out, history_tok);
    brotensor::sync_all();
    dump_latent_f32(op, out);
    std::printf("ardy-denoiser-fwd: T_tok=%d hist=%d out(%d,%d)\n", T_tok,
                history_tok, out.rows, out.cols);
    return 0;
}

// Full spaced-DDIM text-to-motion window: denoise a fresh generation window from
// initial noise with text-only CFG. Matches the reference generation loop.
int run_ardy_sample(int argc, char** argv) {
    const char* w  = arg_after(argc, argv, "--weights");
    const char* mp = arg_after(argc, argv, "--mean");     // motion mean.npy (418, f64)
    const char* sp = arg_after(argc, argv, "--std");      // motion std.npy  (418, f64)
    const char* xp = arg_after(argc, argv, "--x-init");   // (T_tok,148) raw f32 noise
    const char* tp = arg_after(argc, argv, "--text");     // (1,4096) raw f32
    const char* Ts = arg_after(argc, argv, "--T-tok");
    const char* Ss = arg_after(argc, argv, "--steps");
    const char* cs = arg_after(argc, argv, "--cfg");
    const char* hs = arg_after(argc, argv, "--heading");
    const char* op = arg_after(argc, argv, "--out");
    if (!w || !mp || !sp || !xp || !tp || !Ts || !Ss || !cs || !hs || !op) {
        std::fprintf(stderr, "ardy-sample: need --weights --mean --std --x-init "
                             "--text --T-tok --steps --cfg --heading --out\n");
        return 2;
    }
    const int T_tok = std::atoi(Ts);
    const int steps = std::atoi(Ss);
    const float cfg_w = static_cast<float>(std::atof(cs));
    const float heading = static_cast<float>(std::atof(hs));

    brotensor::init();
    brodiffusion::ardy::ArdyDenoiser dn;
    auto f = st::File::open(w);
    dn.load_weights(f);
    const int sdim = dn.stats_dim();  // 418 (float64 stats)
    auto mean = load_npy_f64_as_f32(mp, sdim);
    auto std_ = load_npy_f64_as_f32(sp, sdim);
    dn.set_motion_stats(mean.data(), std_.data(), sdim);

    auto x_init = load_raw_f32(xp, static_cast<size_t>(T_tok) * dn.hybrid_dim());
    auto text   = load_raw_f32(tp, 4096);

    brodiffusion::ardy::ArdyWindowSampler sampler(dn);
    std::vector<float> out;
    sampler.sample(x_init.data(), text.data(), T_tok, heading, steps, cfg_w, out);

    std::ofstream of(op, std::ios::binary | std::ios::trunc);
    if (!of) { std::fprintf(stderr, "ardy-sample: cannot write %s\n", op); return 1; }
    of.write(reinterpret_cast<const char*>(out.data()),
             static_cast<std::streamsize>(out.size() * sizeof(float)));
    std::printf("ardy-sample: T_tok=%d steps=%d cfg=%.3f out(%d,%d)\n",
                T_tok, steps, cfg_w, T_tok, dn.hybrid_dim());
    return 0;
}

// Autoregressive text-to-motion rollout: chain windows into a full-length hybrid
// sequence (recenter + requantize + global-translation tracking). Emits the world-
// frame hybrid before FSQ detokenization. Matches ardy_model.py Ardy.__call__
// (text-only, no history/constraints/crop).
int run_ardy_generate(int argc, char** argv) {
    const char* dw  = arg_after(argc, argv, "--denoiser");   // denoiser.safetensors
    const char* tw  = arg_after(argc, argv, "--tokenizer");  // tokenizer.safetensors
    const char* mp  = arg_after(argc, argv, "--mean");       // motion mean.npy (418, f64)
    const char* sp  = arg_after(argc, argv, "--std");        // motion std.npy  (418, f64)
    const char* qmp = arg_after(argc, argv, "--pq-mean");    // post-quant mean (128, f32)
    const char* qsp = arg_after(argc, argv, "--pq-std");     // post-quant std  (128, f32)
    const char* tp  = arg_after(argc, argv, "--text");       // (4096) raw f32
    const char* np_ = arg_after(argc, argv, "--noise");      // (W*13*148) raw f32
    const char* Fs  = arg_after(argc, argv, "--frames");
    const char* ss  = arg_after(argc, argv, "--steps");
    const char* cs  = arg_after(argc, argv, "--cfg");
    const char* hs  = arg_after(argc, argv, "--heading");
    const char* op  = arg_after(argc, argv, "--out");        // hybrid (T_tok,148)
    const char* omp = arg_after(argc, argv, "--out-motion"); // optional explicit (F,414)
    if (!dw || !tw || !mp || !sp || !qmp || !qsp || !tp || !np_ || !Fs || !ss ||
        !cs || !hs || !op) {
        std::fprintf(stderr, "ardy-generate: need --denoiser --tokenizer --mean "
                             "--std --pq-mean --pq-std --text --noise --frames "
                             "--steps --cfg --heading --out\n");
        return 2;
    }
    const int frames = std::atoi(Fs);
    const int steps  = std::atoi(ss);
    const float cfg_w = static_cast<float>(std::atof(cs));
    const float heading = static_cast<float>(std::atof(hs));

    brotensor::init();
    brodiffusion::ardy::ArdyDenoiser dn;
    {
        auto f = st::File::open(dw);
        dn.load_weights(f);
    }
    const int sdim = dn.stats_dim();  // 418 (f64 stats)
    auto mean = load_npy_f64_as_f32(mp, sdim);
    auto std_ = load_npy_f64_as_f32(sp, sdim);
    dn.set_motion_stats(mean.data(), std_.data(), sdim);

    brodiffusion::ardy::FsqMotionDecoder fsq;
    {
        auto f = st::File::open(tw);
        fsq.load_weights(f);
    }
    const int td = fsq.config().token_dim;  // 128
    auto qmean = load_npy_f32(qmp, td);
    auto qstd  = load_npy_f32(qsp, td);
    fsq.set_post_quant_stats(qmean.data(), qstd.data(), td);

    brodiffusion::ardy::ArdyMotionGenerator gen(dn, fsq);
    const int W = gen.num_windows(frames);
    const int hyb = dn.hybrid_dim();
    const int fpt = dn.config().num_frames_per_token;
    const int G = 52 / fpt;  // tokens per window

    auto text  = load_raw_f32(tp, 4096);
    auto noise = load_raw_f32(np_, static_cast<size_t>(W) * G * hyb);

    std::vector<float> out;
    int T_tok = 0;
    gen.generate_hybrid(text.data(), frames, heading, steps, cfg_w, noise.data(),
                        out, T_tok);

    std::ofstream of(op, std::ios::binary | std::ios::trunc);
    if (!of) { std::fprintf(stderr, "ardy-generate: cannot write %s\n", op); return 1; }
    of.write(reinterpret_cast<const char*>(out.data()),
             static_cast<std::streamsize>(out.size() * sizeof(float)));
    std::printf("ardy-generate: frames=%d windows=%d T_tok=%d out(%d,%d)\n",
                frames, W, T_tok, T_tok, hyb);

    if (omp) {
        std::vector<float> motion;
        gen.detokenize_to_motion(out.data(), T_tok, motion);
        const int mrd = dn.config().motion_rep_dim;  // 414
        const int F = T_tok * fpt;
        std::ofstream mf(omp, std::ios::binary | std::ios::trunc);
        if (!mf) { std::fprintf(stderr, "ardy-generate: cannot write %s\n", omp); return 1; }
        mf.write(reinterpret_cast<const char*>(motion.data()),
                 static_cast<std::streamsize>(motion.size() * sizeof(float)));
        std::printf("ardy-generate: motion(%d,%d)\n", F, mrd);
    }
    return 0;
}

// Detokenize an explicit hybrid sequence into ARDY motion features (F,414). Used
// to isolate the FSQ-decode + local-root conditioning (get_explicit_motion_from_
// hybrid) from the autoregressive rollout: feeding a fixed hybrid tests the
// detokenize path alone.
int run_ardy_detok_motion(int argc, char** argv) {
    const char* dw  = arg_after(argc, argv, "--denoiser");
    const char* tw  = arg_after(argc, argv, "--tokenizer");
    const char* mp  = arg_after(argc, argv, "--mean");
    const char* sp  = arg_after(argc, argv, "--std");
    const char* qmp = arg_after(argc, argv, "--pq-mean");
    const char* qsp = arg_after(argc, argv, "--pq-std");
    const char* xp  = arg_after(argc, argv, "--hybrid");   // (T_tok,148) raw f32
    const char* Ts  = arg_after(argc, argv, "--T-tok");
    const char* op  = arg_after(argc, argv, "--out");
    const char* pp  = arg_after(argc, argv, "--out-posed");  // optional (F,J,3) f32
    if (!dw || !tw || !mp || !sp || !qmp || !qsp || !xp || !Ts || !op) {
        std::fprintf(stderr, "ardy-detok-motion: need --denoiser --tokenizer "
                             "--mean --std --pq-mean --pq-std --hybrid --T-tok "
                             "--out\n");
        return 2;
    }
    const int T_tok = std::atoi(Ts);

    brotensor::init();
    brodiffusion::ardy::ArdyDenoiser dn;
    { auto f = st::File::open(dw); dn.load_weights(f); }
    const int sdim = dn.stats_dim();
    auto mean = load_npy_f64_as_f32(mp, sdim);
    auto std_ = load_npy_f64_as_f32(sp, sdim);
    dn.set_motion_stats(mean.data(), std_.data(), sdim);

    brodiffusion::ardy::FsqMotionDecoder fsq;
    { auto f = st::File::open(tw); fsq.load_weights(f); }
    const int td = fsq.config().token_dim;
    auto qmean = load_npy_f32(qmp, td);
    auto qstd  = load_npy_f32(qsp, td);
    fsq.set_post_quant_stats(qmean.data(), qstd.data(), td);

    auto hyb = load_raw_f32(xp, static_cast<size_t>(T_tok) * dn.hybrid_dim());

    brodiffusion::ardy::ArdyMotionGenerator gen(dn, fsq);
    std::vector<float> motion;
    gen.detokenize_to_motion(hyb.data(), T_tok, motion);

    std::ofstream of(op, std::ios::binary | std::ios::trunc);
    if (!of) { std::fprintf(stderr, "ardy-detok-motion: cannot write %s\n", op); return 1; }
    of.write(reinterpret_cast<const char*>(motion.data()),
             static_cast<std::streamsize>(motion.size() * sizeof(float)));
    const int F = T_tok * dn.config().num_frames_per_token;

    // Optional: unnormalize + FK to world joint positions, exercising
    // ArdyMotionRep::inverse(is_normalized=true) (== motion_rep.inverse with
    // is_normalized=True in the reference).
    if (pp) {
        brodiffusion::ardy::ArdyMotionRep rep(/*fps=*/dn.config().fps);
        rep.set_motion_stats(mean.data(), std_.data(), sdim);
        std::vector<double> mdbl(motion.begin(), motion.end());
        auto dec = rep.inverse(mdbl.data(), F, /*is_normalized=*/true);
        std::vector<float> posed(dec.posed_joints.size());
        for (size_t i = 0; i < posed.size(); ++i)
            posed[i] = static_cast<float>(dec.posed_joints[i]);
        std::ofstream pf(pp, std::ios::binary | std::ios::trunc);
        if (!pf) { std::fprintf(stderr, "ardy-detok-motion: cannot write %s\n", pp); return 1; }
        pf.write(reinterpret_cast<const char*>(posed.data()),
                 static_cast<std::streamsize>(posed.size() * sizeof(float)));
    }

    std::printf("ardy-detok-motion: T_tok=%d motion(%d,%d)%s\n", T_tok, F,
                dn.config().motion_rep_dim, pp ? " +posed" : "");
    return 0;
}

// ARDY text conditioning: prompt -> the (4096) pooled LLM2Vec feature the
// denoiser cross-attends to. Reproduces ardy's prepare_for_tokenization +
// tokenize + skip-instruction mean pooling (see text_conditioner.h). With
// --dump-ids / --dump-mask it also writes the built token sequence + pool mask,
// so the tokenization can be diffed against ardy without the 8B forward.
int run_ardy_text_feat(int argc, char** argv) {
    const char* ew  = arg_after(argc, argv, "--encoder");       // model.safetensors
    const char* cp  = arg_after(argc, argv, "--config");        // llama config.json
    const char* tj  = arg_after(argc, argv, "--tokenizer-json"); // tokenizer.json
    const char* txt = arg_after(argc, argv, "--text");          // prompt string
    const char* op  = arg_after(argc, argv, "--out");           // (4096) f32
    const char* di  = arg_after(argc, argv, "--dump-ids");      // optional (L) i32
    const char* dm  = arg_after(argc, argv, "--dump-mask");     // optional (L) f32
    if (!tj || !txt) {
        std::fprintf(stderr, "ardy-text-feat: need --tokenizer-json --text "
                             "[--encoder <st> --config <json> --out <f32>] "
                             "[--dump-ids <i32>] [--dump-mask <f32>]\n");
        return 2;
    }

    brolm::llama3::Tokenizer tok = brolm::llama3::Tokenizer::load(tj);

    // Token-only path (no encoder): build ids + pool mask and dump them.
    if (di || dm) {
        std::vector<std::int32_t> ids;
        std::vector<float> mask;
        brodiffusion::ardy::build_ardy_text_tokens(tok, txt, ids, mask);
        if (di) {
            std::ofstream f(di, std::ios::binary | std::ios::trunc);
            f.write(reinterpret_cast<const char*>(ids.data()),
                    static_cast<std::streamsize>(ids.size() * sizeof(std::int32_t)));
        }
        if (dm) {
            std::ofstream f(dm, std::ios::binary | std::ios::trunc);
            f.write(reinterpret_cast<const char*>(mask.data()),
                    static_cast<std::streamsize>(mask.size() * sizeof(float)));
        }
        std::printf("ardy-text-feat: L=%zu pooled=%d\n", ids.size(),
                    static_cast<int>(std::count(mask.begin(), mask.end(), 1.0f)));
        if (!op) return 0;   // tokens-only run
    }

    if (!ew || !cp || !op) {
        std::fprintf(stderr, "ardy-text-feat: --encoder --config --out required "
                             "for the pooled embedding\n");
        return 2;
    }

    brotensor::init();
    brolm::llm2vec::Config cfg = brolm::llm2vec::Config::load(cp);
    brolm::llm2vec::Encoder enc(cfg);
    { auto f = st::File::open(ew); enc.load_weights(f); }

    std::vector<float> feat;
    brodiffusion::ardy::ardy_text_feat(tok, enc, txt, feat);

    std::ofstream of(op, std::ios::binary | std::ios::trunc);
    if (!of) { std::fprintf(stderr, "ardy-text-feat: cannot write %s\n", op); return 1; }
    of.write(reinterpret_cast<const char*>(feat.data()),
             static_cast<std::streamsize>(feat.size() * sizeof(float)));
    double mean = 0.0;
    for (float v : feat) mean += v;
    mean /= (feat.empty() ? 1.0 : static_cast<double>(feat.size()));
    std::printf("ardy-text-feat: text_feat(%zu) mean %.6f\n", feat.size(), mean);
    return 0;
}

int run_ardy_backbone_fwd(int argc, char** argv) {
    const char* w  = arg_after(argc, argv, "--weights");
    const char* st_ = arg_after(argc, argv, "--stage");     // root | body
    const char* xp = arg_after(argc, argv, "--x");          // (T, latent) raw f32
    const char* tp = arg_after(argc, argv, "--text");       // (num_text_tokens, llm) raw f32
    const char* Ts = arg_after(argc, argv, "--T");
    const char* ts = arg_after(argc, argv, "--timestep");
    const char* hs = arg_after(argc, argv, "--heading");
    const char* op = arg_after(argc, argv, "--out");
    if (!w || !st_ || !xp || !tp || !Ts || !ts || !hs || !op) {
        std::fprintf(stderr, "ardy-backbone-fwd: need --weights --stage --x "
                             "--text --T --timestep --heading --out\n");
        return 2;
    }
    const int T = std::atoi(Ts);
    const int timestep = std::atoi(ts);
    const float heading = static_cast<float>(std::atof(hs));
    const bool is_body = std::strcmp(st_, "body") == 0;

    brotensor::init();
    brodiffusion::ardy::ArdyDenoiserBackbone::Config cfg;
    cfg.output_dim = is_body ? 128 : 20;
    brodiffusion::ardy::ArdyDenoiserBackbone bb(cfg);

    auto f = st::File::open(w);
    bb.load_weights(f, std::string("denoiser.backbone.") +
                       (is_body ? "body_model." : "root_model."));

    auto xh = load_raw_f32(xp, static_cast<size_t>(T) * cfg.latent_dim);
    auto th = load_raw_f32(tp, static_cast<size_t>(cfg.num_text_tokens) * cfg.llm_dim);
    brotensor::Tensor x = brodiffusion::detail::upload_host(xh.data(), T, cfg.latent_dim);

    // First-generation window: token_index = arange(T) (origin at frame 0).
    std::vector<int> tok_idx(T);
    for (int i = 0; i < T; ++i) tok_idx[i] = i;

    brotensor::Tensor out;
    bb.forward(x, th.data(), timestep, heading, tok_idx.data(), T,
               /*key_mask=*/nullptr, out);
    brotensor::sync_all();
    dump_latent_f32(op, out);
    std::printf("ardy-backbone-fwd: stage=%s T=%d out(%d,%d)\n",
                is_body ? "body" : "root", T, out.rows, out.cols);
    return 0;
}

}  // namespace brodiffusion::cli
