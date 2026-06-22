// DiT-only nondeterminism harness — loads ONLY the PixArt denoiser (no T5, no
// VAE), feeds random conditioning + latent, and runs forward_body. Small enough
// to run under compute-sanitizer (racecheck/initcheck) in seconds to localize
// the nondeterministic kernel exposed by CUDA-graph capture. Also reports an
// eager-vs-eager bit-mismatch as a bonus signal.
//
// usage: dit_race <model_dir> [iters]
#include "brodiffusion/dit/pixart.h"
#include "brodiffusion/denoiser.h"

#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"
#include "brotensor/ops.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace bt = brotensor;
using namespace brodiffusion;

static bt::Tensor rand_dev(int r, int c, unsigned seed) {
    std::mt19937 g(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> h(static_cast<size_t>(r) * c);
    for (auto& x : h) x = nd(g);
    return bt::Tensor::from_host(h.data(), r, c).to(bt::default_device());
}

static size_t mismatch(const bt::Tensor& a, const bt::Tensor& b) {
    bt::sync_all();
    size_t mm = 0;
    if (a.dtype == bt::Dtype::FP32) {
        const std::vector<float> x = a.to_host_vector();
        const std::vector<float> y = b.to_host_vector();
        for (size_t i = 0; i < x.size() && i < y.size(); ++i)
            if (x[i] != y[i]) ++mm;
    } else {
        const std::vector<uint16_t> x = a.to_host_vector_fp16();
        const std::vector<uint16_t> y = b.to_host_vector_fp16();
        for (size_t i = 0; i < x.size() && i < y.size(); ++i)
            if (x[i] != y[i]) ++mm;
    }
    return mm;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: dit_race <model_dir> [iters]\n"); return 2; }
    const std::string model = argv[1];
    const int iters = argc > 2 ? std::atoi(argv[2]) : 4;
    // Tiny H/W (e.g. 16) shrinks every grid + the O(L^2) attention so
    // compute-sanitizer racecheck stays within memory while running the SAME
    // kernels. Must be divisible by patch_size (2).
    const int Harg = argc > 3 ? std::atoi(argv[3]) : 128;
    const int Warg = argc > 4 ? std::atoi(argv[4]) : 128;

    bt::init();
    std::fprintf(stderr, "[dit_race] default device=%d (0=CPU,1=CUDA)\n",
                 static_cast<int>(bt::default_device()));

    fs::path tdir = fs::path(model) / "transformer";
    std::string sfile;
    for (const auto& e : fs::directory_iterator(tdir))
        if (e.path().extension() == ".safetensors") { sfile = e.path().string(); break; }
    if (sfile.empty()) {
        std::fprintf(stderr, "[dit_race] no .safetensors in %s\n", tdir.string().c_str());
        return 2;
    }
    std::fprintf(stderr, "[dit_race] transformer: %s\n", sfile.c_str());
    bt::safetensors::File f = bt::safetensors::File::open(sfile);

    dit::PixArtConfig cfg;
    dit::PixArtDenoiser d(cfg);
    d.load_weights(f, "");
    d.finalize_weights();

    Conditioning cond;
    cond.text_embeddings = rand_dev(13, cfg.caption_channels, 123);  // fake caption
    cond.has_uncond = false;
    PreparedConditioning prepared = d.prepare(cond);

    const int H = Harg, W = Warg, IC = cfg.in_channels;
    bt::Tensor latent_f32 = rand_dev(1, IC * H * W, 777);
    bt::Tensor latent;
    if (d.compute_dtype() != bt::Dtype::FP32)
        bt::cast(latent_f32, latent, d.compute_dtype());
    else
        latent = latent_f32.clone();

    bt::Tensor out1, out2;
    for (int it = 0; it < iters; ++it) {
        d.prepare_step(20.0f, prepared);
        d.forward_body(latent, H, W, prepared, Branch::Cond, out1);
        d.forward_body(latent, H, W, prepared, Branch::Cond, out2);
        const size_t mm = mismatch(out1, out2);
        std::fprintf(stderr, "[dit_race] iter %d: eager-twice mismatch=%zu/%lld\n",
                     it, mm, static_cast<long long>(out1.rows) * out1.cols);
    }
    return 0;
}
