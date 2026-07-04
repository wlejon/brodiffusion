// Kernel-level microbenchmark at Krea 2 shapes — attributes per-step time to
// the individual GEMM / attention kernels so implementation-level numbers
// (s/step) can be compared against per-kernel roofline and cuBLAS-class
// baselines (e.g. a torch script timing the same shapes).
//
// Shapes: joint sequence B = text(30..512) + image(4096) rows; DiT body
// linears at hidden 6144 / intermediate 16384 / kv 1536; 48/12 GQA flash
// attention at head_dim 128.
//
// Usage: bench_krea2_kernels [B]   (default 4126)
#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace bt = brotensor;

namespace {

bt::Tensor randn_fp16(int rows, int cols, std::mt19937& rng) {
    std::normal_distribution<float> d(0.0f, 0.02f);
    std::vector<std::uint16_t> h(static_cast<std::size_t>(rows) * cols);
    for (auto& v : h) v = bt::fp32_to_fp16_bits(d(rng));
    bt::Tensor cpu = bt::Tensor::empty_on(bt::Device::CPU, rows, cols,
                                          bt::Dtype::FP16);
    std::memcpy(cpu.host_raw_mut(), h.data(), h.size() * 2);
    return cpu.to(bt::default_device());
}

struct QW {
    bt::Tensor w8, scales;
};

QW quant_random(int out, int in, std::mt19937& rng) {
    std::normal_distribution<float> d(0.0f, 0.02f);
    const std::size_t n = static_cast<std::size_t>(out) * in;
    std::vector<std::uint16_t> w16(n);
    for (auto& v : w16) v = bt::fp32_to_fp16_bits(d(rng));
    std::vector<std::int8_t> q(n);
    std::vector<float> sc(static_cast<std::size_t>(out));
    bt::quantize_int8_per_row_host(w16.data(), out, in, q.data(), sc.data());
    QW r;
    r.w8 = bt::Tensor::from_host_int8(q.data(), out, in);
    r.scales = bt::Tensor::from_host(sc.data(), out, 1);
    return r;
}

template <typename F>
double time_op(F&& f, int iters) {
    f();                     // warmup + autotune
    f();
    bt::sync_all();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) f();
    bt::sync_all();
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now() - t0).count() /
           iters;
}

void report(const char* name, double secs, double flops) {
    std::printf("%-34s %8.3f ms   %7.1f TFLOPS\n", name, secs * 1e3,
                flops / secs / 1e12);
}

}  // namespace

int main(int argc, char** argv) {
    const int B = (argc > 1) ? std::atoi(argv[1]) : 4126;
    const int H = 6144, FF = 16384, KV = 1536, HD = 128, NQ = 48, NKV = 12;
    const int iters = 20;

    bt::init();
    if (bt::default_device() != bt::Device::CUDA) {
        std::fprintf(stderr, "bench_krea2_kernels: CUDA backend required\n");
        return 1;
    }
    std::mt19937 rng(0);

    bt::Tensor x = randn_fp16(B, H, rng);
    bt::Tensor xff = randn_fp16(B, FF, rng);
    bt::Tensor y;

    std::printf("B = %d rows (joint seq), %d iters/op\n\n", B, iters);

    // ── INT8 W8A16 linears (the quantized-load hot path) ────────────────
    {
        QW q = quant_random(H, H, rng);
        report("int8w  6144x6144  (attn q/gate/out)",
               time_op([&] { bt::linear_forward_batched_int8w_fp16(
                                 q.w8, q.scales, nullptr, x, y); }, iters),
               2.0 * B * H * H);
    }
    {
        QW q = quant_random(KV, H, rng);
        report("int8w  1536x6144  (attn k/v)",
               time_op([&] { bt::linear_forward_batched_int8w_fp16(
                                 q.w8, q.scales, nullptr, x, y); }, iters),
               2.0 * B * KV * H);
    }
    {
        QW q = quant_random(FF, H, rng);
        report("int8w 16384x6144  (ff gate/up)",
               time_op([&] { bt::linear_forward_batched_int8w_fp16(
                                 q.w8, q.scales, nullptr, x, y); }, iters),
               2.0 * B * FF * H);
    }
    {
        QW q = quant_random(H, FF, rng);
        report("int8w  6144x16384 (ff down)",
               time_op([&] { bt::linear_forward_batched_int8w_fp16(
                                 q.w8, q.scales, nullptr, xff, y); }, iters),
               2.0 * B * H * FF);
    }

    // ── dense FP16 linear for comparison ────────────────────────────────
    {
        bt::Tensor w = randn_fp16(H, H, rng);
        report("fp16   6144x6144  (dense ref)",
               time_op([&] { bt::linear_forward_batched_fp16(
                                 w, nullptr, x, y); }, iters),
               2.0 * B * H * H);
        bt::Tensor wff = randn_fp16(FF, H, rng);
        report("fp16  16384x6144  (dense ref)",
               time_op([&] { bt::linear_forward_batched_fp16(
                                 wff, nullptr, x, y); }, iters),
               2.0 * B * FF * H);
    }

    // ── flash attention at the body's GQA shape ─────────────────────────
    {
        bt::Tensor Q = randn_fp16(B, NQ * HD, rng);
        bt::Tensor K = randn_fp16(B, NQ * HD, rng);   // pre-widened K/V
        bt::Tensor V = randn_fp16(B, NQ * HD, rng);
        bt::Tensor O;
        std::vector<float> mask(static_cast<std::size_t>(B), 1.0f);
        bt::Tensor mask_dev = bt::Tensor::from_host(mask.data(), B, 1);
        const float* dmask = static_cast<const float*>(mask_dev.data);
        // 2 matmuls (QK^T, PV): 2 * 2 * B^2 * D_total
        report("flash attn 48h x128 (widened kv)",
               time_op([&] { bt::flash_attention_forward(
                                 Q, K, V, dmask, NQ, false, O); }, iters),
               4.0 * static_cast<double>(B) * B * (NQ * HD));

        // GQA K/V widening cost (96 strided copies, as gqa_attention_masked).
        bt::Tensor Kn = randn_fp16(B, NKV * HD, rng);
        bt::Tensor Krep, Vrep;
        Krep = bt::Tensor::empty(B, NQ * HD, bt::Dtype::FP16);
        Vrep = bt::Tensor::empty(B, NQ * HD, bt::Dtype::FP16);
        const int group = NQ / NKV;
        report("gqa k/v widen (96 strided copies)",
               time_op([&] {
                   for (int qh = 0; qh < NQ; ++qh) {
                       const int kvh = qh / group;
                       bt::copy_d2d_strided(Kn, kvh * HD, NKV * HD, Krep,
                                            qh * HD, NQ * HD, HD, B);
                       bt::copy_d2d_strided(Kn, kvh * HD, NKV * HD, Vrep,
                                            qh * HD, NQ * HD, HD, B);
                   }
               }, iters),
               1.0);  // not FLOPs-meaningful
    }

    return 0;
}
