#include "brodiffusion/value_head.h"

#include "brotensor/ops.h"
#include "brotensor/tensor.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <stdexcept>
#include <vector>

namespace brodiffusion::value_head {

namespace bt = ::brotensor;

namespace {

void upload_fp32_normal(int rows, int cols, float std,
                        std::mt19937_64& rng, bt::GpuTensor& dst) {
    std::normal_distribution<float> nrm(0.0f, std);
    std::vector<float> host(static_cast<std::size_t>(rows) * cols);
    for (auto& v : host) v = nrm(rng);
    bt::upload(host.data(), rows, cols, dst);
}

void upload_fp32_zero(int rows, int cols, bt::GpuTensor& dst) {
    std::vector<float> host(static_cast<std::size_t>(rows) * cols, 0.0f);
    bt::upload(host.data(), rows, cols, dst);
}

}  // namespace

ValueHead::ValueHead(const Config& cfg) : cfg_(cfg) {
    if (cfg_.latent_dim <= 0 || cfg_.branching < 1 || cfg_.hidden_dim <= 0) {
        throw std::runtime_error("ValueHead: bad config dims");
    }

    std::mt19937_64 rng(cfg_.seed);
    upload_fp32_normal(cfg_.hidden_dim, in_dim(),    cfg_.init_std_W1, rng, W1_);
    upload_fp32_zero  (cfg_.hidden_dim, 1,                              b1_);
    upload_fp32_normal(1,               cfg_.hidden_dim, cfg_.init_std_W2, rng, W2_);
    upload_fp32_zero  (1,               1,                              b2_);

    // Adam state — same shape as the param, zeroed.
    upload_fp32_zero(cfg_.hidden_dim, in_dim(),    m_W1_);
    upload_fp32_zero(cfg_.hidden_dim, in_dim(),    v_W1_);
    upload_fp32_zero(cfg_.hidden_dim, 1,           m_b1_);
    upload_fp32_zero(cfg_.hidden_dim, 1,           v_b1_);
    upload_fp32_zero(1,               cfg_.hidden_dim, m_W2_);
    upload_fp32_zero(1,               cfg_.hidden_dim, v_W2_);
    upload_fp32_zero(1,               1,           m_b2_);
    upload_fp32_zero(1,               1,           v_b2_);

    // Grad scratch — same shape as the param.
    upload_fp32_zero(cfg_.hidden_dim, in_dim(),    g_W1_);
    upload_fp32_zero(cfg_.hidden_dim, 1,           g_b1_);
    upload_fp32_zero(1,               cfg_.hidden_dim, g_W2_);
    upload_fp32_zero(1,               1,           g_b2_);
}

void ValueHead::forward(const bt::GpuTensor& X_BD, bt::GpuTensor& Y_B1) {
    // Cache X for backward. We could keep a view, but the trainer reuses
    // the same X buffer for the backward call so an owning clone keeps the
    // contract clean and avoids surprises if the caller mutates X.
    X_cached_ = X_BD.clone();

    bt::linear_forward_batched_gpu(W1_, b1_, X_BD, h_pre_);
    bt::relu_forward_batched_gpu(h_pre_, h_);
    bt::linear_forward_batched_gpu(W2_, b2_, h_, Y_B1);
}

void ValueHead::backward_and_step(const bt::GpuTensor& dY_B1) {
    g_W1_.zero();  g_b1_.zero();
    g_W2_.zero();  g_b2_.zero();

    // dH = dY * W2 (sort of) — handled by linear_backward_batched_gpu.
    bt::linear_backward_batched_gpu(W2_, h_, dY_B1, dH_, g_W2_, g_b2_);
    bt::relu_backward_batched_gpu(h_pre_, dH_, dHpre_);
    bt::linear_backward_batched_gpu(W1_, X_cached_, dHpre_, dX_unused_,
                                    g_W1_, g_b1_);

    ++step_n_;
    bt::adam_step_gpu(W1_, g_W1_, m_W1_, v_W1_,
                      cfg_.lr, cfg_.beta1, cfg_.beta2, cfg_.eps, step_n_);
    bt::adam_step_gpu(b1_, g_b1_, m_b1_, v_b1_,
                      cfg_.lr, cfg_.beta1, cfg_.beta2, cfg_.eps, step_n_);
    bt::adam_step_gpu(W2_, g_W2_, m_W2_, v_W2_,
                      cfg_.lr, cfg_.beta1, cfg_.beta2, cfg_.eps, step_n_);
    bt::adam_step_gpu(b2_, g_b2_, m_b2_, v_b2_,
                      cfg_.lr, cfg_.beta1, cfg_.beta2, cfg_.eps, step_n_);
}

namespace {

void dump_tensor(std::ofstream& f, const bt::GpuTensor& t) {
    std::vector<float> host(t.size());
    bt::download(t, host.data());
    f.write(reinterpret_cast<const char*>(host.data()),
            static_cast<std::streamsize>(host.size() * sizeof(float)));
}

void load_tensor(std::ifstream& f, int rows, int cols, bt::GpuTensor& dst) {
    std::vector<float> host(static_cast<std::size_t>(rows) * cols);
    f.read(reinterpret_cast<char*>(host.data()),
           static_cast<std::streamsize>(host.size() * sizeof(float)));
    if (!f) throw std::runtime_error("ValueHead::load: short read");
    bt::upload(host.data(), rows, cols, dst);
}

}  // namespace

void ValueHead::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("ValueHead::save: cannot open " + path);
    auto w_i32 = [&](std::int32_t v) {
        f.write(reinterpret_cast<const char*>(&v), sizeof(v));
    };
    w_i32(0x42445648);   // 'BDVH'
    w_i32(1);
    w_i32(cfg_.latent_dim);
    w_i32(cfg_.branching);
    w_i32(cfg_.hidden_dim);
    dump_tensor(f, W1_);
    dump_tensor(f, b1_);
    dump_tensor(f, W2_);
    dump_tensor(f, b2_);
}

void ValueHead::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("ValueHead::load: cannot open " + path);
    std::int32_t magic = 0, ver = 0, ld = 0, B = 0, hd = 0;
    f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    f.read(reinterpret_cast<char*>(&ver),   sizeof(ver));
    f.read(reinterpret_cast<char*>(&ld),    sizeof(ld));
    f.read(reinterpret_cast<char*>(&B),     sizeof(B));
    f.read(reinterpret_cast<char*>(&hd),    sizeof(hd));
    if (magic != 0x42445648 || ver != 1) {
        throw std::runtime_error("ValueHead::load: bad magic/version");
    }
    if (ld != cfg_.latent_dim || B != cfg_.branching || hd != cfg_.hidden_dim) {
        throw std::runtime_error("ValueHead::load: config mismatch with file");
    }
    load_tensor(f, cfg_.hidden_dim, in_dim(),        W1_);
    load_tensor(f, cfg_.hidden_dim, 1,               b1_);
    load_tensor(f, 1,               cfg_.hidden_dim, W2_);
    load_tensor(f, 1,               1,               b2_);
    step_n_ = 0;  // optimizer state not persisted; resume = fresh Adam state.
}

}  // namespace brodiffusion::value_head
