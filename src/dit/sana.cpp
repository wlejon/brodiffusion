// SanaTransformer2DModel denoiser — stub.
//
// See dit/sana.h. Config, lifetime, and the trivial Denoiser overrides only;
// the transformer math lands in a later chunk.

#include "brodiffusion/dit/sana.h"

#include "brodiffusion/dit/common.h"

#include <stdexcept>

namespace brodiffusion::dit {

SanaDenoiser::SanaDenoiser(const SanaConfig& cfg) : cfg_(cfg) {}

SanaDenoiser::~SanaDenoiser() = default;

brotensor::Dtype SanaDenoiser::compute_dtype() const {
    // Same internal arithmetic dtype as Flux (BF16 on CUDA, FP32 on CPU).
    return flux_compute_dtype();
}

void SanaDenoiser::load_weights(const brotensor::safetensors::File& f,
                                const std::string& prefix) {
    (void)f;
    (void)prefix;
    throw std::runtime_error(
        "dit::SanaDenoiser: not implemented yet (chunk C)");
}

void SanaDenoiser::finalize_weights() {
    throw std::runtime_error(
        "dit::SanaDenoiser: not implemented yet (chunk C)");
}

PreparedConditioning SanaDenoiser::prepare(const Conditioning& cond) {
    (void)cond;
    throw std::runtime_error(
        "dit::SanaDenoiser: not implemented yet (chunk C)");
}

void SanaDenoiser::forward(const brotensor::Tensor& latent, int H_lat, int W_lat,
                           float timestep, const PreparedConditioning& prepared,
                           Branch branch, brotensor::Tensor& out) {
    (void)latent;
    (void)H_lat;
    (void)W_lat;
    (void)timestep;
    (void)prepared;
    (void)branch;
    (void)out;
    throw std::runtime_error(
        "dit::SanaDenoiser: not implemented yet (chunk C)");
}

}  // namespace brodiffusion::dit
