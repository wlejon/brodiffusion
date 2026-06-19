// AutoencoderDC (DC-AE f32c32) decoder — stub.
//
// See vae_dcae.h. Config + lifetime only; the decode math lands in a later
// chunk.

#include "brodiffusion/vae_dcae.h"

#include <stdexcept>

namespace brodiffusion::dcae {

Decoder::Decoder(const DecoderConfig& cfg) : cfg_(cfg) {}

Decoder::~Decoder() = default;

void Decoder::load_weights(const brotensor::safetensors::File& f,
                           const std::string& prefix) {
    (void)f;
    (void)prefix;
    throw std::runtime_error(
        "dcae::Decoder: not implemented yet (chunk B)");
}

void Decoder::decode(const brotensor::Tensor& latent,
                     int H_lat, int W_lat,
                     brotensor::Tensor& out) {
    (void)latent;
    (void)H_lat;
    (void)W_lat;
    (void)out;
    throw std::runtime_error(
        "dcae::Decoder: not implemented yet (chunk B)");
}

}  // namespace brodiffusion::dcae
