// Shared CLI helpers — argv scanning, PNG writing, raw-f32 latent IO.
// Declared in commands.h; used by every subcommand translation unit.

#include "commands.h"

#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include "broimage/encode.h"
#include "broimage/preproc.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace brodiffusion::cli {

const char* arg_after(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc - 1; ++i) {
        if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
    }
    return nullptr;
}

std::vector<LoraSpec> collect_loras(int argc, char** argv) {
    std::vector<LoraSpec> out;
    for (int i = 1; i < argc - 1; ++i) {
        if (std::strcmp(argv[i], "--lora") != 0) continue;
        std::string raw = argv[i + 1];
        // Split on the LAST ':' so Windows drive letters ("D:\foo:0.8") parse
        // correctly — the path may contain colons. A trailing
        // ":<number>" is the scale; anything else is part of the path.
        LoraSpec s;
        auto pos = raw.rfind(':');
        bool has_scale = false;
        if (pos != std::string::npos && pos != 0 && pos > 1) {
            const std::string tail = raw.substr(pos + 1);
            // Heuristic: a scale is a parseable float of strlen() > 0.
            char* endp = nullptr;
            float v = std::strtof(tail.c_str(), &endp);
            if (endp && *endp == '\0' && !tail.empty()) {
                s.scale = v;
                s.path  = raw.substr(0, pos);
                has_scale = true;
            }
        }
        if (!has_scale) s.path = raw;
        out.push_back(std::move(s));
    }
    return out;
}

std::vector<ControlSpec> collect_controls(int argc, char** argv,
                                          bool& usage_error) {
    std::vector<ControlSpec> out;
    usage_error = false;
    for (int i = 1; i < argc - 1; ++i) {
        const char* a = argv[i];
        const char* v = argv[i + 1];
        if (std::strcmp(a, "--control") == 0) {
            ControlSpec s;
            s.weights_path = v;
            out.push_back(std::move(s));
        } else if (std::strcmp(a, "--control-image") == 0) {
            if (out.empty()) {
                std::fprintf(stderr,
                    "controlnet: --control-image must follow --control\n");
                usage_error = true;
                return {};
            }
            out.back().image_path = v;
        } else if (std::strcmp(a, "--control-scale") == 0) {
            if (out.empty()) {
                std::fprintf(stderr,
                    "controlnet: --control-scale must follow --control\n");
                usage_error = true;
                return {};
            }
            out.back().scale = static_cast<float>(std::atof(v));
        } else if (std::strcmp(a, "--control-window") == 0) {
            if (out.empty()) {
                std::fprintf(stderr,
                    "controlnet: --control-window must follow --control\n");
                usage_error = true;
                return {};
            }
            // Format: "<start>:<end>" (both fractions in [0,1]).
            std::string raw = v;
            auto pos = raw.find(':');
            if (pos == std::string::npos) {
                std::fprintf(stderr,
                    "controlnet: --control-window expects <start>:<end>\n");
                usage_error = true;
                return {};
            }
            out.back().start_step =
                static_cast<float>(std::atof(raw.substr(0, pos).c_str()));
            out.back().end_step =
                static_cast<float>(std::atof(raw.substr(pos + 1).c_str()));
        }
    }
    for (const auto& s : out) {
        if (s.image_path.empty()) {
            std::fprintf(stderr,
                "controlnet: every --control needs a matching "
                "--control-image\n");
            usage_error = true;
            return {};
        }
    }
    return out;
}

// f32_nchw_to_u8_nhwc with scale=127.5, bias=127.5 maps [-1,1] -> [0,255]
// with the same round+clamp the old hand-rolled PPM writer used.
int write_png(const char* out_path, const std::vector<float>& img,
              int W, int H) {
    std::vector<std::uint8_t> rgb(static_cast<std::size_t>(3) * H * W);
    broimage::f32_nchw_to_u8_nhwc(img.data(), 1, 3, H, W,
                                  127.5f, 127.5f, rgb.data());
    if (!broimage::encode_png_file(out_path, rgb.data(), W, H, 3)) {
        std::fprintf(stderr, "txt2img: cannot write PNG %s\n", out_path);
        return 1;
    }
    std::printf("Wrote %s\n", out_path);
    return 0;
}

std::vector<float> load_latent_f32(const char* path, int expected_count) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        throw std::runtime_error(std::string("cannot open --latent-in: ") + path);
    }
    const std::streamsize bytes = f.tellg();
    const std::streamsize want  =
        static_cast<std::streamsize>(expected_count) * 4;
    if (bytes != want) {
        throw std::runtime_error(
            "--latent-in size mismatch: " + std::to_string(bytes) +
            " bytes, expected " + std::to_string(want) + " (" +
            std::to_string(expected_count) + " float32)");
    }
    std::vector<float> v(static_cast<std::size_t>(expected_count));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), want);
    return v;
}

void dump_latent_f32(const char* path, const brotensor::Tensor& t) {
    const std::size_t n = static_cast<std::size_t>(t.rows) * t.cols;
    std::vector<float> vals;
    if (t.dtype == brotensor::Dtype::FP16) {
        std::vector<std::uint16_t> bits(n);
        t.copy_to_host_fp16(bits.data());
        brotensor::sync_all();
        vals.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            vals[i] = brotensor::fp16_bits_to_fp32(bits[i]);
        }
    } else if (t.dtype == brotensor::Dtype::BF16) {
        std::vector<std::uint16_t> bits(n);
        t.copy_to_host_bf16(bits.data());
        brotensor::sync_all();
        vals.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            vals[i] = brotensor::bf16_bits_to_fp32(bits[i]);
        }
    } else {
        vals = t.to_host_vector();
    }
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        throw std::runtime_error(std::string("cannot open --latent-out: ") + path);
    }
    f.write(reinterpret_cast<const char*>(vals.data()),
            static_cast<std::streamsize>(vals.size() * sizeof(float)));
}

}  // namespace brodiffusion::cli
