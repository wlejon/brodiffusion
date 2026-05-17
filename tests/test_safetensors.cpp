#include "brodiffusion/safetensors.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace st = brodiffusion::safetensors;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

// Build a minimal safetensors file with two tensors:
//   "alpha" : F32, shape [2,3], 24 bytes
//   "beta"  : F16, shape [4],   8 bytes
static std::filesystem::path write_fixture() {
    auto path = std::filesystem::temp_directory_path() / "brodiffusion_test.safetensors";

    const std::string header =
        "{\"__metadata__\":{\"framework\":\"test\"},"
        "\"alpha\":{\"dtype\":\"F32\",\"shape\":[2,3],\"data_offsets\":[0,24]},"
        "\"beta\":{\"dtype\":\"F16\",\"shape\":[4],\"data_offsets\":[24,32]}}";

    std::vector<uint8_t> payload(32, 0);
    // alpha: 6 floats 1.0..6.0
    for (int i = 0; i < 6; ++i) {
        float v = static_cast<float>(i + 1);
        std::memcpy(payload.data() + i * 4, &v, 4);
    }
    // beta: 4 fp16 bit patterns (1.0, 2.0, 3.0, 4.0)
    uint16_t halves[4] = {0x3c00, 0x4000, 0x4200, 0x4400};
    std::memcpy(payload.data() + 24, halves, 8);

    uint64_t hdr_size = header.size();
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("cannot create fixture file");
    f.write(reinterpret_cast<const char*>(&hdr_size), 8);
    f.write(header.data(), header.size());
    f.write(reinterpret_cast<const char*>(payload.data()), payload.size());
    return path;
}

int main() {
    auto path = write_fixture();

    // Scope the File so its mmap is released before we try to delete the
    // backing file. Windows refuses to remove a file with an open mapping.
    {
        auto file = st::File::open(path.string());
        CHECK(file.size() == 2);

        const auto* alpha = file.find("alpha");
        CHECK(alpha != nullptr);
        if (alpha) {
            CHECK(alpha->dtype == st::Dtype::F32);
            CHECK(alpha->shape.size() == 2);
            CHECK(alpha->shape[0] == 2 && alpha->shape[1] == 3);
            CHECK(alpha->nbytes == 24);
            const float* fp = reinterpret_cast<const float*>(alpha->data);
            for (int i = 0; i < 6; ++i) {
                CHECK(fp[i] == static_cast<float>(i + 1));
            }
        }

        const auto* beta = file.find("beta");
        CHECK(beta != nullptr);
        if (beta) {
            CHECK(beta->dtype == st::Dtype::F16);
            CHECK(beta->shape.size() == 1 && beta->shape[0] == 4);
            CHECK(beta->nbytes == 8);
            const uint16_t* hp = reinterpret_cast<const uint16_t*>(beta->data);
            CHECK(hp[0] == 0x3c00);
            CHECK(hp[3] == 0x4400);
        }

        CHECK(file.find("nope") == nullptr);
        bool threw = false;
        try { (void)file.get("nope"); } catch (const std::runtime_error&) { threw = true; }
        CHECK(threw);
    }

    // Error path: garbage file should throw.
    auto bad_path = std::filesystem::temp_directory_path() / "brodiffusion_bad.safetensors";
    {
        std::ofstream bf(bad_path, std::ios::binary | std::ios::trunc);
        bf << "not a safetensors file";
    }
    bool bad_threw = false;
    try { (void)st::File::open(bad_path.string()); }
    catch (const std::runtime_error&) { bad_threw = true; }
    CHECK(bad_threw);

    // Use the error_code overloads; the bad fixture has no mapping but the
    // good one's mapping is already released by the scope above.
    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::filesystem::remove(bad_path, ec);

    if (g_failures == 0) std::printf("safetensors: OK\n");
    else std::fprintf(stderr, "safetensors: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
