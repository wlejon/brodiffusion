#include "brodiffusion/version.h"

#include <cstdio>
#include <cstring>

static int usage() {
    std::printf(
        "brodiffusion %s\n"
        "\n"
        "Usage:\n"
        "  brodiffusion --version\n"
        "  brodiffusion txt2img --model <safetensors> --prompt <text> --out <ppm>\n"
        "                       [--negative <text>] [--steps N] [--cfg F]\n"
        "                       [--width N] [--height N] [--seed N]\n"
        "\n"
        "--out writes uncompressed PPM (P6) — a dev convenience, not a real\n"
        "image format. The library returns RGB8 host buffers; encoding is the\n"
        "consumer's job (bro's image-api on integration).\n"
        "\n"
        "Pipelines are not implemented yet. See README.md for status.\n",
        brodiffusion::version_string());
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) return usage();
    if (std::strcmp(argv[1], "--version") == 0 || std::strcmp(argv[1], "-v") == 0) {
        std::printf("brodiffusion %s\n", brodiffusion::version_string());
        return 0;
    }
    if (std::strcmp(argv[1], "txt2img") == 0) {
        std::fprintf(stderr, "txt2img: not implemented yet\n");
        return 2;
    }
    return usage();
}
