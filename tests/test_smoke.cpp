#include "brodiffusion/version.h"

#include <cstdio>
#include <cstring>

int main() {
    const char* v = brodiffusion::version_string();
    if (!v || std::strlen(v) == 0) {
        std::fprintf(stderr, "version_string() returned empty\n");
        return 1;
    }
    std::printf("brodiffusion %s\n", v);
    return 0;
}
