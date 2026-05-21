// safetensors_dir — see safetensors_dir.h.

#include "brodiffusion/detail/safetensors_dir.h"

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace brodiffusion::detail {

namespace fs = std::filesystem;
namespace st = ::brotensor::safetensors;

std::vector<st::File> open_component_files(const std::string& component_dir) {
    const fs::path dir(component_dir);
    if (!fs::exists(dir) || !fs::is_directory(dir)) {
        throw std::runtime_error(
            "open_component_files: not a directory: '" + component_dir + "'");
    }

    // Collect every *.safetensors path, sorted by filename so a sharded set
    // (`...-00001-of-0000N`, `...-00002-of-0000N`, ...) loads in order.
    std::vector<fs::path> paths;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() == ".safetensors") {
            paths.push_back(entry.path());
        }
    }
    if (paths.empty()) {
        throw std::runtime_error(
            "open_component_files: no *.safetensors file in '"
            + component_dir + "'");
    }
    std::sort(paths.begin(), paths.end());

    std::vector<st::File> files;
    files.reserve(paths.size());
    for (const auto& p : paths) {
        files.push_back(st::File::open(p.string()));
    }
    return files;
}

}  // namespace brodiffusion::detail
