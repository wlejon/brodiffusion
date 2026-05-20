cmake_minimum_required(VERSION 3.24)
project(brodiffusion LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/MP>
        $<$<COMPILE_LANGUAGE:CXX>:/permissive->
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
    )
    if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
endif()

# --- GPU backend selection ---
#
# brotensor always builds its CPU backend; CUDA and Metal are additive and
# mutually exclusive. brodiffusion forwards whichever the user passed on the
# cmake line. With neither set, brodiffusion builds and runs CPU-only in
# FP32: brotensor's CPU backend implements the full diffusion op set, and
# brodiffusion's own SD1.5 fused kernels fall back to brotensor-op
# compositions (src/fused.cpp). With -DBROTENSOR_WITH_CUDA=ON the SD1.5-tuned
# fused CUDA kernels compile in additively and are selected at runtime for
# GPU-resident tensors — CPU by default, GPU when available.
option(BROTENSOR_WITH_CUDA  "Build the CUDA GPU backend"  OFF)
option(BROTENSOR_WITH_METAL "Build the Metal GPU backend" OFF)

# brodiffusion ships SD1.5-specific fused CUDA kernels (e.g. fused_resblock.cu);
# enable the CUDA language only when CUDA is the selected backend.
if(BROTENSOR_WITH_CUDA)
    # CMAKE_CUDA_ARCHITECTURES MUST be set before enable_language(CUDA) —
    # enable_language initializes it to nvcc's default (52 on Windows) if
    # unset, after which the (NOT DEFINED) guard would silently skip the
    # `native` override and we'd end up targeting Maxwell.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# brodiffusion ships SD1.5-specific fused Metal kernels (fused_resblock.mm,
# fused_transformer.mm); enable Objective-C++ only when Metal is selected.
if(BROTENSOR_WITH_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "BROTENSOR_WITH_METAL requires an Apple platform.")
    endif()
    enable_language(OBJCXX)
endif()

# --- Siblings: bromath (header-only) + brotensor (GPU ops) ---
#
# Prefer standalone repos at ../<name>; fall back to third_party/ submodules.
# Matches the pattern documented in bro/docs/multi-repo-workflow.md.

set(BROMATH_DIR "${CMAKE_SOURCE_DIR}/../bromath" CACHE PATH
    "Path to standalone bromath repo (set to a missing path to force submodule fallback)")
if(NOT TARGET bromath)
    if(EXISTS "${BROMATH_DIR}/CMakeLists.txt")
        add_subdirectory("${BROMATH_DIR}" "${CMAKE_BINARY_DIR}/bromath" EXCLUDE_FROM_ALL)
    elseif(EXISTS "${CMAKE_SOURCE_DIR}/third_party/bromath/CMakeLists.txt")
        add_subdirectory(third_party/bromath EXCLUDE_FROM_ALL)
    else()
        message(FATAL_ERROR
            "bromath not found at ${BROMATH_DIR} or third_party/bromath. "
            "Clone github.com/wlejon/bromath next to this repo, or override with -DBROMATH_DIR=...")
    endif()
endif()

set(BROTENSOR_DIR "${CMAKE_SOURCE_DIR}/../brotensor" CACHE PATH
    "Path to standalone brotensor repo (set to a missing path to force submodule fallback)")
if(NOT TARGET brotensor)
    if(EXISTS "${BROTENSOR_DIR}/CMakeLists.txt")
        add_subdirectory("${BROTENSOR_DIR}" "${CMAKE_BINARY_DIR}/brotensor" EXCLUDE_FROM_ALL)
    elseif(EXISTS "${CMAKE_SOURCE_DIR}/third_party/brotensor/CMakeLists.txt")
        add_subdirectory(third_party/brotensor EXCLUDE_FROM_ALL)
    else()
        message(FATAL_ERROR
            "brotensor not found at ${BROTENSOR_DIR} or third_party/brotensor. "
            "Clone the brotensor repo next to this repo, or override with -DBROTENSOR_DIR=...")
    endif()
endif()

# --- brodiffusion library ---

add_library(brodiffusion STATIC)
add_library(brodiffusion::brodiffusion ALIAS brodiffusion)

target_include_directories(brodiffusion PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(brodiffusion PUBLIC cxx_std_20)
target_link_libraries(brodiffusion PUBLIC
    bromath::bromath
    brotensor::brotensor
)

if(MSVC)
    target_compile_options(brodiffusion PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
    target_compile_options(brodiffusion PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
endif()

target_sources(brodiffusion PRIVATE
    src/version.cpp
    src/tokenizer_clip.cpp
    src/clip.cpp
    src/vae.cpp
    src/unet.cpp
    src/scheduler.cpp
    src/lcm_scheduler.cpp
    src/lora.cpp
    src/pipeline.cpp
    src/clip_image.cpp
    src/clip_score.cpp
    src/fused.cpp
    src/optim.cpp
)

# SD1.5-tuned fused GPU kernels. fused.cpp dispatches a GPU-resident tensor to
# the CUDA or Metal kernel for the active backend and falls back to a
# brotensor-op composition on CPU. The optimizer (src/optim.cpp, always built)
# is backend-agnostic — it is composed from device-polymorphic brotensor ops.
if(BROTENSOR_WITH_CUDA)
    target_sources(brodiffusion PRIVATE
        src/fused_resblock.cu
        src/fused_transformer.cu
    )
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(brodiffusion PUBLIC CUDA::cudart)
endif()

if(BROTENSOR_WITH_METAL)
    # Metal mirrors of the fused CUDA kernels — Objective-C++ (.mm) compiled
    # with ARC. They reach brotensor's Metal device + buffer pool through the
    # public <brotensor/metal_interop.h> surface.
    target_sources(brodiffusion PRIVATE
        src/fused_resblock.mm
        src/fused_transformer.mm
    )
    set_source_files_properties(
        src/fused_resblock.mm
        src/fused_transformer.mm
        PROPERTIES
            COMPILE_FLAGS "-fobjc-arc"
            LANGUAGE OBJCXX
    )
    target_link_libraries(brodiffusion PUBLIC
        "-framework Foundation"
        "-framework Metal"
    )
endif()

# --- CLI ---
option(BRODIFFUSION_CLI "Build the brodiffusion CLI" ${PROJECT_IS_TOP_LEVEL})
if(BRODIFFUSION_CLI)
    add_executable(brodiffusion_cli cli/main.cpp)
    target_link_libraries(brodiffusion_cli PRIVATE brodiffusion)
    set_target_properties(brodiffusion_cli PROPERTIES OUTPUT_NAME brodiffusion)
endif()

# --- Install / export ---
#
# Off by default: brodiffusion is consumed as a sibling repo via the same
# add_subdirectory pattern bro uses for its other libraries, not via
# find_package. The export logic below also can't currently coexist with the
# EXCLUDE_FROM_ALL'd bromath/brotensor (they aren't in our export set).
option(BRODIFFUSION_INSTALL "Generate install targets for brodiffusion" OFF)
if(BRODIFFUSION_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    install(TARGETS brodiffusion
        EXPORT brodiffusionTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
    install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

    install(EXPORT brodiffusionTargets
        FILE brodiffusionTargets.cmake
        NAMESPACE brodiffusion::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/brodiffusion
    )

    set(_cfg_in "${CMAKE_CURRENT_BINARY_DIR}/brodiffusionConfig.cmake.in")
    file(WRITE "${_cfg_in}"
        "@PACKAGE_INIT@\n"
        "include(\"\${CMAKE_CURRENT_LIST_DIR}/brodiffusionTargets.cmake\")\n"
    )
    configure_package_config_file("${_cfg_in}"
        "${CMAKE_CURRENT_BINARY_DIR}/brodiffusionConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/brodiffusion
    )
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/brodiffusionConfig.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/brodiffusion
    )
endif()

# --- Tests ---
option(BRODIFFUSION_TESTS "Build brodiffusion tests" ON)
if(BRODIFFUSION_TESTS AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    enable_testing()
    add_subdirectory(tests)
endif()
