cmake_minimum_required(VERSION 3.24)
project(brodiffusion LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# This project ships a shared library (krea2_capi) that whole-archives
# libbrodiffusion.a and the vendored sibling static libs. On Linux/ELF, a
# static archive linked into a shared object must be position-independent, or
# ld rejects its R_X86_64_PC32/TPOFF32 relocations ("recompile with -fPIC").
# Set this before the add_subdirectory() calls below so it propagates into
# every vendored sibling's scope (bromath / brotensor / brolm / broimage) as
# well as this project's own targets.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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

# --- brolm (text encoders + tokenizers — brodiffusion's text frontend) ---
#
# Prefer the standalone repo at ../brolm; fall back to third_party/brolm.
# brolm pulls in bromath + brotensor guarded by if(NOT TARGET ...), so it
# reuses the targets added above.
set(BROLM_DIR "${CMAKE_SOURCE_DIR}/../brolm" CACHE PATH
    "Path to standalone brolm repo (set to a missing path to force submodule fallback)")
if(NOT TARGET brolm)
    if(EXISTS "${BROLM_DIR}/CMakeLists.txt")
        add_subdirectory("${BROLM_DIR}" "${CMAKE_BINARY_DIR}/brolm" EXCLUDE_FROM_ALL)
    elseif(EXISTS "${CMAKE_SOURCE_DIR}/third_party/brolm/CMakeLists.txt")
        add_subdirectory(third_party/brolm EXCLUDE_FROM_ALL)
    else()
        message(FATAL_ERROR
            "brolm not found at ${BROLM_DIR} or third_party/brolm. "
            "Clone the brolm repo next to this repo, or override with -DBROLM_DIR=...")
    endif()
endif()

# --- broimage (image decode/encode + host preproc) ---
set(BROIMAGE_DIR "${CMAKE_SOURCE_DIR}/../broimage" CACHE PATH
    "Path to standalone broimage repo (set to a missing path to force submodule fallback)")
if(NOT TARGET broimage)
    if(EXISTS "${BROIMAGE_DIR}/CMakeLists.txt")
        add_subdirectory("${BROIMAGE_DIR}" "${CMAKE_BINARY_DIR}/broimage" EXCLUDE_FROM_ALL)
    elseif(EXISTS "${CMAKE_SOURCE_DIR}/third_party/broimage/CMakeLists.txt")
        add_subdirectory(third_party/broimage EXCLUDE_FROM_ALL)
    else()
        message(FATAL_ERROR
            "broimage not found at ${BROIMAGE_DIR} or third_party/broimage. "
            "Clone the broimage repo next to this repo, or override with -DBROIMAGE_DIR=...")
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
# FastNoiseLite is vendored as a single header and used only by the terrain
# synthetic map. PRIVATE: it stays out of our public headers (SyntheticMap hides
# it behind a pimpl), so consumers do not inherit 100 KB of it.
target_include_directories(brodiffusion PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/fastnoiselite)
target_link_libraries(brodiffusion PUBLIC
    bromath::bromath
    brotensor::brotensor
    brolm::brolm
    broimage::broimage
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
    src/vae.cpp
    src/vae_dcae.cpp
    src/vae_qwenimage.cpp
    src/unet.cpp
    src/unet_blocks.cpp
    src/controlnet.cpp
    src/scheduler.cpp
    src/lcm_scheduler.cpp
    src/flow_match_scheduler.cpp
    src/scm_scheduler.cpp
    src/dpm_solver.cpp
    src/model_config.cpp
    src/sana_text.cpp
    src/krea2_text.cpp
    src/cond_control.cpp
    src/detail/safetensors_dir.cpp
    src/lora.cpp
    src/pipeline.cpp
    src/image_io.cpp
    src/fused.cpp
    src/optim.cpp
    src/dit/common.cpp
    src/dit/flux.cpp
    src/dit/krea2.cpp
    src/dit/sana.cpp
    src/dit/pixart.cpp
    src/ardy/motion_rep.cpp
    src/ardy/fsq_decoder.cpp
    src/ardy/denoiser_backbone.cpp
    src/ardy/denoiser.cpp
    src/ardy/sampler.cpp
    src/ardy/text_conditioner.cpp
    src/terrain/mp_unet.cpp
    src/terrain/sampler.cpp
    src/terrain/synthetic_map.cpp
    src/terrain/infinite_tensor.cpp
    src/terrain/world_pipeline.cpp
    src/triposplat/vae_encoder.cpp
    src/triposplat/flow_model.cpp
    src/triposplat/flow_sobol_pe.cpp
    src/triposplat/octree_decoder.cpp
    src/triposplat/sampler.cpp
)

# SD1.5-tuned fused GPU kernels. fused.cpp dispatches a GPU-resident tensor to
# the CUDA or Metal kernel for the active backend and falls back to a
# brotensor-op composition on CPU. The optimizer (src/optim.cpp, always built)
# is backend-agnostic — it is composed from device-polymorphic brotensor ops.
if(BROTENSOR_WITH_CUDA)
    target_sources(brodiffusion PRIVATE
        src/fused_resblock.cu
        src/fused_transformer.cu
        src/triposplat/flow_rope.cu
    )
    find_package(CUDAToolkit REQUIRED)
    # Static CUDA runtime so the shipped binary carries no cudart DLL dependency
    # and launches on machines without the CUDA toolkit (falls back to CPU).
    target_link_libraries(brodiffusion PUBLIC CUDA::cudart_static)
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
    # OUTPUT_NAME collides with the static lib's basename: on MSVC, an
    # executable that exports any symbol also emits an import library, and
    # `brodiffusion.exe` would emit `brodiffusion.lib` straight over the
    # `brodiffusion.lib` it is linking against (LNK1149). Only bites on a full
    # relink — an incremental build that skips the exe link hides it. Give the
    # exe's import lib its own basename; the .exe name is unaffected.
    set_target_properties(brodiffusion_cli PROPERTIES
        OUTPUT_NAME brodiffusion
        ARCHIVE_OUTPUT_NAME brodiffusion_cli)

    # DiT-only nondeterminism harness (no T5/VAE) — compute-sanitizer target.
    add_executable(dit_race tools/dit_race.cpp)
    target_link_libraries(dit_race PRIVATE brodiffusion)

    # Krea 2 kernel-shape microbenchmark (GEMM / flash-attention attribution).
    add_executable(bench_krea2_kernels tools/bench_krea2_kernels.cpp)
    target_link_libraries(bench_krea2_kernels PRIVATE brodiffusion)
endif()

# --- Krea 2 research C API (krea2_capi.dll, consumed via ctypes) ---
option(BRODIFFUSION_KREA2_CAPI "Build the Krea 2 research C API DLL" ${PROJECT_IS_TOP_LEVEL})
if(BRODIFFUSION_KREA2_CAPI)
    add_library(krea2_capi SHARED src/capi/krea2_capi.cpp)
    target_link_libraries(krea2_capi PRIVATE brodiffusion)
    target_compile_definitions(krea2_capi PRIVATE K2_CAPI_BUILD)
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
