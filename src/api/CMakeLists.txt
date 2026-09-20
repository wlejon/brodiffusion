add_library(brodiffusion_api STATIC
    api.cpp
    host_class.cpp
    native_diffusion_values.cpp
    native_diffusion_pipeline.cpp
    native_diffusion_control.cpp
    native_diffusion_krea2.cpp
    native_diffusion_state.cpp
    native_diffusion_triposplat.cpp
    native_diffusion_vae.cpp
)

target_include_directories(brodiffusion_api PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
)

target_link_libraries(brodiffusion_api PUBLIC
    brodiffusion
    brotensor
    brovisionml
    bronze_runtime_shared
)

target_compile_features(brodiffusion_api PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(brodiffusion_api PRIVATE /W4)
else()
    target_compile_options(brodiffusion_api PRIVATE -Wall -Wextra)
endif()
