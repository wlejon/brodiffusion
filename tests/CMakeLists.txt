function(brodiffusion_add_test name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE brodiffusion)
    if(MSVC)
        target_compile_options(${name} PRIVATE /W4)
    else()
        target_compile_options(${name} PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME ${name} COMMAND ${name})
endfunction()

brodiffusion_add_test(brodiffusion_test_smoke          test_smoke.cpp)
brodiffusion_add_test(brodiffusion_test_vae            test_vae.cpp)
brodiffusion_add_test(brodiffusion_test_vae_encoder    test_vae_encoder.cpp)
brodiffusion_add_test(brodiffusion_test_image_io       test_image_io.cpp)
brodiffusion_add_test(brodiffusion_test_unet           test_unet.cpp)
brodiffusion_add_test(brodiffusion_test_unet_residuals test_unet_residuals.cpp)
brodiffusion_add_test(brodiffusion_test_controlnet     test_controlnet.cpp)
brodiffusion_add_test(brodiffusion_test_controlnet_pipeline test_controlnet_pipeline.cpp)
brodiffusion_add_test(brodiffusion_test_scheduler      test_scheduler.cpp)
brodiffusion_add_test(brodiffusion_test_torch_rng      test_torch_rng.cpp)
brodiffusion_add_test(brodiffusion_test_lcm_scheduler  test_lcm_scheduler.cpp)
brodiffusion_add_test(brodiffusion_test_flow_match_scheduler test_flow_match_scheduler.cpp)
brodiffusion_add_test(brodiffusion_test_scm_scheduler  test_scm_scheduler.cpp)
brodiffusion_add_test(brodiffusion_test_lora           test_lora.cpp)
brodiffusion_add_test(brodiffusion_test_pipeline       test_pipeline.cpp)
brodiffusion_add_test(brodiffusion_test_quantize_unet  test_quantize_unet.cpp)
brodiffusion_add_test(brodiffusion_test_pipeline_state test_pipeline_state.cpp)
brodiffusion_add_test(brodiffusion_test_cond_control   test_cond_control.cpp)
brodiffusion_add_test(brodiffusion_test_control_schedule test_control_schedule.cpp)
brodiffusion_add_test(brodiffusion_test_img2img        test_img2img.cpp)
brodiffusion_add_test(brodiffusion_test_step_graph     test_step_graph.cpp)
brodiffusion_add_test(brodiffusion_test_inpaint        test_inpaint.cpp)
brodiffusion_add_test(brodiffusion_test_fused_parity   test_fused_parity.cpp)
brodiffusion_add_test(brodiffusion_test_device_parity  test_device_parity.cpp)
brodiffusion_add_test(brodiffusion_test_model_config   test_model_config.cpp)
brodiffusion_add_test(brodiffusion_test_sana_config    test_sana_config.cpp)
brodiffusion_add_test(brodiffusion_test_flux_blocks   test_flux_blocks.cpp)
brodiffusion_add_test(brodiffusion_test_flux_pipeline test_flux_pipeline.cpp)

# DC-AE (AutoencoderDC f32c32) Sana decoder smoke test. Gated entirely on the
# real Sana VAE checkpoint under weights/sana-600m/vae (skips cleanly when
# absent — there is no compact synthetic fixture for this decoder).
brodiffusion_add_test(brodiffusion_test_dcae_decoder test_dcae_decoder.cpp)
target_compile_definitions(brodiffusion_test_dcae_decoder PRIVATE
    BRODIFFUSION_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/weights")

# Terrain world determinism: the same GPU read twice is bit-identical, gated on
# weights/terrain-diffusion-30m-bro (skips cleanly when absent).
brodiffusion_add_test(brodiffusion_test_terrain_determinism test_terrain_determinism.cpp)
target_compile_definitions(brodiffusion_test_terrain_determinism PRIVATE
    BRODIFFUSION_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/weights")

# Qwen-Image VAE (Krea 2 decoder) test: an unconditional synthetic-weights
# smoke test plus a real-weights numerical smoke test gated on
# weights/krea-2-raw/vae (skips cleanly when absent).
brodiffusion_add_test(brodiffusion_test_krea2_vae test_krea2_vae.cpp)
target_compile_definitions(brodiffusion_test_krea2_vae PRIVATE
    BRODIFFUSION_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/weights")

# Qwen-Image 2.1 VAE: synthetic-weights smoke (every residual-shortcut
# reduction) plus a real-weights encode->decode round trip gated on
# weights/qwen-image-2.1/vae (skips cleanly when absent).
brodiffusion_add_test(brodiffusion_test_qwenimage21_vae test_qwenimage21_vae.cpp)
target_compile_definitions(brodiffusion_test_qwenimage21_vae PRIVATE
    BRODIFFUSION_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/weights")

# Krea 2 text conditioning (prompt -> tapped Qwen3-VL hidden states + mask).
# Template-invariant checks always run; the end-to-end path is gated on the
# real Qwen3-VL-4B text encoder + tokenizer under weights/krea-2-raw.
brodiffusion_add_test(brodiffusion_test_krea2_text test_krea2_text.cpp)
target_compile_definitions(brodiffusion_test_krea2_text PRIVATE
    BRODIFFUSION_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/weights")

# Qwen-Image 2.1 text conditioning (prompt -> Qwen3-VL-8B last hidden states).
# Template-invariant checks always run; the tokenizer and end-to-end parts are
# gated on weights/qwen-image-2.1 (processor/ + text_encoder/), and the model
# is loaded INT8 so it fits alongside anything else on the card. Numerical
# parity is scripts/qwenimage21_text_parity.sh.
brodiffusion_add_test(brodiffusion_test_qwenimage21_text test_qwenimage21_text.cpp)
target_compile_definitions(brodiffusion_test_qwenimage21_text PRIVATE
    BRODIFFUSION_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/weights")

# Krea 2 image DiT (Krea2Transformer2DModel). Synthetic smoke always runs; the
# real 12.9B forward is opt-in (BRODIFFUSION_KREA2_DIT_REAL=1) since it exceeds a
# 24 GB card in BF16. Numerical parity is scripts/krea2_dit_parity.sh.
brodiffusion_add_test(brodiffusion_test_krea2_dit test_krea2_dit.cpp)
target_compile_definitions(brodiffusion_test_krea2_dit PRIVATE
    BRODIFFUSION_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/weights")

# Qwen-Image 2.1 image DiT (QwenImage21Transformer2DModel) + its Denoiser.
# Synthetic smoke (including the prefix KV cache's "cached step == full
# prefill" contract) always runs; the real 7.1B forward is opt-in
# (BRODIFFUSION_QI21_DIT_REAL=1) and loads INT8 so it fits a 24 GB card.
# Numerical parity is scripts/qwenimage21_dit_parity.sh.
brodiffusion_add_test(brodiffusion_test_qwenimage21_dit test_qwenimage21_dit.cpp)
target_compile_definitions(brodiffusion_test_qwenimage21_dit PRIVATE
    BRODIFFUSION_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/weights")

# The multi-slot binding lists and the idempotent prefix-KV dial, over the
# same synthetic checkpoint (qwenimage21_fixture.h). Pure CPU/GPU maths, no
# weights needed, so it always runs.
brodiffusion_add_test(brodiffusion_test_qwenimage21_slots
                      test_qwenimage21_slots.cpp)

# Sana Linear DiT (SanaTransformer2DModel) denoiser smoke test. Gated entirely
# on the real Sana 0.6B transformer checkpoint under weights/sana-600m/transformer
# (skips cleanly when absent — no compact synthetic fixture for this DiT).
brodiffusion_add_test(brodiffusion_test_sana_dit test_sana_dit.cpp)
target_compile_definitions(brodiffusion_test_sana_dit PRIVATE
    BRODIFFUSION_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/weights")

# Sana Gemma-2 text encoder (prompt -> last_hidden_state). Gated entirely on the
# real Sana 0.6B text encoder + tokenizer under weights/sana-600m (skips cleanly
# when absent). Also exercises the JSON -Infinity fix via load_model_config.
brodiffusion_add_test(brodiffusion_test_sana_text test_sana_text.cpp)
target_compile_definitions(brodiffusion_test_sana_text PRIVATE
    BRODIFFUSION_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/weights")

# Sana end-to-end Pipeline (txt2img) integration test. Gated entirely on the
# real Sana 0.6B model directory under weights/sana-600m (skips cleanly when
# absent — no compact synthetic fixture for the DiT / DC-AE / Gemma stack).
brodiffusion_add_test(brodiffusion_test_sana_pipeline test_sana_pipeline.cpp)
target_compile_definitions(brodiffusion_test_sana_pipeline PRIVATE
    BRODIFFUSION_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/weights")

# TripoSplat Flux.2 VAE image-encoder parity test: structural checks always run;
# the numeric-parity part is gated on the real checkpoint
# (weights/triposplat/vae/flux2-vae.safetensors) and golden_flux2vae.bin under
# weights/triposplat/vae/golden (generated out-of-repo, never committed). Skips
# cleanly when either is absent.
brodiffusion_add_test(brodiffusion_test_flux2_vae_encoder test_flux2_vae_encoder.cpp)
target_compile_definitions(brodiffusion_test_flux2_vae_encoder PRIVATE
    BRODIFFUSION_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/weights")

# TripoSplat flow DiT (LatentSeqMMFlowModel). Same gating: real flow checkpoint
# + golden_flow.bin under weights/triposplat/diffusion_models (out-of-repo).
brodiffusion_add_test(brodiffusion_test_flow_model test_flow_model.cpp)
target_compile_definitions(brodiffusion_test_flow_model PRIVATE
    BRODIFFUSION_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/weights")

# TripoSplat OctreeGaussianDecoder (octree occupancy predictor + elastic GS
# head). Same gating: real decoder checkpoint + golden_decoder.bin under
# weights/triposplat/vae (out-of-repo).
brodiffusion_add_test(brodiffusion_test_octree_decoder test_octree_decoder.cpp)
target_compile_definitions(brodiffusion_test_octree_decoder PRIVATE
    BRODIFFUSION_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/weights")

# TripoSplat flow Euler CFG sampler (sample_latent). Gated on the flow
# checkpoint + golden_sampler.bin under weights/triposplat/diffusion_models.
brodiffusion_add_test(brodiffusion_test_flow_sampler test_flow_sampler.cpp)
target_compile_definitions(brodiffusion_test_flow_sampler PRIVATE
    BRODIFFUSION_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/weights")

# Bronze standalone JavaScript API test
if(BRODIFFUSION_ENABLE_API)
    add_executable(brodiffusion_test_api test_diffusion_api.cpp test_diffusion_surface.cpp
        test_diffusion_api_weights.cpp)
    # The weights-gated half looks under <repo>/weights/sd15 and skips without it.
    target_compile_definitions(brodiffusion_test_api PRIVATE
        BRODIFFUSION_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/weights")
    target_include_directories(brodiffusion_test_api PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/api
    )
    target_link_libraries(brodiffusion_test_api PRIVATE
        brodiffusion_api
        bronze_runtime_shared
    )
    # bronze's shared runtime is built into BRONZE_SHARED_RUNTIME_DIR, not
    # beside this executable, and the PE loader only looks beside the .exe
    # (0xc0000135 + a modal "dll was not found" otherwise). Stage it next to
    # the test as a post-build step and put both dirs on ctest's PATH.
    if(WIN32)
        add_custom_command(TARGET brodiffusion_test_api POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:bronze_runtime_shared>
                $<TARGET_FILE_DIR:brodiffusion_test_api>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_LINKER_FILE:bronze_runtime_shared>
                $<TARGET_FILE_DIR:brodiffusion_test_api>
        )
    endif()
    # It runs JavaScript, which brass compiles to machine code for the target;
    # on one brass has no code generator for (BRASS_HOST_BACKEND, decided in
    # brass's CMakeLists; x86_64 and AArch64 have one) that code cannot run. The
    # executable still builds there, which is the check that the embed links;
    # the test is not registered rather than registered to bus-error.
    if(NOT BRASS_HOST_BACKEND)
        message(STATUS "brodiffusion: brodiffusion_test_api not registered (brass has no code generator for this target)")
    else()
        add_test(NAME brodiffusion_test_api COMMAND brodiffusion_test_api)
        # The same binding test under bronze's GC stress oracle: a collection
        # on every allocation, from-space poisoned afterwards, so a Value held
        # across an allocating embed call crashes or reads garbage here
        # instead of passing by luck. `ctest -LE gcstress` skips it.
        add_test(NAME brodiffusion_test_api_gcstress COMMAND brodiffusion_test_api)
        set_tests_properties(brodiffusion_test_api_gcstress PROPERTIES LABELS gcstress)
        if(WIN32)
            set_tests_properties(brodiffusion_test_api PROPERTIES
                ENVIRONMENT "PATH=$<TARGET_FILE_DIR:bronze_runtime_shared>;$<TARGET_FILE_DIR:brodiffusion_test_api>;$ENV{PATH}")
            set_tests_properties(brodiffusion_test_api_gcstress PROPERTIES
                ENVIRONMENT "BRONZE_GC_STRESS=1;BRONZE_GC_POISON=1;PATH=$<TARGET_FILE_DIR:bronze_runtime_shared>;$<TARGET_FILE_DIR:brodiffusion_test_api>;$ENV{PATH}")
        else()
            set_tests_properties(brodiffusion_test_api_gcstress PROPERTIES
                ENVIRONMENT "BRONZE_GC_STRESS=1;BRONZE_GC_POISON=1")
        endif()
    endif()
endif()

