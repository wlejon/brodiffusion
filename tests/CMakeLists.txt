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

# Qwen-Image VAE (Krea 2 decoder) test: an unconditional synthetic-weights
# smoke test plus a real-weights numerical smoke test gated on
# weights/krea-2-raw/vae (skips cleanly when absent).
brodiffusion_add_test(brodiffusion_test_krea2_vae test_krea2_vae.cpp)
target_compile_definitions(brodiffusion_test_krea2_vae PRIVATE
    BRODIFFUSION_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/weights")

# Krea 2 text conditioning (prompt -> tapped Qwen3-VL hidden states + mask).
# Template-invariant checks always run; the end-to-end path is gated on the
# real Qwen3-VL-4B text encoder + tokenizer under weights/krea-2-raw.
brodiffusion_add_test(brodiffusion_test_krea2_text test_krea2_text.cpp)
target_compile_definitions(brodiffusion_test_krea2_text PRIVATE
    BRODIFFUSION_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/weights")

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
