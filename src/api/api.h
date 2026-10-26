#pragma once

#include "embed/embed.h"

#include <functional>
#include <string>

namespace brodiffusion::api {

// ---- Access ------------------------------------------------------------------
//
// Every native that opens a path the app named asks the host's access gate
// (embed/embed_access.h, bronze::embed::checkAccess) first, on the calling JS
// thread, before anything is opened, stat'ed or queued. The gate is handed the
// path as the library will open it (after the path resolver where the native
// resolves), absolute and lexically normal, in UTF-8.
//   AccessKind::Read:
//     bro.diffusion.loadModel (the model directory, a directory covering what
//     is in it, and opts.textEncoderPath), createPipeline (vocabPath,
//     mergesPath), loadTerrain (the weights directory); Pipeline.loadWeights
//     (each file), applyLora, addControlNet, loadControlDictionary,
//     reloadTextEncoder and qwenImage21ReloadTextEncoder (the model directory
//     and the encoder path); every generate-shaped call (generate,
//     generateAsync, textToImage, imageToImage, inpaint, prime,
//     setIdentityAnchor, krea2PrimeFromTaps, qwenImage21PrimeEdit,
//     qwenImage21PrimeFromText, qwenImage21EncodePromptImages) for each image
//     path its options name (initImagePath, maskImagePath, controls[].imagePath,
//     conditionImages paths); VAE.loadWeights;
//     bro.triposplat.load (dinov3, vae, flow, decoder, birefnet) and
//     TripoSplatPipeline.generate given an image path.
//   AccessKind::Write: bro.triposplat.exportPLY / exportSplat and
//     TripoSplatPipeline.exportPLY / savePly / exportSplat / saveSplat (the file
//     written).
// qwenImage21SavePrefixCache saves to an in-memory slot and touches no file.
//
// A refusal is thrown from the call as an Error carrying the gate's sentence as
// its message, code "EACCES" and `path` (the path as the gate was handed it),
// named "PermissionRevoked" for a grant taken back while the app ran. Nothing
// is read or written.

/// Mounts `bro.diffusion` and `bro.triposplat` onto `bro` in the current Bronze realm.
void installDiffusion();

/// Pump asynchronous diffusion jobs on the current realm/thread.
void tickDiffusionAsync();

/// Cancel and drain all active async diffusion jobs on this thread.
void shutdownDiffusionAsync();

/// Install the host's asset-path resolver. Model dirs, weight files, LoRAs,
/// ControlNets and control dictionaries are then reachable by app-relative and
/// mounted (`/app/...`) paths, as they were before the bronze port — the old
/// binding's setDiffusionAppContext(). Without a resolver every path is used
/// verbatim, which is what a standalone build wants.
void setPathResolver(std::function<std::string(const std::string&)> resolver);

} // namespace brodiffusion::api

using brodiffusion::api::installDiffusion;
