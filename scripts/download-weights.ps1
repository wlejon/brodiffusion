# Download model weights for brodiffusion end-to-end testing.
#
# -Model sd15            (default): SD1.5 diffusers components from
#                                   stable-diffusion-v1-5/stable-diffusion-v1-5.
#                                   FP16 component files (the full
#                                   v1-5-pruned-emaonly.safetensors uses the
#                                   original LDM tensor names which the
#                                   loaders don't translate).
# -Model lcm-dreamshaper            LCM-distilled Dreamshaper-7 from
#                                   SimianLuo/LCM_Dreamshaper_v7. The repo
#                                   ships diffusers-format components but
#                                   does NOT ship fp16-suffixed variants of
#                                   every file — the script tries the fp16
#                                   name first and falls back to the fp32
#                                   diffusion_pytorch_model.safetensors. If
#                                   only fp32 is available, our upload helper
#                                   (`upload_fp16_checked`) accepts F32 and
#                                   converts host-side.
# -Model flux-schnell               Flux.1-schnell diffusers components from
#                                   black-forest-labs/FLUX.1-schnell. The
#                                   transformer and the T5-XXL text encoder
#                                   ship SHARDED — the script fetches each
#                                   `*.index.json` weight-map first, extracts
#                                   the shard filenames from it, and downloads
#                                   each shard. NOTE: these weights are large
#                                   (~24 GB transformer + ~10 GB T5).
#
# Authentication: requires `hf auth login` to have been run. The LCM
# Dreamshaper repo is public but rate-limited without auth.
#
# Output: <repo>/weights/<model>/
#   text_encoder/model[.fp16].safetensors
#   unet/diffusion_pytorch_model[.fp16].safetensors
#   vae/diffusion_pytorch_model[.fp16].safetensors
#   tokenizer/vocab.json
#   tokenizer/merges.txt
#
# NOTE: SimianLuo/LCM_Dreamshaper_v7 file names may drift over time. If a
# specific component path errors out, inspect the HF file listing and adjust
# the `$files_lcm_dreamshaper` list below — that's the only edit needed.

[CmdletBinding()]
param(
    [ValidateSet("sd15", "lcm-dreamshaper", "clip-vit-l-14", "flux-schnell", "t5-xxl")]
    [string]$Model  = "sd15",
    [string]$Repo   = "",
    [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"

switch ($Model) {
    "sd15" {
        if (-not $Repo)   { $Repo   = "stable-diffusion-v1-5/stable-diffusion-v1-5" }
        if (-not $OutDir) { $OutDir = "$PSScriptRoot/../weights/sd15" }
        $files = @(
            "text_encoder/model.fp16.safetensors",
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "vae/diffusion_pytorch_model.fp16.safetensors",
            "tokenizer/vocab.json",
            "tokenizer/merges.txt"
        )
        # No fallback list: the SD1.5 mirror is known to ship fp16 variants.
        $fallback = @{}
    }
    "clip-vit-l-14" {
        # OpenAI CLIP ViT-L/14 (the full model — vision + text + both
        # projections). Same architecture's text branch is what SD1.5 uses
        # under the hood; we still load this separately so a scorer doesn't
        # contend with the SD pipeline's mid-generation text-encoder state.
        if (-not $Repo)   { $Repo   = "openai/clip-vit-large-patch14" }
        if (-not $OutDir) { $OutDir = "$PSScriptRoot/../weights/clip-vit-l-14" }
        $files = @(
            "model.safetensors"
        )
        $fallback = @{}
    }
    "lcm-dreamshaper" {
        if (-not $Repo)   { $Repo   = "SimianLuo/LCM_Dreamshaper_v7" }
        if (-not $OutDir) { $OutDir = "$PSScriptRoot/../weights/lcm-dreamshaper" }
        # Try fp16 variants first; the script falls back to the fp32 file
        # name (per-file via $fallback) if the fp16 file does not exist in
        # the repo. SimianLuo/LCM_Dreamshaper_v7 is known to ship fp32 at
        # least for the unet; the loaders accept either.
        $files = @(
            "text_encoder/model.fp16.safetensors",
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "vae/diffusion_pytorch_model.fp16.safetensors",
            "tokenizer/vocab.json",
            "tokenizer/merges.txt"
        )
        $fallback = @{
            "text_encoder/model.fp16.safetensors"          = "text_encoder/model.safetensors"
            "unet/diffusion_pytorch_model.fp16.safetensors" = "unet/diffusion_pytorch_model.safetensors"
            "vae/diffusion_pytorch_model.fp16.safetensors"  = "vae/diffusion_pytorch_model.safetensors"
        }
    }
    "t5-xxl" {
        # Just the T5-XXL text encoder + its tokenizer — the standalone target
        # for working on the T5 encoder without the ~24 GB Flux transformer.
        # ~9.5 GB (FP16).
        #
        # The Flux.1-schnell repo (which ships T5-XXL as `text_encoder_2`) is
        # now gated, so the weights come from comfyanonymous/flux_text_encoders
        # instead — the standard community T5-XXL for Flux: a single,
        # un-sharded `t5xxl_fp16.safetensors` whose tensor names match what
        # t5::TextEncoder expects. The tokenizer.json is the plain T5
        # SentencePiece Unigram model (identical across t5 / t5-v1.1), from
        # google-t5/t5-base; config.json is informational. A `$files` entry
        # may carry a `repo|path` override to pull from a different repo.
        if (-not $Repo)   { $Repo   = "comfyanonymous/flux_text_encoders" }
        if (-not $OutDir) { $OutDir = "$PSScriptRoot/../weights/t5-xxl" }
        $files = @(
            "t5xxl_fp16.safetensors",
            "google-t5/t5-base|tokenizer.json",
            "google/t5-v1_1-xxl|config.json"
        )
        $fallback = @{}
    }
    "flux-schnell" {
        # Flux.1-schnell, diffusers format. The transformer + the T5-XXL text
        # encoder ship sharded — only the `.index.json` weight-maps are listed
        # here; the shard files themselves are discovered from those indexes
        # after the main download loop. NOTE: large download
        # (~24 GB transformer + ~10 GB T5).
        if (-not $Repo)   { $Repo   = "black-forest-labs/FLUX.1-schnell" }
        if (-not $OutDir) { $OutDir = "$PSScriptRoot/../weights/flux-schnell" }
        $files = @(
            "model_index.json",
            "scheduler/scheduler_config.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
            "text_encoder/config.json",
            "text_encoder/model.safetensors",
            "text_encoder_2/config.json",
            "text_encoder_2/model.safetensors.index.json",
            "tokenizer/vocab.json",
            "tokenizer/merges.txt",
            "tokenizer_2/tokenizer.json"
        )
        $fallback = @{}
    }
}

$resolved = (Resolve-Path -LiteralPath $OutDir -ErrorAction SilentlyContinue)
if (-not $resolved) {
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    $resolved = (Resolve-Path -LiteralPath $OutDir)
}
$OutDir = $resolved.Path

Write-Host "Model:   $Model"
Write-Host "Repo:    $Repo"
Write-Host "Target:  $OutDir"

# A $files entry may be a bare path (fetched from $Repo) or `repo|path` to
# pull that one file from a different repo — used by t5-xxl, whose weights,
# tokenizer, and config live in three separate public repos.
foreach ($entry in $files) {
    $entRepo = $Repo
    $f       = $entry
    if ($entry -match '\|') {
        $entRepo, $f = $entry -split '\|', 2
    }
    Write-Host "==> hf download $entRepo $f"
    & hf download $entRepo $f --local-dir $OutDir
    if ($LASTEXITCODE -ne 0) {
        if ($fallback.ContainsKey($f)) {
            $alt = $fallback[$f]
            Write-Host "    primary failed, trying fallback: $alt"
            & hf download $entRepo $alt --local-dir $OutDir
            if ($LASTEXITCODE -ne 0) {
                throw "hf download failed for both $f and $alt (exit $LASTEXITCODE)"
            }
        } else {
            throw "hf download failed for $f (exit $LASTEXITCODE)"
        }
    }
}

# --- sharded components -----------------------------------------------------
# For models whose components ship sharded (flux-schnell), the loop above
# fetched the `*.index.json` weight-maps. Extract the shard filenames from
# each index and fetch every shard.
$indexFiles = Get-ChildItem -LiteralPath $OutDir -Recurse -File `
    -Filter "*.index.json" -ErrorAction SilentlyContinue
foreach ($idx in $indexFiles) {
    $compRel = $idx.Directory.Name
    Write-Host ""
    Write-Host "Expanding shards from $($idx.FullName.Substring($OutDir.Length + 1))"
    $text = Get-Content -LiteralPath $idx.FullName -Raw
    $shards = [regex]::Matches($text,
        '[A-Za-z0-9_.-]+-[0-9]+-of-[0-9]+\.safetensors') |
        ForEach-Object { $_.Value } | Sort-Object -Unique
    foreach ($s in $shards) {
        $rel = "$compRel/$s"
        Write-Host "==> hf download $Repo $rel"
        & hf download $Repo $rel --local-dir $OutDir
        if ($LASTEXITCODE -ne 0) {
            throw "hf download failed for shard $rel (exit $LASTEXITCODE)"
        }
    }
}

Write-Host ""
Write-Host "Done. Files in $OutDir :"
Get-ChildItem -LiteralPath $OutDir -Recurse -File |
    ForEach-Object {
        $sz = "{0,10:N0}" -f $_.Length
        Write-Host "  $sz  $($_.FullName.Substring($OutDir.Length + 1))"
    }
