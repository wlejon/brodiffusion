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
    [ValidateSet("sd15", "lcm-dreamshaper")]
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

foreach ($f in $files) {
    Write-Host "==> hf download $Repo $f"
    & hf download $Repo $f --local-dir $OutDir
    if ($LASTEXITCODE -ne 0) {
        if ($fallback.ContainsKey($f)) {
            $alt = $fallback[$f]
            Write-Host "    primary failed, trying fallback: $alt"
            & hf download $Repo $alt --local-dir $OutDir
            if ($LASTEXITCODE -ne 0) {
                throw "hf download failed for both $f and $alt (exit $LASTEXITCODE)"
            }
        } else {
            throw "hf download failed for $f (exit $LASTEXITCODE)"
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
