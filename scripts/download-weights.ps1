# Download SD1.5 weights for brodiffusion end-to-end testing.
#
# Pulls the diffusers-format FP16 component files from the canonical mirror
# at stable-diffusion-v1-5/stable-diffusion-v1-5. We use the component files
# (not the full v1-5-pruned-emaonly.safetensors) because:
#   - they are already FP16 (the full checkpoint ships as F32),
#   - their tensor names match the diffusers naming our loaders expect
#     (the full checkpoint uses the original LDM names like input_blocks/
#     output_blocks/middle_block which we don't currently translate).
#
# Authentication: requires `hf auth login` to have been run.
#
# Output: <repo>/weights/sd15/
#   text_encoder/model.fp16.safetensors                   (~235 MB)
#   unet/diffusion_pytorch_model.fp16.safetensors         (~1.6 GB)
#   vae/diffusion_pytorch_model.fp16.safetensors          (~160 MB)
#   tokenizer/vocab.json
#   tokenizer/merges.txt

[CmdletBinding()]
param(
    [string]$Repo = "stable-diffusion-v1-5/stable-diffusion-v1-5",
    [string]$OutDir = "$PSScriptRoot/../weights/sd15"
)

$ErrorActionPreference = "Stop"

$resolved = (Resolve-Path -LiteralPath $OutDir -ErrorAction SilentlyContinue)
if (-not $resolved) {
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    $resolved = (Resolve-Path -LiteralPath $OutDir)
}
$OutDir = $resolved.Path

Write-Host "Repo:    $Repo"
Write-Host "Target:  $OutDir"

$files = @(
    "text_encoder/model.fp16.safetensors",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/diffusion_pytorch_model.fp16.safetensors",
    "tokenizer/vocab.json",
    "tokenizer/merges.txt"
)

foreach ($f in $files) {
    Write-Host "==> hf download $Repo $f"
    & hf download $Repo $f --local-dir $OutDir
    if ($LASTEXITCODE -ne 0) {
        throw "hf download failed for $f (exit $LASTEXITCODE)"
    }
}

Write-Host ""
Write-Host "Done. Files in $OutDir :"
Get-ChildItem -LiteralPath $OutDir -Recurse -File |
    ForEach-Object {
        $sz = "{0,10:N0}" -f $_.Length
        Write-Host "  $sz  $($_.FullName.Substring($OutDir.Length + 1))"
    }
