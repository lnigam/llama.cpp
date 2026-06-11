param(
    [string] $Model = "D:\Day0-1\northbloom\diffusion-gemma-26b-v10-q4_k_m.gguf",
    [string] $Exe = "D:\Day0-1\northbloom\llama.cpp-codex-sampling\pr-changes\llama.cpp\build-review\bin\Release\llama-diffusion-gemma-cli.exe",
    [string] $OutDir = "D:\Day0-1\northbloom\llama.cpp-codex-sampling\pr-changes\llama.cpp\profiles\diffusion-kernel-efficiency\ncu-admin",
    [string] $Prompt = "Write one short sentence about CUDA graphs.",
    [int] $TopK = 64
)

$ErrorActionPreference = "Stop"

function Resolve-Ncu {
    $cmd = Get-Command ncu.exe -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    $candidates = @(
        "C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.2.0\target\windows-desktop-win7-x64\ncu.exe",
        "C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.2.0\host\target-windows-x64\ncu.exe",
        "C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.1.0\target\windows-desktop-win7-x64\ncu.exe",
        "C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.1.0\host\target-windows-x64\ncu.exe",
        "C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.3.1\target\windows-desktop-win7-x64\ncu.exe",
        "C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.3.1\host\target-windows-x64\ncu.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    $found = Get-ChildItem "C:\Program Files\NVIDIA Corporation" -Recurse -Filter ncu.exe -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($found) {
        return $found.FullName
    }

    throw "Could not find ncu.exe. Add Nsight Compute to PATH or edit Resolve-Ncu in this script."
}

if (-not (Test-Path $Exe)) {
    throw "Executable not found: $Exe"
}
if (-not (Test-Path $Model)) {
    throw "Model not found: $Model"
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$ncu = Resolve-Ncu
Write-Host "Using NCU: $ncu"
Write-Host "Output : $OutDir"

$runArgs = @(
    $Exe,
    "-m", $Model,
    "-p", $Prompt,
    "-n", "256",
    "--diffusion-steps", "48",
    "--top-k", "$TopK",
    "-ngl", "all",
    "-sm", "none",
    "-mg", "0",
    "-fa", "on",
    "--ctx-size", "4096",
    "--seed", "42",
    "--no-mmproj",
    "--diffusion-cuda-direct-self-cond",
    "--diffusion-cuda-final-tokens-on-stop",
    "--diffusion-cuda-tight-top-k",
    "--diffusion-cuda-top-k-local-k", "4"
)

function Set-EnvMap($map) {
    foreach ($kv in $map.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable($kv.Key, $kv.Value, "Process")
    }
}

function Clear-ExperimentEnv {
    $names = @(
        "GGML_CUDA_MMQ_STREAM_K",
        "GGML_CUDA_MMQ_STREAM_K_DIVISOR",
        "GGML_CUDA_MMQ_STREAM_K_DIVISOR_MIN_PCT",
        "GGML_CUDA_MMQ_AVOID_FIXUP",
        "GGML_CUDA_MMQ_AVOID_FIXUP_MIN_EFF",
        "GGML_CUDA_DISABLE_GRAPHS"
    )
    foreach ($name in $names) {
        [Environment]::SetEnvironmentVariable($name, $null, "Process")
    }
}

function Invoke-NcuProfile {
    param(
        [string] $Name,
        [string] $KernelRegex,
        [string] $MetricSet = "basic",
        [int] $LaunchSkip = 0,
        [int] $LaunchCount = 4,
        [string[]] $ExtraArgs = @(),
        [hashtable] $ExtraEnv = @{}
    )

    Clear-ExperimentEnv
    Set-EnvMap $ExtraEnv

    $export = Join-Path $OutDir $Name
    $args = @(
        "--force-overwrite",
        "--target-processes", "all",
        "--graph-profiling", "node",
        "--replay-mode", "application",
        "--set", $MetricSet,
        "--kernel-name", "regex:$KernelRegex",
        "--launch-skip", "$LaunchSkip",
        "--launch-count", "$LaunchCount",
        "--export", $export,
        "--"
    ) + $runArgs + $ExtraArgs

    Write-Host ""
    Write-Host "== $Name =="
    Write-Host "kernel regex: $KernelRegex"
    Write-Host "set=$MetricSet skip=$LaunchSkip count=$LaunchCount"
    & $ncu @args
    if ($LASTEXITCODE -ne 0) {
        throw "NCU failed for $Name with exit code $LASTEXITCODE"
    }
}

Invoke-NcuProfile `
    -Name "01_topk_fused_local4_detailed" `
    -KernelRegex "diffusion_sample_topk_fused_local_kernel" `
    -MetricSet "detailed" `
    -LaunchSkip 0 `
    -LaunchCount 4 `
    -ExtraArgs @("--diffusion-cuda-fused-top-k-sample")

Invoke-NcuProfile `
    -Name "02_topk_baseline_select_detailed" `
    -KernelRegex "diffusion_select_topk_local_kernel" `
    -MetricSet "detailed" `
    -LaunchSkip 0 `
    -LaunchCount 4 `
    -ExtraArgs @("--diffusion-cuda-top-k-local-k", "8")

Invoke-NcuProfile `
    -Name "03_mmq_denoise_basic" `
    -KernelRegex "mul_mat_q|mul_mat_q_stream_k_fixup|quantize_mmq_q8_1" `
    -MetricSet "basic" `
    -LaunchSkip 900 `
    -LaunchCount 16 `
    -ExtraArgs @("--diffusion-cuda-fused-top-k-sample")

Invoke-NcuProfile `
    -Name "04_mmq_no_fixup_aggressive_basic" `
    -KernelRegex "mul_mat_q|mul_mat_q_stream_k_fixup|quantize_mmq_q8_1" `
    -MetricSet "basic" `
    -LaunchSkip 900 `
    -LaunchCount 16 `
    -ExtraArgs @("--diffusion-cuda-fused-top-k-sample") `
    -ExtraEnv @{
        GGML_CUDA_MMQ_STREAM_K_DIVISOR = "1"
        GGML_CUDA_MMQ_AVOID_FIXUP = "1"
        GGML_CUDA_MMQ_AVOID_FIXUP_MIN_EFF = "0"
    }

Invoke-NcuProfile `
    -Name "05_attention_basic" `
    -KernelRegex "flash_attn_ext_f16|flash_attn_stream_k_fixup" `
    -MetricSet "basic" `
    -LaunchSkip 80 `
    -LaunchCount 16 `
    -ExtraArgs @("--diffusion-cuda-fused-top-k-sample")

Invoke-NcuProfile `
    -Name "06_misc_hotspots_basic" `
    -KernelRegex "cpy_scalar_transpose|reduce_rows_f32|k_get_rows_float|softcap_f32" `
    -MetricSet "basic" `
    -LaunchSkip 0 `
    -LaunchCount 16 `
    -ExtraArgs @("--diffusion-cuda-fused-top-k-sample")

Write-Host ""
Write-Host "NCU collection complete. Reports are in:"
Write-Host $OutDir
