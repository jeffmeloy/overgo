[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$cudaRoot = $env:CUDA_PATH
if ([string]::IsNullOrWhiteSpace($cudaRoot)) {
    $cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
}
$nvcc = Join-Path $cudaRoot "bin\nvcc.exe"
if (-not (Test-Path -LiteralPath $nvcc)) {
    throw "nvcc was not found at $nvcc"
}

$compilerCandidates = @(
    "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64",
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC"
)
$compilerDirectory = $null
foreach ($candidate in $compilerCandidates) {
    if (Test-Path -LiteralPath (Join-Path $candidate "cl.exe")) {
        $compilerDirectory = $candidate
        break
    }
}
if ($null -eq $compilerDirectory) {
    throw "A CUDA 12.9-compatible MSVC x64 compiler was not found"
}

$kernels = @(
    @{
        Source = "kernels\cuda\vector_add.cu"
        Output = "internal\cuda\kernel\vector_add.ptx"
    },
    @{
        Source = "kernels\cuda\ops_f32.cu"
        Output = "internal\cuda\kernel\ops_f32.ptx"
    }
)
foreach ($kernel in $kernels) {
    $source = Join-Path $repositoryRoot $kernel.Source
    $output = Join-Path $repositoryRoot $kernel.Output
    & $nvcc -ptx -arch=compute_89 -use_fast_math -ccbin $compilerDirectory -o $output $source
    if ($LASTEXITCODE -ne 0) {
        throw "nvcc failed with exit code $LASTEXITCODE"
    }
    Write-Output "Generated $output"
}

$validationPath = Join-Path $repositoryRoot "internal\cuda\kernel\validation.go"
$validation = [System.IO.File]::ReadAllText($validationPath)
$runtimePins = @(
    @{
        Name = "VectorAddSHA256"
        Asset = Join-Path $repositoryRoot "internal\cuda\kernel\vector_add.ptx"
    },
    @{
        Name = "OpsF32SHA256"
        Asset = Join-Path $repositoryRoot "internal\cuda\kernel\ops_f32.ptx"
    }
)
foreach ($pin in $runtimePins) {
    $hash = (Get-FileHash -LiteralPath $pin.Asset -Algorithm SHA256).Hash.ToLowerInvariant()
    $pattern = "(?m)^(\s*" + [regex]::Escape($pin.Name) + "\s*=\s*`")[0-9a-f]{64}(`"\s*)$"
    if ([regex]::Matches($validation, $pattern).Count -ne 1) {
        throw "Expected exactly one $($pin.Name) runtime pin in $validationPath"
    }
    $validation = [regex]::Replace(
        $validation,
        $pattern,
        ('${1}' + $hash + '${2}')
    )
}
$utf8NoBOM = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($validationPath, $validation, $utf8NoBOM)

go run ./cmd/kernel-manifest -update
if ($LASTEXITCODE -ne 0) {
    throw "kernel manifest update failed"
}
go run ./cmd/kernel-manifest
if ($LASTEXITCODE -ne 0) {
    throw "kernel manifest verification failed"
}
