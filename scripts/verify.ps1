[CmdletBinding()]
param(
    [switch]$CUDA
)

$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $PSScriptRoot

Push-Location $repositoryRoot
try {
    gofmt -w cmd internal
    if ($LASTEXITCODE -ne 0) {
        throw "gofmt failed"
    }
    go test ./...
    if ($LASTEXITCODE -ne 0) {
        throw "unit tests failed"
    }
    go run ./cmd/sbom -check
    if ($LASTEXITCODE -ne 0) {
        throw "SBOM verification failed"
    }
    go run ./cmd/kernel-manifest
    if ($LASTEXITCODE -ne 0) {
        throw "kernel manifest verification failed"
    }
    go run ./cmd/compatibility -check
    if ($LASTEXITCODE -ne 0) {
        throw "compatibility verification failed"
    }

    if ($CUDA) {
        $previous = $env:LLAMACPP2GO_CUDA_TEST
        try {
            $env:LLAMACPP2GO_CUDA_TEST = "1"
            go test ./internal/cuda/... -count=1
            if ($LASTEXITCODE -ne 0) {
                throw "CUDA integration tests failed"
            }
            go run ./cmd/cuda-info
            if ($LASTEXITCODE -ne 0) {
                throw "cuda-info failed"
            }
            go run ./cmd/cuda-smoke
            if ($LASTEXITCODE -ne 0) {
                throw "cuda-smoke failed"
            }
        } finally {
            $env:LLAMACPP2GO_CUDA_TEST = $previous
        }
    }
} finally {
    Pop-Location
}
