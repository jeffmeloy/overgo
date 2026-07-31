[CmdletBinding()]
param(
    [ValidatePattern('^[0-9]+(ms|s|m)$')]
    [string]$Duration = "5s"
)

$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$targets = @(
    @{ Package = "./internal/gguf";      Name = "FuzzParseNeverPanics" },
    @{ Package = "./internal/tokenizer"; Name = "FuzzEncodeDecodeNeverPanics" },
    @{ Package = "./internal/sampling";  Name = "FuzzGBNFCompileNeverPanics" },
    @{ Package = "./internal/sampling";  Name = "FuzzJSONSchemaToGrammarNeverPanics" },
    @{ Package = "./internal/sampling";  Name = "FuzzSamplerLoadStateNeverPanics" },
    @{ Package = "./internal/inference"; Name = "FuzzCacheStateNeverPanics" },
    @{ Package = "./internal/inference"; Name = "FuzzSessionStateNeverPanics" },
    @{ Package = "./internal/server";    Name = "FuzzJSONEndpointsNeverPanic" }
)

Push-Location $repositoryRoot
try {
    foreach ($target in $targets) {
        go test $target.Package "-run=^$" "-fuzz=$($target.Name)" "-fuzztime=$Duration"
        if ($LASTEXITCODE -ne 0) {
            throw "fuzz target $($target.Name) failed"
        }
    }
} finally {
    Pop-Location
}
