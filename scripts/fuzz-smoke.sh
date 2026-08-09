#!/usr/bin/env bash
# fuzz-smoke: never-panic fuzz sweep (floor component 8; replaces
# fuzz-smoke.ps1). Duration per target via $1, default 5s.
set -euo pipefail
duration="${1:-5s}"
case "$duration" in
  *[0-9]ms|*[0-9]s|*[0-9]m) ;;
  *) echo "fuzz-smoke: duration must match ^[0-9]+(ms|s|m)$" >&2; exit 2 ;;
esac
cd "$(dirname "$0")/.."
targets=(
  "./internal/gguf FuzzParseNeverPanics"
  "./internal/tokenizer FuzzEncodeDecodeNeverPanics"
  "./internal/sampling FuzzGBNFCompileNeverPanics"
  "./internal/sampling FuzzJSONSchemaToGrammarNeverPanics"
  "./internal/sampling FuzzSamplerLoadStateNeverPanics"
  "./internal/inference FuzzCacheStateNeverPanics"
  "./internal/inference FuzzSessionStateNeverPanics"
  "./internal/server FuzzJSONEndpointsNeverPanic"
)
for target in "${targets[@]}"; do
  read -r package name <<<"$target"
  echo "[fuzz] $package $name ($duration)"
  go test "$package" "-run=^$" "-fuzz=$name" "-fuzztime=$duration"
done
echo "fuzz-smoke: all ${#targets[@]} targets clean"
