#!/usr/bin/env bash
# fuzz-smoke: fuzzes every fuzz target the tree holds, each for a short duration
# ($1, default 5s), so a parser or template fault that mutation finds is found
# before a run. The targets are listed from the tree, not kept by hand; the
# scheduled fuzz workflow (.github/workflows/fuzz.yml) runs it.
set -euo pipefail
duration="${1:-5s}"
case "$duration" in
  *[0-9]ms|*[0-9]s|*[0-9]m) ;;
  *) echo "fuzz-smoke: duration must match ^[0-9]+(ms|s|m)$" >&2; exit 2 ;;
esac
cd "$(dirname "$0")/.."
count=0
for directory in $(grep -rl --include='*_test.go' '^func Fuzz' cmd internal | xargs -n1 dirname | sort -u); do
  for name in $(go test -list '^Fuzz' "./$directory" | grep '^Fuzz' || true); do
    echo "[fuzz] ./$directory $name ($duration)"
    go test "./$directory" "-run=^$" "-fuzz=^${name}\$" "-fuzztime=$duration"
    count=$((count + 1))
  done
done
if [ "$count" -eq 0 ]; then
  echo "fuzz-smoke: the tree holds no fuzz target" >&2
  exit 1
fi
echo "fuzz-smoke: all $count targets clean"
