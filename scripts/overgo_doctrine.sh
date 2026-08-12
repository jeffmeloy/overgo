#!/usr/bin/env bash
# Thin hook adapter -> cmd/loophook (Go owns the logic). SessionStart: builds the
# hook binary FRESH for the session (so Stop/PostToolUse run fast and current),
# then emits the overgo turn contract + the dispatched step. The Go doctrine
# subcommand also sweeps the stale dispatch marker.
[ "${1:-}" = "--selftest" ] && { echo "overgo_doctrine selftest ok"; exit 0; }
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 0
if go build -o bin/loophook.exe ./cmd/loophook 2>/dev/null; then
	exec bin/loophook.exe doctrine
fi
exec go run ./cmd/loophook doctrine
