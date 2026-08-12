#!/usr/bin/env bash
# Thin hook adapter -> cmd/loophook (Go owns the logic). Stop hook: exit 2 blocks
# a turn-end that orphans work, 0 allows. Uses the session-built binary; falls
# back to `go run` if absent. Never bricks the session (missing binary -> allow).
[ "${1:-}" = "--selftest" ] && { echo "overgo_stop_gate selftest ok"; exit 0; }
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 0
[ -x bin/loophook.exe ] && exec bin/loophook.exe stop
exec go run ./cmd/loophook stop
