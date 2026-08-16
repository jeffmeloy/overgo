#!/usr/bin/env bash
# Thin hook adapter -> cmd/loophook (Go owns the logic). Stop hook: exit 2 blocks
# a turn-end that orphans work, 0 allows. Uses the session-built binary; falls
# back to `go run` if absent. Never bricks the session (missing binary -> allow).
[ "${1:-}" = "--selftest" ] && { echo "overgo_stop_gate selftest ok"; exit 0; }
# Repo scoping: govern only sessions whose active project is this repo.
# CLAUDE_PROJECT_DIR is the harness-typed signal (probe 2026-08-16: populated).
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -n "${CLAUDE_PROJECT_DIR:-}" ]; then
  project="$(cd "$CLAUDE_PROJECT_DIR" 2>/dev/null && pwd || echo)"
  [ "$project" = "$here" ] || exit 0
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 0
[ -x bin/loophook.exe ] && exec bin/loophook.exe stop
exec go run ./cmd/loophook stop
