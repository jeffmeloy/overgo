#!/usr/bin/env bash
# Thin hook adapter -> cmd/loophook (Go owns the logic). SessionStart: builds the
# hook binary FRESH for the session (so Stop/PostToolUse run fast and current),
# then emits the overgo turn contract + the dispatched step. The Go doctrine
# subcommand also sweeps the stale dispatch marker.
[ "${1:-}" = "--selftest" ] && { echo "overgo_doctrine selftest ok"; exit 0; }
# Repo scoping: govern only sessions whose active project is this repo.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -n "${CLAUDE_PROJECT_DIR:-}" ]; then
  project="$(cd "$CLAUDE_PROJECT_DIR" 2>/dev/null && pwd || echo)"
  [ "$project" = "$here" ] || exit 0
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 0
# Build the shell guard alongside the hook binary: scripts/guard.sh fails open
# when bin/guard.exe is absent, so an unbuilt guard silently disables the
# destructive-command and raw-commit protections for the whole session.
go build -o bin/guard.exe ./cmd/guard 2>/dev/null || echo "overgo doctrine: WARNING bin/guard.exe build failed; shell guard is INACTIVE" >&2
if go build -o bin/loophook.exe ./cmd/loophook 2>/dev/null; then
	exec bin/loophook.exe doctrine
fi
exec go run ./cmd/loophook doctrine
