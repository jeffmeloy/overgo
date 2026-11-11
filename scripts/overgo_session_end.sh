#!/usr/bin/env bash
# Thin hook adapter -> cmd/loophook (Go owns the logic). SessionEnd hook: records
# that the session ended, the one sign the leftovers' session line takes to finish
# its scratch folder. Uses the session-built binary; falls back to `go run` if
# absent. Never blocks the session's end (a failure is only reported).
[ "${1:-}" = "--selftest" ] && { echo "overgo_session_end selftest ok"; exit 0; }
# Repo scoping: govern only sessions whose active project is this repo.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -n "${CLAUDE_PROJECT_DIR:-}" ]; then
  project="$(cd "$CLAUDE_PROJECT_DIR" 2>/dev/null && pwd || echo)"
  [ "$project" = "$here" ] || exit 0
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 0
[ -x bin/loophook.exe ] && exec bin/loophook.exe session-end
exec go run ./cmd/loophook session-end
