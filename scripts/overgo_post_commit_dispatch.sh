#!/usr/bin/env bash
# Thin hook adapter -> cmd/loophook (Go owns the logic). PostToolUse: arms/clears
# docs/.dispatch_pending at the commit boundary so the Stop gate refuses a
# milestone-stop until the next step is dispatched.
[ "${1:-}" = "--selftest" ] && { echo "overgo_post_commit_dispatch selftest ok"; exit 0; }
# Repo scoping: govern only sessions whose active project is this repo.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -n "${CLAUDE_PROJECT_DIR:-}" ]; then
  project="$(cd "$CLAUDE_PROJECT_DIR" 2>/dev/null && pwd || echo)"
  [ "$project" = "$here" ] || exit 0
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 0
[ -x bin/loophook.exe ] && exec bin/loophook.exe post-commit
exec go run ./cmd/loophook post-commit
