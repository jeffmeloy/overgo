#!/usr/bin/env bash
# PreToolUse safety adapter. Go owns parsing and policy; a missing binary
# fails open so a build gap never bricks the session (the corpus is where
# strictness lives). Build: go build -o bin/guard.exe ./cmd/guard
repo="$(cd "$(dirname "$0")/.." && pwd)"
# Repo scoping: govern only sessions whose active project is this repo.
if [ -n "${CLAUDE_PROJECT_DIR:-}" ]; then
  project="$(cd "$CLAUDE_PROJECT_DIR" 2>/dev/null && pwd || echo)"
  [ "$project" = "$repo" ] || exit 0
fi
bin="$repo/bin/guard.exe"
[ -x "$bin" ] || exit 0
exec "$bin"
