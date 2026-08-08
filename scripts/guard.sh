#!/usr/bin/env bash
# PreToolUse safety adapter. Go owns parsing and policy; a missing binary
# fails open so a build gap never bricks the session (the corpus is where
# strictness lives). Build: go build -o bin/guard.exe ./cmd/guard
repo="$(cd "$(dirname "$0")/.." && pwd)"
bin="$repo/bin/guard.exe"
[ -x "$bin" ] || exit 0
exec "$bin"
