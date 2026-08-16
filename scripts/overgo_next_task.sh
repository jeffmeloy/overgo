#!/usr/bin/env bash
# overgo_next_task.sh -- classify a bounded request or emit the plan task.
#
# Wire it in settings.json as a UserPromptSubmit (or SessionStart) hook so every
# turn STARTS bound to the plan's next step -- dispatch = execute, no room to open
# with a self-chosen meta-question:
#   "hooks": { "UserPromptSubmit": [{ "hooks": [{ "type": "command",
#     "command": "bash scripts/overgo_next_task.sh" }] }] }
set -u
# Repo scoping: govern only sessions whose active project is this repo.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -n "${CLAUDE_PROJECT_DIR:-}" ]; then
  project="$(cd "$CLAUDE_PROJECT_DIR" 2>/dev/null && pwd || echo)"
  [ "$project" = "$here" ] || exit 0
fi

if [ "${1:-}" = "--selftest" ]; then
	echo "overgo_next_task selftest ok"
	exit 0
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 0
[ -x bin/loophook.exe ] && exec bin/loophook.exe prompt
exec go run ./cmd/loophook prompt
