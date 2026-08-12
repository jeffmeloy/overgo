#!/usr/bin/env bash
# overgo_stop_gate.sh -- the execution-driving Stop hook.
#
# Refuses to let a turn END unless the turn either made plan PROGRESS (a commit
# landed or the dispatched step changed), has a GATE commit in flight, or recorded
# a LEGITIMATE stop (go run ./cmd/plan -stop <reason>). A self-invented
# "checkpoint" or "should I continue?" is none of these -- it is blocked. This is
# the piece that makes dispatch DRIVE execution instead of only tracking it.
#
# Wire it in settings.json as a Stop hook (owner environment step):
#   "hooks": { "Stop": [{ "hooks": [{ "type": "command",
#     "command": "bash scripts/overgo_stop_gate.sh" }] }] }
set -u

if [ "${1:-}" = "--selftest" ]; then
	echo "overgo_stop_gate selftest ok"
	exit 0
fi

# Re-attempt valve: the harness re-runs the Stop hook with stop_hook_active=true
# right after a block. Let the retry through -- otherwise a turn with no plan
# progress (a bounded question, a status readout, a genuine hold) can never be
# ended and the session hard-wedges. Read stdin AFTER --selftest so a manual
# selftest (no piped stdin) does not hang on cat.
in="$(cat 2>/dev/null || true)"
case "$in" in
*'"stop_hook_active":true'* | *'"stop_hook_active": true'*) exit 0 ;;
esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 0
STATE="docs/.loop_state" # gitignored: last-seen HEAD + dispatched step

head="$(git rev-parse HEAD 2>/dev/null || echo none)"
next="$(go run ./cmd/plan -next 2>/dev/null | head -1)"

prev_head=""
prev_next=""
if [ -f "$STATE" ]; then
	prev_head="$(sed -n 1p "$STATE")"
	prev_next="$(sed -n 2p "$STATE")"
fi

# Plan complete -> nothing to drive.
if echo "$next" | grep -qi "plan complete"; then
	exit 0
fi

# Progress this turn: a commit landed, or the dispatched step advanced.
progressed=0
[ "$head" != "$prev_head" ] && progressed=1
[ "$next" != "$prev_next" ] && progressed=1

# A gate commit in flight (background commit is legitimate mid-turn).
gate_running=0
if command -v tasklist >/dev/null 2>&1; then
	tasklist 2>/dev/null | grep -qi "gate.exe" && gate_running=1
else
	pgrep -f "gate.exe" >/dev/null 2>&1 && gate_running=1
fi

# A legitimate stop recorded at the CURRENT HEAD (stale markers do not count).
fresh_stop=0
if [ -f docs/plan_stop.json ] && grep -q "\"head\":\"$head\"" docs/plan_stop.json 2>/dev/null; then
	fresh_stop=1
fi

if [ "$progressed" = 1 ] || [ "$gate_running" = 1 ] || [ "$fresh_stop" = 1 ]; then
	printf '%s\n%s\n' "$head" "$next" >"$STATE"
	exit 0
fi

cat >&2 <<'MSG'
overgo loop gate: this turn made no plan progress.
Do the dispatched step -- `go run ./cmd/plan -prompt` for the task, build it,
commit via the plan-bound gate (`go run ./cmd/gate -plan <item>/<step> ...`), then
`go run ./cmd/plan -advance <item> <step>`.
If you genuinely cannot proceed, record a legitimate stop instead:
  go run ./cmd/plan -stop <user-stop|irreversible|external-prereq>: <detail>
A "checkpoint", "milestone", or "should I continue?" is NOT a valid stop.
MSG
exit 2
