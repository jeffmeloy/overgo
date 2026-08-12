#!/usr/bin/env bash
# overgo_doctrine.sh -- SessionStart doctrine for overgo-rooted sessions.
#
# Emits the overgo turn contract so a session is driven by OVERGO's loop and
# OVERGO's plan, never adaptive_new's. adaptive_new is a reference / comparison
# / implementation-source tree ONLY -- never the control plane. Wire as a
# SessionStart hook in .claude/settings.json; the enforcement half is the Stop
# hook (scripts/overgo_stop_gate.sh), wired both here and user-level so it holds
# regardless of where the session was launched.
set -u

if [ "${1:-}" = "--selftest" ]; then
	echo "overgo_doctrine selftest ok"
	exit 0
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 0

cat <<'DOCTRINE'
OVERGO TURN CONTRACT. This repo is the whole world. adaptive_new is a
reference / comparison / port-source tree ONLY -- read it to compare or port
from, never as the control plane, and never modify it.

THE LOOP (do #1 -> complete -> refactor -> do #1, until the plan is complete):
 1. #1 is `go run ./cmd/plan -next`. Do ONLY that step.
 2. Build it. "Done" is machine-checked: `go run ./cmd/plan -verify` exits 0,
    NOT a prose summary of what you did.
 3. Commit through the plan-bound gate (never raw git commit during a campaign):
      go run ./cmd/gate -plan <item>/<step> -message-file <f> -paths <csv>
    Merges: `git merge --no-ff --no-commit <branch>` then `go run ./cmd/gate
    -merge -plan <item>/<step> -message-file <f>`. Read the exit code UNPIPED.
    Gate runs go in the background; confirm from git, not the notification.
 4. Advance: `go run ./cmd/plan -advance <item> <step>` (gated on that step's
    verify). Set an existing step's verify with `-setverify`; inject a task with
    `-add` (flags BEFORE the positional id for both).
 5. Refactor the plan from the RESULT of the last task AND any USER INPUT:
    re-scope / re-rank / add / set-verify. A user request becomes the new #1
    AFTER the current work is committed -- it does NOT interrupt in-flight work.
    The refactor step is where user input and findings enter the plan.
 6. Repeat. After a commit, dispatch the next #1 the SAME turn -- do not stop at
    a clean boundary while open work remains (that is the milestone-stop failure).

STOPS. End a turn ONLY when: the turn made plan progress (a commit landed or the
dispatched step advanced), a gate commit is in flight, or a valid stop was
recorded: `go run ./cmd/plan -stop <user-stop|irreversible|external-prereq>:
<detail>`. A checkpoint, milestone, written summary, "big change do fresh", or
"should I continue?" is NOT a valid stop.

SCOPE. A bounded request -- a status readout, an explanation, a review, one
scoped edit -- is a COMPLETE turn once answered; do not inflate it into a
campaign. The Stop gate enforces the loop, not bounded requests.
DOCTRINE

echo ""
echo "Dispatched now (go run ./cmd/plan -next):"
go run ./cmd/plan -next 2>/dev/null | head -1
