package main

// doctrineText is the overgo turn contract emitted at SessionStart so a session
// is driven by OVERGO's loop and plan, never adaptive_new's. adaptive_new is a
// reference / comparison / port-source tree only -- read it, never drive from it.
const doctrineText = `OVERGO TURN CONTRACT. This repo is the whole world. adaptive_new is a
reference / comparison / port-source tree ONLY -- read it to compare or port
from, never as the control plane, and never modify it.

THE LOOP (do #1 -> complete -> refactor -> do #1, until the plan is complete):
 1. #1 is ` + "`go run ./cmd/plan -next`" + `. Do ONLY that step.
 2. Build it. "Done" is machine-checked: ` + "`go run ./cmd/plan -verify`" + ` exits 0,
    NOT a prose summary of what you did.
 3. Commit through the plan-bound gate (never raw git commit during a campaign):
      go run ./cmd/gate -plan <item>/<step> -message-file <f> -paths <csv>
    Merges: git merge --no-ff --no-commit <branch> then go run ./cmd/gate -merge
    -plan <item>/<step> -message-file <f>. Read the exit code UNPIPED. Gate runs
    go in the background; confirm from git, not the notification.
 4. Advance: ` + "`go run ./cmd/plan -advance <item> <step>`" + ` (gated on the step's
    verify). Set a step's verify with -setverify; inject a task with -add (flags
    BEFORE the positional id for both).
 5. Refactor the plan from the RESULT of the last task AND any USER INPUT:
    re-scope / re-rank / add / set-verify. A user request becomes the new #1
    AFTER the current work is committed -- it does NOT interrupt in-flight work.
 6. Repeat. Prefer continuing to the next #1; continuity is owned by cmd/loop
    (the harness-agnostic driver), never argued turn-by-turn.

STOPS. The Stop gate protects WORK and the LOOP: it blocks a turn-end that
would orphan turn-created uncommitted work (commit through the gate or
revert), and it blocks a turn-end while the plan holds ANY open row -- a
landed commit is necessary progress, never a sanctioned end; the loop runs
until the plan is EMPTY. A prompt from the user NEVER pauses the loop (owner
rule 2026-08-20): answer it, then continue the dispatched row in the same
turn. The one sanctioned pause is an explicit user stop, recorded with
go run ./cmd/plan -stop <user-stop|irreversible|external-prereq>:<detail>.
For unattended continuity run go run ./cmd/loop.

SCOPE. A bounded request -- a status readout, an explanation, or a review --
bounds the ANSWER, not the turn: keep the answer to what was asked (no
inflating it into a campaign), then continue the dispatched row.`
