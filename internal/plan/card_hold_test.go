package plan

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/testcheck"
)

// TestDispatchPassesOverCardStepsWhileASessionsRunHoldsTheCard holds dispatch to host work while a session's own run
// holds the card (fail-early-runs/card-held): on 2026-11-11 guidance-ops' readings took the card for about two days and
// dispatch offered their step, which waited on the card, as it passes over card steps only for a measurement run. With
// the card free the card step is dispatched; while a session's run recorded in the checkout's process state holds it,
// the host step behind it is; once that run's process has ended, the card step again. The hold is written as its record
// reads, so the check reads it the same way before and after the change.
func TestDispatchPassesOverCardStepsWhileASessionsRunHoldsTheCard(t *testing.T) {
	t.Parallel()
	card := Item{ID: "readings", Title: "readings", Status: StatusOpen, Steps: []Step{{ID: "draw", Title: "draw", Status: StatusOpen, Verify: "true", NeedsCard: true}}}
	host := Item{ID: "host", Title: "host", Status: StatusOpen, Steps: []Step{{ID: "do", Title: "do", Status: StatusOpen, Verify: "true"}}}
	repository := planRepository(t, Plan{Campaign: "fixture", Items: []Item{card, host}})
	next := func() string {
		t.Helper()
		return testcheck.Must(ResolveDispatch(t.Context(), repository, DispatchRequest{Mode: ExecutionInteractive}))(t).Item
	}
	hold := func(process int) {
		t.Helper()
		record := testcheck.Must(json.Marshal(map[string]any{"process": process, "purpose": "the readings", "started": time.Now()}))(t)
		state := filepath.Join(repository, ".overgo-runtime")
		testcheck.NoError(t, os.MkdirAll(state, 0o755))
		testcheck.NoError(t, os.WriteFile(filepath.Join(state, "card_hold.json"), record, 0o644))
	}
	if free := next(); free != "readings" {
		t.Fatalf("with the card free dispatch gave %s; want the card step", free)
	}
	hold(os.Getpid())
	if held := next(); held != "host" {
		t.Errorf("while a session's run holds the card dispatch gave %s; want the host step", held)
	}
	ended := exec.Command(os.Args[0], "-test.run=^$")
	testcheck.NoError(t, ended.Run())
	hold(ended.Process.Pid)
	if lapsed := next(); lapsed != "readings" {
		t.Errorf("once the holding run's process ended dispatch gave %s; want the card step again", lapsed)
	}
}
