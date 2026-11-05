package main

import (
	"encoding/json"
	"strings"
	"testing"

	"overgo/internal/plan"
	"overgo/internal/testcheck"
)

// TestLoopHookConsumesDispatchJSON pins the hook's consumption of the
// dispatch as data: the JSON the plan tool prints decodes into the row the
// stop message names, and a complete dispatch is read from its field.
func TestLoopHookConsumesDispatchJSON(t *testing.T) {
	var dispatch plan.Dispatch
	encoded := `{"complete":false,"item":"item","step":"do","item_title":"item title","step_title":"make the change","verify":"go test ./x","line":"item / do: item title -- make the change"}`
	testcheck.NoError(t, json.Unmarshal([]byte(encoded), &dispatch))
	message := stopMessage(stopContinue, true, dispatch)
	if !strings.Contains(message, "the row landed") || !strings.HasSuffix(message, "Dispatched now: item / do: item title -- make the change\n") {
		t.Fatalf("stop message = %q", message)
	}
	if message := stopMessage(stopOrphan, false, dispatch); !strings.Contains(message, "orphan uncommitted work") {
		t.Fatalf("orphan message = %q", message)
	}
	var done plan.Dispatch
	testcheck.NoError(t, json.Unmarshal([]byte(`{"complete":true,"line":"plan complete: every item is done"}`), &done))
	if !done.Complete || stopDecision(false, false, done.Complete, false) != stopAllow {
		t.Fatalf("complete dispatch = %+v did not allow the turn to end", done)
	}
}
