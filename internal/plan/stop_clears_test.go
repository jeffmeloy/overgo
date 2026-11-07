package plan

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestAnOutsideStopClearsWhenItsCauseDoes holds a stop on an outside need to the check it names: on 2026-11-07 a stop
// on the trunk's hook setting outlived the owner's fixing it, waiting for a resume by hand. A hardware stop whose check
// fails stands; once the check passes, the next read of the work resumes it, recording what cleared it. An outside need
// with no check is refused, as is a requested stop that names one; and a requested stop is never cleared but by the
// owner's resume.
func TestAnOutsideStopClearsWhenItsCauseDoes(t *testing.T) {
	t.Parallel()
	root, event := stopFixture(t)
	event.ReasonCode, event.Detail, event.ClearsWhen = stopHardware, "a second graphics card", "test -f card-present"
	recorded := recordStop(t, event)
	stop := testcheck.Must(ReadStop(t.Context(), root, orchestrationHead, ExecutionInteractive))(t)
	if after, cleared, err := clearOutsideStop(t.Context(), root, orchestrationHead, ExecutionInteractive, stop); err != nil || cleared || !after.Blocked {
		t.Fatalf("with its check failing, the hardware stop reads %+v, cleared %v (%v); want it standing", after, cleared, err)
	}
	testcheck.NoError(t, os.WriteFile(filepath.Join(root, "card-present"), nil, 0o600))
	after, cleared, err := clearOutsideStop(t.Context(), root, orchestrationHead, ExecutionInteractive, stop)
	if err != nil || !cleared || after.Blocked || after.State != stopResumed || after.Event.Previous != recorded.ID || !strings.Contains(after.Event.Detail, "test -f card-present") {
		t.Fatalf("with its check passing, the hardware stop reads %+v, cleared %v (%v); want it resumed, the check named", after, cleared, err)
	}

	unchecked := event
	unchecked.ClearsWhen, unchecked.Previous = "", after.ID
	if _, err := RecordStop(t.Context(), unchecked); err == nil {
		t.Fatal("a hardware stop with no check that finds it met was recorded")
	}
	requested := event
	requested.ReasonCode, requested.Detail, requested.Previous = StopRequested, "the owner said stop", after.ID
	if _, err := RecordStop(t.Context(), requested); err == nil {
		t.Fatal("a requested stop that names a check was recorded")
	}
	requested.ClearsWhen = ""
	recordStop(t, requested)
	standing := testcheck.Must(ReadStop(t.Context(), root, orchestrationHead, ExecutionInteractive))(t)
	if after, cleared, err := clearOutsideStop(t.Context(), root, orchestrationHead, ExecutionInteractive, standing); err != nil || cleared || !after.Blocked {
		t.Fatalf("a requested stop reads %+v, cleared %v (%v); want it standing until the owner resumes it", after, cleared, err)
	}
}
