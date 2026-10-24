package plan

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/processlock"
)

// TestControlVocabularyNamesNoPerson holds the control records' codes free of persons, as the doctrine is.
func TestControlVocabularyNamesNoPerson(t *testing.T) {
	t.Parallel()
	for _, code := range append(slices.Clone(validStopReasons), ResumeCode, MaintenanceCode) {
		for part := range strings.SplitSeq(code, "-") {
			if slices.Contains([]string{"user", "operator", "owner", "human"}, part) {
				t.Errorf("control code %q names a person", code)
			}
		}
	}
}

// TestLegacyStopKeepsItsIdentity holds a stop file written in the codes the current ones replaced to the identities
// its decisions were issued with: its stop and its grant read with their written IDs, the grant admits its exact work
// before and after a grant in the current codes joins it, the stop shows in the current codes, and the stop's written
// ID resumes it, retiring the grant and writing the current codes alone.
func TestLegacyStopKeepsItsIdentity(t *testing.T) {
	t.Parallel()
	root, event := stopFixture(t)
	event.ReasonCode = "user-stop"
	stop, err := identify(event)
	if err != nil {
		t.Fatal(err)
	}
	grant := event
	grant.Kind, grant.ReasonCode, grant.Detail, grant.Task, grant.Previous = ControlMaintenance, "operator-maintenance", "repair", "row/do", stop.ID
	issued, err := identify(grant)
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.MarshalIndent(stopState{Current: event, Maintenance: []ControlEvent{grant}}, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := processlock.EnsureStateDirectory(root); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(root, processlock.StateDirectory, stopFile)
	if err := os.WriteFile(path, data, stopFileMode); err != nil {
		t.Fatal(err)
	}
	read := func() StopStatus {
		t.Helper()
		current, err := ReadStop(t.Context(), root, orchestrationHead, ExecutionInteractive)
		if err != nil || current.State != stopActive || !current.Blocked || current.ID != stop.ID {
			t.Fatalf("a stop written as user-stop reads as %+v (%v), want the active stop %s", current, err, stop.ID)
		}
		if err := current.RequireExecution(grant.Lane, grant.Worker, ExecutionInteractive, grant.Task, issued.ID.String()); err != nil {
			t.Fatalf("the grant issued as %s no longer admits its work: %v", issued.ID, err)
		}
		return current
	}
	if shown := read().String(); !strings.Contains(shown, "reason="+stopRequested+": ") || strings.Contains(shown, "user-stop") {
		t.Fatalf("the stop shows as %s, want the current codes", shown)
	}
	next := grant
	next.ReasonCode, next.Task = MaintenanceCode, "row/next"
	recordStop(t, next)
	read()
	resume := event
	resume.Kind, resume.ReasonCode, resume.Detail, resume.Previous = ControlResume, ResumeCode, "continue", stop.ID
	recordStop(t, resume)
	written, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(written), "user-stop") || !strings.Contains(string(written), `"reason_code": "resume"`) {
		t.Fatalf("the resume wrote %s, want the current codes alone", written)
	}
	resumed, err := ReadStop(t.Context(), root, orchestrationHead, ExecutionInteractive)
	if err != nil || resumed.State != stopResumed || resumed.Blocked {
		t.Fatalf("the resumed stop reads as %+v (%v)", resumed, err)
	}
	if resumed.RequireExecution(grant.Lane, grant.Worker, ExecutionInteractive, grant.Task, issued.ID.String()) == nil {
		t.Fatal("the resume left the grant it retired admitting its work")
	}
}
