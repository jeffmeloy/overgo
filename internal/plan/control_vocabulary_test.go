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

// TestControlVocabularyNamesNoPerson holds the control records' codes free of persons, as the doctrine is, and a stop
// file written in the codes they replaced to a stop that reads, and resumes, in the current ones.
func TestControlVocabularyNamesNoPerson(t *testing.T) {
	t.Parallel()
	for _, code := range append(slices.Clone(validStopReasons), ResumeCode, MaintenanceCode) {
		for part := range strings.SplitSeq(code, "-") {
			if slices.Contains([]string{"user", "operator", "owner", "human"}, part) {
				t.Errorf("control code %q names a person", code)
			}
		}
	}
	root, event := stopFixture(t)
	event.ReasonCode = "user-stop"
	data, err := json.MarshalIndent(stopState{Current: event}, "", "  ")
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
	current, err := ReadStop(t.Context(), root, orchestrationHead, ExecutionInteractive)
	if err != nil || current.State != stopActive || !current.Blocked || current.Event.ReasonCode != stopRequested {
		t.Fatalf("a stop written as user-stop reads as %+v (%v), want an active %s stop", current, err, stopRequested)
	}
	resume := event
	resume.Kind, resume.ReasonCode, resume.Detail, resume.Previous = ControlResume, ResumeCode, "continue", current.ID
	recordStop(t, resume)
	written, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(written), "user-stop") || !strings.Contains(string(written), `"reason_code": "resume"`) {
		t.Fatalf("the resume wrote %s, want the current codes alone", written)
	}
	if resumed, err := ReadStop(t.Context(), root, orchestrationHead, ExecutionInteractive); err != nil || resumed.State != stopResumed || resumed.Blocked {
		t.Fatalf("the resumed stop reads as %+v (%v)", resumed, err)
	}
}
