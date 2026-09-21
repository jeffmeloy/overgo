package gate

import (
	"errors"
	"strings"
	"testing"
	"unicode/utf8"

	"overgo/internal/automationcheck"
	"overgo/internal/clioptions"
	"overgo/internal/runrecord"
)

// TestFailedLaneNamesItsTest holds a failed step to keeping what it said. A
// lane's error ends with the tail of its output, which names the test that
// failed or was running when it hung; the recorded step keeps that end,
// bounded by the diagnostic tail and cut on a character, where it used to
// keep only the lane's name. A step that passed records nothing.
func TestFailedLaneNamesItsTest(t *testing.T) {
	t.Parallel()
	const named = "panic: test timed out after 20m0s\n\trunning tests:\n\t\tTestWebUIBrowserVideoCapture (19m58s)"
	failure := errors.New(strings.Repeat("é selection noise ", clioptions.DiagnosticTailBytes) + named)
	failed := gateEvidenceRecord("webui-lane", runrecord.PhaseTest, automationcheck.Evidence{}, failure, "")
	if failed.Outcome != runrecord.StepFailed || !strings.HasSuffix(failed.Detail, named) {
		t.Fatalf("failed step = %+v, want it to end with what the lane said", failed)
	}
	if len(failed.Detail) > clioptions.DiagnosticTailBytes || !utf8.ValidString(failed.Detail) {
		t.Fatalf("detail is %d bytes, valid=%v; want it bounded by %d and cut on a character", len(failed.Detail), utf8.ValidString(failed.Detail), clioptions.DiagnosticTailBytes)
	}
	if passed := gateEvidenceRecord("webui-lane", runrecord.PhaseTest, automationcheck.Evidence{}, nil, ""); passed.Detail != "" {
		t.Fatalf("a step that passed recorded %q", passed.Detail)
	}
}
