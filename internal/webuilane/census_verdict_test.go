package webuilane

import (
	"strings"
	"testing"
)

// TestLaneVerdictRefusesSkipsAndMissingEvidence pins the lane's verdict:
// a skipped test or a run without a pass is refused, a required journey
// line must have been written, and a complete run with its lines passes.
func TestLaneVerdictRefusesSkipsAndMissingEvidence(t *testing.T) {
	passed := strings.Join([]string{
		"=== RUN   TestWebUIBrowserFirstRun",
		"    webui_browser_firstrun_test.go:214: cold-start leg: the picker served a.gguf from the cold page and the front page followed",
		"--- PASS: TestWebUIBrowserFirstRun (1.00s)",
		"webui lane: PASS browser=chrome",
	}, "\n")
	if err := LaneVerdict(passed, []string{"cold-start leg: the picker served"}); err != nil {
		t.Fatalf("complete run refused: %v", err)
	}
	if err := LaneVerdict(passed, []string{"vision leg"}); err == nil || !strings.Contains(err.Error(), `"vision leg"`) {
		t.Fatalf("missing evidence accepted: %v", err)
	}
	skipped := "=== RUN   TestWebUIBrowserFirstRun\n--- SKIP: TestWebUIBrowserFirstRun (0.00s)\n--- PASS: TestWebUIBrowserAcceptance (0.50s)\n"
	if err := LaneVerdict(skipped, nil); err == nil || !strings.Contains(err.Error(), "TestWebUIBrowserFirstRun (0.00s) was skipped") {
		t.Fatalf("skipped test accepted: %v", err)
	}
	if err := LaneVerdict("webui lane: PASS browser=chrome\n", nil); err == nil || !strings.Contains(err.Error(), "no browser test passed") {
		t.Fatalf("empty run accepted: %v", err)
	}
}
