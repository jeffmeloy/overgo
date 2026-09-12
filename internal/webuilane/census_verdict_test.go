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

// TestFailureLinesNameEachFailedTestsLastLine pins the failure report: a
// failed test is named with the last line its source wrote and that
// line's continuation, a passing test contributes nothing, and a test's
// lines never leak into the next.
func TestFailureLinesNameEachFailedTestsLastLine(t *testing.T) {
	output := strings.Join([]string{
		"=== RUN   TestWebUIBrowserFirstRun",
		"    webui_browser_firstrun_test.go:62: first-run journey: model a.gguf",
		"2026/09/10 12:37:27 serving model on http://127.0.0.1:1",
		"    webui_browser_firstrun_test.go:264: the rail shows the new title: webui lane: browser predicate did not become true",
		`        context deadline exceeded; page: {"editor":true}`,
		"--- FAIL: TestWebUIBrowserFirstRun (213.13s)",
		"=== RUN   TestWebUIBrowserScreens",
		"    webui_browser_screens_test.go:62: screens leg: captured 104 states",
		"--- PASS: TestWebUIBrowserScreens (4.36s)",
		"=== RUN   TestWebUIBrowserLayoutAudit",
		"--- FAIL: TestWebUIBrowserLayoutAudit (0.01s)",
		"FAIL",
	}, "\n")
	failures := FailureLines(output)
	want := []string{
		`TestWebUIBrowserFirstRun: webui_browser_firstrun_test.go:264: the rail shows the new title: webui lane: browser predicate did not become true context deadline exceeded; page: {"editor":true}`,
		"TestWebUIBrowserLayoutAudit: ",
	}
	if len(failures) != len(want) || failures[0] != want[0] || failures[1] != want[1] {
		t.Fatalf("failures = %q, want %q", failures, want)
	}
	if lines := FailureLines("=== RUN   TestA\n--- PASS: TestA (0.01s)\n"); len(lines) != 0 {
		t.Fatalf("a passing run reported %q", lines)
	}
}
