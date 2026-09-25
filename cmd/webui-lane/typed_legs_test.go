package main

import (
	"encoding/json"
	"strings"
	"testing"

	"overgo/internal/webuilane"
)

// TestLaneJudgesTypedLegOutcomes holds the lane to judging a run by its
// typed events and leg records, not by the words its tests log: a required
// leg counts only when the test that recorded it passed, whatever the log
// says; a named test that skipped or a run without a pass proves nothing;
// a failed run names each failed test with its failing step and the tests
// that never ended; and the tests' output reaches the lane's own output.
func TestLaneJudgesTypedLegOutcomes(t *testing.T) {
	t.Parallel()
	events := func(lines ...webuilane.TestEvent) string {
		var stream strings.Builder
		for _, event := range lines {
			encoded, err := json.Marshal(event)
			if err != nil {
				t.Fatal(err)
			}
			stream.Write(encoded)
			stream.WriteByte('\n')
		}
		return stream.String()
	}
	stream := events(
		webuilane.TestEvent{Action: "run", Test: "TestWebUIBrowserPasses"},
		webuilane.TestEvent{Action: "output", Test: "TestWebUIBrowserPasses", Output: "    passes_test.go:9: remembered settings leg: kept\n"},
		webuilane.TestEvent{Action: "pass", Test: "TestWebUIBrowserPasses"},
		webuilane.TestEvent{Action: "run", Test: "TestWebUIBrowserFails"},
		webuilane.TestEvent{Action: "output", Test: "TestWebUIBrowserFails", Output: "    fails_test.go:12: approve in place leg: decided\n"},
		webuilane.TestEvent{Action: "output", Test: "TestWebUIBrowserFails", Output: "    fails_test.go:20: pending row still shown\n"},
		webuilane.TestEvent{Action: "output", Test: "TestWebUIBrowserFails", Output: "        page: inbox\n"},
		webuilane.TestEvent{Action: "fail", Test: "TestWebUIBrowserFails"},
		webuilane.TestEvent{Action: "run", Test: "TestWebUIBrowserHangs"},
	)
	var passedOn strings.Builder
	run, err := webuilane.ReadTestRun(strings.NewReader(stream), &passedOn)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(passedOn.String(), "remembered settings leg: kept") || !strings.Contains(passedOn.String(), "page: inbox") {
		t.Fatalf("output was not passed on: %q", passedOn.String())
	}
	legs := []webuilane.LegRecord{
		{Test: "TestWebUIBrowserPasses", Leg: "remembered settings leg"},
		{Test: "TestWebUIBrowserFails", Leg: "approve in place leg"},
	}
	if err := webuilane.LaneVerdict(run, legs, []string{"remembered settings leg"}); err != nil {
		t.Fatalf("a leg a passing test recorded was refused: %v", err)
	}
	if err := webuilane.LaneVerdict(run, legs, []string{"approve in place leg"}); err == nil {
		t.Fatal("a leg recorded by a failed test was credited")
	}
	if err := webuilane.LaneVerdict(run, nil, []string{"remembered settings leg"}); err == nil {
		t.Fatal("a leg only logged, never recorded, was credited")
	}
	failures := run.Failures()
	if !strings.Contains(failures, "TestWebUIBrowserFails: fails_test.go:20: pending row still shown page: inbox") ||
		!strings.Contains(failures, "unfinished=[TestWebUIBrowserHangs]") || strings.Contains(failures, "TestWebUIBrowserPasses") {
		t.Fatalf("failures = %s", failures)
	}

	skipped, err := webuilane.ReadTestRun(strings.NewReader(events(
		webuilane.TestEvent{Action: "run", Test: "TestWebUIBrowserPasses"},
		webuilane.TestEvent{Action: "pass", Test: "TestWebUIBrowserPasses"},
		webuilane.TestEvent{Action: "run", Test: "TestWebUIBrowserSkips"},
		webuilane.TestEvent{Action: "skip", Test: "TestWebUIBrowserSkips"},
	)), &strings.Builder{})
	if err != nil {
		t.Fatal(err)
	}
	if err := webuilane.LaneVerdict(skipped, nil, nil); err == nil || !strings.Contains(err.Error(), "TestWebUIBrowserSkips was skipped") {
		t.Fatalf("a skipped test was accepted: %v", err)
	}
	empty, err := webuilane.ReadTestRun(strings.NewReader(""), &strings.Builder{})
	if err != nil {
		t.Fatal(err)
	}
	if err := webuilane.LaneVerdict(empty, nil, nil); err == nil || !strings.Contains(err.Error(), "no browser test passed") {
		t.Fatalf("a run without a pass was accepted: %v", err)
	}
}
