package main

import (
	"bytes"
	"strings"
	"testing"
	"time"

	"overgo/internal/testevidence"
)

// TestALaneNamesEachTestAsItStarts holds a lane's go test run to saying where it stands as it goes: each top-level test
// is named when it starts, so the lane's last line names the test that runs, as a journey that waited from 04:10 to
// 10:20 on 2026-11-14 was not; a test that took a minute or more is reported with its wall when it passes, and a short
// one only by its start.
func TestALaneNamesEachTestAsItStarts(t *testing.T) {
	t.Parallel()
	var out bytes.Buffer
	watch := newLaneWatch(&out)
	test := func(name, action string, elapsed time.Duration) testevidence.TestProgress {
		return testevidence.TestProgress{Package: "overgo/internal/server", Test: name, Action: action, Elapsed: elapsed}
	}
	watch.observe(test("TestModelJourneyFirstRun", "run", 0))
	lines := strings.Split(strings.TrimSpace(out.String()), "\n")
	if last := lines[len(lines)-1]; !strings.HasPrefix(last, "run\tovergo/internal/server\tTestModelJourneyFirstRun\tsince ") {
		t.Fatalf("a lane whose journey has started ends its output with %q, want the journey named as running", last)
	}
	watch.observe(test("TestModelJourneyFirstRun", "pass", 13*time.Minute))
	watch.observe(test("TestPlanBinding", "run", 0))
	watch.observe(test("TestPlanBinding", "pass", time.Second))
	if !strings.Contains(out.String(), "pass\tovergo/internal/server\tTestModelJourneyFirstRun\t13m0s") {
		t.Fatalf("a long test's wall was not reported when it passed:\n%s", out.String())
	}
	if strings.Contains(out.String(), "pass\tovergo/internal/server\tTestPlanBinding") {
		t.Fatalf("a short test's pass was reported:\n%s", out.String())
	}
}
