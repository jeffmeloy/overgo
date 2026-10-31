package main

import (
	"bytes"
	"strings"
	"testing"
	"time"

	"overgo/internal/testevidence"
)

// TestALaneTestPastTwiceItsMedianIsReportedStalled holds a lane's go test run
// to saying where its hours go: a test that took a minute or more is reported
// with its wall when it passes and the wall is kept, its newest five; a test
// run again is reported stalled once it runs past twice the median of its kept
// walls, and its report is withdrawn when it ends first; a test with no kept
// wall is held to none, and a short test is neither reported nor kept. The
// walls kept are read back by the next run in the same checkout.
func TestALaneTestPastTwiceItsMedianIsReportedStalled(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	var out bytes.Buffer
	watch := newLaneWatch(root, &out)
	const key = "overgo/internal/evaluation TestKQuantProductsHoldHeldOut"
	watch.walls[key] = []float64{3600, 3900, 4200}
	var waits []time.Duration
	var reports []func()
	stopped := 0
	watch.after = func(wait time.Duration, report func()) func() bool {
		waits, reports = append(waits, wait), append(reports, report)
		return func() bool { stopped++; return true }
	}
	test := func(name, action string, elapsed time.Duration) testevidence.TestProgress {
		return testevidence.TestProgress{Package: "overgo/internal/evaluation", Test: name, Action: action, Elapsed: elapsed}
	}

	watch.observe(test("TestKQuantProductsHoldHeldOut", "run", 0))
	if len(waits) != 1 || waits[0] != 2*3900*time.Second {
		t.Fatalf("a test whose median wall is 65m is watched for %v, want 2h10m", waits)
	}
	reports[0]()
	if !strings.Contains(out.String(), "TestKQuantProductsHoldHeldOut has run") || !strings.Contains(out.String(), "past 2 times its median wall of 1h5m0s: stalled") {
		t.Fatalf("a test past twice its median was not reported stalled:\n%s", out.String())
	}
	watch.observe(test("TestKQuantProductsHoldHeldOut", "pass", 70*time.Minute))
	if stopped != 1 || !strings.Contains(out.String(), "pass\tovergo/internal/evaluation\tTestKQuantProductsHoldHeldOut\t1h10m0s") {
		t.Fatalf("a passing long test's report was not withdrawn (%d) or its wall not reported:\n%s", stopped, out.String())
	}
	for range 4 {
		watch.observe(test("TestKQuantProductsHoldHeldOut", "pass", time.Hour))
	}
	if walls := watch.walls[key]; len(walls) != recordedWalls || walls[0] != 4200 || walls[recordedWalls-1] != 3600 {
		t.Fatalf("kept walls = %v, want the newest five", walls)
	}

	watch.observe(test("TestQwen35CellsHoldHeldOut", "run", 0))
	watch.observe(test("TestQwen35CellsHoldHeldOut", "pass", 2*time.Hour))
	watch.observe(test("TestPlanBinding", "run", 0))
	watch.observe(test("TestPlanBinding", "pass", time.Second))
	if len(waits) != 1 {
		t.Fatalf("a test with no kept wall was watched: %v", waits)
	}
	if _, kept := watch.walls["overgo/internal/evaluation TestPlanBinding"]; kept || strings.Contains(out.String(), "TestPlanBinding") {
		t.Fatal("a short test was kept or reported")
	}
	watch.close()
	if read := readLaneWalls(root); len(read[key]) != recordedWalls || len(read["overgo/internal/evaluation TestQwen35CellsHoldHeldOut"]) != 1 {
		t.Fatalf("the next run reads walls %v", read)
	}
}
