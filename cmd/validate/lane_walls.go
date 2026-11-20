package main

import (
	"fmt"
	"io"
	"maps"
	"os"
	"slices"
	"strings"
	"sync"
	"time"

	"overgo/internal/clioptions"
	"overgo/internal/plan"
	"overgo/internal/testevidence"
)

// reportedTestWall is the shortest wall a passing lane test is reported at: the models lane runs thousands, and on
// 2026-10-30 internal/evaluation's model tests took 3h58m of its 5h23m with nothing saying where.
const reportedTestWall = time.Minute

// laneWatch watches a lane's go test run: it names each top-level test as it starts, so the lane's last line names the
// test that runs, reports each test that took reportedTestWall or more when it passes, and reports each test that fails,
// with its output, when it fails, adding it to the run record at record where a measurement run gives one
// (clioptions.MeasurementRunEnvironment). A stall report armed from walls an earlier run kept said nothing from 04:10 to
// 10:20 on 2026-11-14, its record empty, while a journey waited: the running test's name says where a lane stands.
type laneWatch struct {
	output io.Writer
	record string
	mu     sync.Mutex
	// open are the tests that started and have not ended, by package and name, with when each started (ended).
	open map[string]time.Time
}

// newLaneWatch watches a run, reporting to output.
func newLaneWatch(output io.Writer) *laneWatch {
	return &laneWatch{output: output, record: os.Getenv(clioptions.MeasurementRunEnvironment), open: map[string]time.Time{}}
}

// observe takes one top-level test's start or end (testevidence.GoTestOptions.Tests).
func (w *laneWatch) observe(progress testevidence.TestProgress) {
	key := progress.Package + " " + progress.Test
	w.mu.Lock()
	defer w.mu.Unlock()
	if progress.Action == "run" {
		w.open[key] = time.Now()
	} else {
		delete(w.open, key)
	}
	switch {
	case progress.Action == "run":
		fmt.Fprintf(w.output, "run\t%s\t%s\tsince %s\n", progress.Package, progress.Test, time.Now().Format(time.TimeOnly))
	case progress.Action == "fail":
		fmt.Fprintf(w.output, "%s\t%s\t%s\t%s\n", progress.Action, progress.Package, progress.Test, progress.Elapsed.Round(time.Second))
		fmt.Fprintf(w.output, "validate -run: %s failed:\n%s\n", key, strings.TrimRight(progress.Output, "\n"))
		w.recordFailure(key)
	case progress.Action == "pass" && progress.Elapsed >= reportedTestWall:
		fmt.Fprintf(w.output, "%s\t%s\t%s\t%s\n", progress.Action, progress.Package, progress.Test, progress.Elapsed.Round(time.Second))
	}
}

// recordFailure adds a failed test to the run record where a measurement run gives one.
func (w *laneWatch) recordFailure(key string) {
	if w.record == "" {
		return
	}
	if err := plan.RecordFailure(w.record, key); err != nil {
		fmt.Fprintf(w.output, "validate -run: the run record %s does not name %s: %v\n", w.record, key, err)
	}
}

// ended reports, the moment a lane's test process ends without its result, each test it was running as failed, with
// how the process ended and the most memory one of its processes committed, and records each in the run where a
// measurement run gives one (lanes-ended). A process its tree's bound ends writes no event of its own: ten tests the
// run at f46760944 ended at its bound reached the lanes log in the run's summary alone, at 20:20 on 2026-11-19, 8h32m
// after the run began, where a watch over the log read only the failures go test reported.
func (w *laneWatch) ended(err error, peak uint64) {
	w.mu.Lock()
	defer w.mu.Unlock()
	for _, key := range slices.Sorted(maps.Keys(w.open)) {
		pkg, test, _ := strings.Cut(key, " ")
		fmt.Fprintf(w.output, "fail\t%s\t%s\t%s\n", pkg, test, time.Since(w.open[key]).Round(time.Second))
		fmt.Fprintf(w.output, "validate -run: %s ended without its result, its largest process at %.1f GiB: %s\n", key,
			float64(peak)/(1<<30), strings.ReplaceAll(err.Error(), "\n", "; "))
		w.recordFailure(key)
	}
	clear(w.open)
}

// ran adds a package the lane ran, its wall and the lane's peak process memory by its end, to the run record where a
// measurement run gives one.
func (w *laneWatch) ran(done testevidence.PackageProgress) {
	if w.record == "" {
		return
	}
	if err := plan.RecordPackage(w.record, plan.PackageRun{Package: done.Package, Wall: done.Elapsed, PeakBytes: done.PeakBytes}); err != nil {
		fmt.Fprintf(w.output, "validate -run: the run record %s does not name %s: %v\n", w.record, done.Package, err)
	}
}
