package main

import (
	"fmt"
	"io"
	"os"
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
}

// newLaneWatch watches a run, reporting to output.
func newLaneWatch(output io.Writer) *laneWatch {
	return &laneWatch{output: output, record: os.Getenv(clioptions.MeasurementRunEnvironment)}
}

// observe takes one top-level test's start or end (testevidence.GoTestOptions.Tests).
func (w *laneWatch) observe(progress testevidence.TestProgress) {
	key := progress.Package + " " + progress.Test
	w.mu.Lock()
	defer w.mu.Unlock()
	switch {
	case progress.Action == "run":
		fmt.Fprintf(w.output, "run\t%s\t%s\tsince %s\n", progress.Package, progress.Test, time.Now().Format(time.TimeOnly))
	case progress.Action == "fail":
		fmt.Fprintf(w.output, "%s\t%s\t%s\t%s\n", progress.Action, progress.Package, progress.Test, progress.Elapsed.Round(time.Second))
		fmt.Fprintf(w.output, "validate -run: %s failed:\n%s\n", key, strings.TrimRight(progress.Output, "\n"))
		if w.record != "" {
			if err := plan.RecordFailure(w.record, key); err != nil {
				fmt.Fprintf(w.output, "validate -run: the run record %s does not name %s: %v\n", w.record, key, err)
			}
		}
	case progress.Action == "pass" && progress.Elapsed >= reportedTestWall:
		fmt.Fprintf(w.output, "%s\t%s\t%s\t%s\n", progress.Action, progress.Package, progress.Test, progress.Elapsed.Round(time.Second))
	}
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
