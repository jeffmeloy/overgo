package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	"overgo/internal/clioptions"
	"overgo/internal/jsonfile"
	"overgo/internal/plan"
	"overgo/internal/processlock"
	"overgo/internal/tensorstats"
	"overgo/internal/testevidence"
)

// laneWallsFile keeps, in the process state of the checkout validate runs in, the walls of the lane tests that took
// reportedTestWall or more: each test's newest recordedWalls, which judge whether the test, run again, has stalled. On
// 2026-10-30 internal/evaluation's model tests took 3h58m of the models lane's 5h23m, and nothing said where.
const laneWallsFile = "lane_walls.json"

const (
	// recordedWalls is how many of a test's newest walls are kept and judge it: a median of five is moved by no one
	// reading taken beside other work.
	recordedWalls = 5
	// reportedTestWall is the shortest wall a lane test's is recorded and reported at: shorter tests are the go test
	// deadline's to bound, and the models lane runs thousands.
	reportedTestWall = time.Minute
	// stallMultiple is how many times its median wall a test runs before it is reported stalled: a model test's
	// walls on this machine stand within half of one another between runs.
	stallMultiple = 2
	// medianQuantile is the quantile a median is.
	medianQuantile = 0.5
	// skippedAfterFailure is the wall, in the runs before, past which a lane's queued package is skipped once a lane
	// test has failed: a failed lane runs no cell, and on 2026-10-31 internal/evaluation's held-out tests, 3h58m of
	// the models lane's 5h23m the run before, were still to run after cmd/validate failed in its first minute; each
	// other package of the lane took under 15 minutes.
	skippedAfterFailure = 15 * time.Minute
)

// laneWalls are lane tests' recorded walls in seconds, newest last, by package and test.
type laneWalls map[string][]float64

// readLaneWalls reads the walls kept for the checkout at root; none where none are kept.
func readLaneWalls(root string) laneWalls {
	walls := laneWalls{}
	if jsonfile.Decode(filepath.Join(root, processlock.StateDirectory, laneWallsFile), &walls) != nil {
		return laneWalls{}
	}
	return walls
}

// median is the median of a test's recorded walls, and whether any are recorded.
func (walls laneWalls) median(key string) (time.Duration, bool) {
	recorded := slices.Sorted(slices.Values(walls[key]))
	if len(recorded) == 0 {
		return 0, false
	}
	return time.Duration(tensorstats.Quantile(recorded, medianQuantile) * float64(time.Second)), true
}

// packageWall is the wall the lane tests of the package a command names as ./directory took in the runs before: the
// sum of their newest recorded walls.
func (walls laneWalls) packageWall(pkg string) time.Duration {
	suffix := "/" + strings.TrimPrefix(pkg, "./")
	var wall float64
	for key, recorded := range walls {
		name, _, _ := strings.Cut(key, " ")
		if strings.HasSuffix(name, suffix) && len(recorded) > 0 {
			wall += recorded[len(recorded)-1]
		}
	}
	return time.Duration(wall * float64(time.Second))
}

// laneWatch watches a lane's go test run: it reports a test running past stallMultiple times its median wall while
// it runs, reports and records each test that took reportedTestWall or more when it passes, and reports each test
// that fails, with its output, when it fails, adding it to the run record at record where a measurement run gives
// one (clioptions.MeasurementRunEnvironment).
type laneWatch struct {
	root     string
	output   io.Writer
	walls    laneWalls
	record   string
	after    func(time.Duration, func()) func() bool
	mu       sync.Mutex
	running  map[string]func() bool
	failures []string
}

// newLaneWatch watches a run in the checkout at root, reporting to output.
func newLaneWatch(root string, output io.Writer) *laneWatch {
	return &laneWatch{root: root, output: output, walls: readLaneWalls(root), record: os.Getenv(clioptions.MeasurementRunEnvironment),
		running: map[string]func() bool{},
		after:   func(wait time.Duration, report func()) func() bool { return time.AfterFunc(wait, report).Stop }}
}

// long is whether the lane tests of the package a command names as ./directory took skippedAfterFailure or more in
// the runs before.
func (w *laneWatch) long(pkg string) bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.walls.packageWall(pkg) >= skippedAfterFailure
}

// failed is each test that has failed in the run, as package and test.
func (w *laneWatch) failed() []string {
	w.mu.Lock()
	defer w.mu.Unlock()
	return slices.Clone(w.failures)
}

// observe takes one top-level test's start or end (testevidence.GoTestOptions.Tests).
func (w *laneWatch) observe(progress testevidence.TestProgress) {
	key := progress.Package + " " + progress.Test
	w.mu.Lock()
	defer w.mu.Unlock()
	if progress.Action == "run" {
		if median, timed := w.walls.median(key); timed {
			began := time.Now()
			w.running[key] = w.after(stallMultiple*median, func() {
				w.mu.Lock()
				defer w.mu.Unlock()
				fmt.Fprintf(w.output, "validate -run: %s %s has run %s since %s, past %d times its median wall of %s: stalled\n",
					progress.Package, progress.Test, time.Since(began).Round(time.Second), began.Format(time.TimeOnly), stallMultiple, median.Round(time.Second))
			})
		}
		return
	}
	if stop := w.running[key]; stop != nil {
		stop()
		delete(w.running, key)
	}
	if progress.Action == "fail" {
		w.failures = append(w.failures, key)
		fmt.Fprintf(w.output, "%s\t%s\t%s\t%s\n", progress.Action, progress.Package, progress.Test, progress.Elapsed.Round(time.Second))
		fmt.Fprintf(w.output, "validate -run: %s failed:\n%s\n", key, strings.TrimRight(progress.Output, "\n"))
		if w.record != "" {
			if err := plan.RecordFailure(w.record, key); err != nil {
				fmt.Fprintf(w.output, "validate -run: the run record %s does not name %s: %v\n", w.record, key, err)
			}
		}
	}
	if progress.Action == "pass" && progress.Elapsed >= reportedTestWall {
		fmt.Fprintf(w.output, "%s\t%s\t%s\t%s\n", progress.Action, progress.Package, progress.Test, progress.Elapsed.Round(time.Second))
		recorded := append(w.walls[key], progress.Elapsed.Seconds())
		w.walls[key] = recorded[max(len(recorded)-recordedWalls, 0):]
	}
}

// close stops the reports of tests still running and keeps the walls recorded; a wall that cannot be kept changes
// no verdict.
func (w *laneWatch) close() {
	w.mu.Lock()
	defer w.mu.Unlock()
	for key, stop := range w.running {
		stop()
		delete(w.running, key)
	}
	if processlock.EnsureStateDirectory(w.root) == nil {
		_ = jsonfile.Write(filepath.Join(w.root, processlock.StateDirectory, laneWallsFile), w.walls, clioptions.PrivateFileMode)
	}
}
