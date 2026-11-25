package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"runtime/debug"
	"slices"
	"strings"
	"time"

	"overgo/internal/clioptions"
	"overgo/internal/gitauthority"
	"overgo/internal/jsonfile"
	"overgo/internal/plan"
	"overgo/internal/processcontrol"
	"overgo/internal/processlock"
)

// A row's closing commit runs the cycle's lanes at that commit (lanes-on-close): the landing that completes a row's last
// open step names its commit the one the lanes run at next and starts a runner of them, which runs them in a checkout of
// their own under the trunk's process state, so the working folder stays free, then again at the newest commit that
// closed a row while they ran, until none is newer. A close while a run goes waits for it rather than end it: the lanes
// take turns with each commit (lanes-turns), so a run can outlast the time between two closes, and a run each close
// overtook would never end. Two of the five lanes had failed at 1ab2ac601, 66 commits after their last run, since
// nothing ran them while the plan held rows. A run runs only what the changes since each lane's last pass reach
// (validate -reached): running every lane on every close held the card from 11:57 to 19:30 on 2026-11-18 (lanes-reach).
const (
	// lanesCheckout is the detached worktree the lanes run in on a row's close, and lanesLock the lock a runner holds
	// while it runs them there.
	lanesCheckout = "lanes"
	lanesLock     = lanesCheckout + ".lock"
	// lanesDueFile keeps the newest commit that closed a row, the commit the lanes run at next, which a closing landing
	// writes, and lanesBeganFile the commit a runner last began them at, which only the runner holding the lock writes.
	lanesDueFile   = "lanes_due.json"
	lanesBeganFile = "lanes_began.json"
	// lanesLogFile is where a runner says what it runs and how each run ended.
	lanesLogFile = "lanes.log"
	// lanesRunnerPrefix begins the directory each runner's program is built into.
	lanesRunnerPrefix = "lanes-runner-"
)

// lanesDue is the newest commit that closed a row, and the rows it closed.
type lanesDue struct {
	Commit string   `json:"commit"`
	Rows   []string `json:"rows"`
}

// closedRows are the rows of reference that commit's plan no longer holds: the landing that completes a row's last open
// step removes the row (plan.Item).
func closedRows(root, reference, commit string) ([]string, error) {
	published, err := gitauthority.Query(context.Background(), root, "show", commit+":"+plan.Path)
	if err != nil {
		return nil, err
	}
	document, err := plan.Parse(published)
	if err != nil {
		return nil, err
	}
	var closed []string
	for row := range strings.SplitSeq(reference, ",") {
		item, _, _ := strings.Cut(strings.TrimSpace(row), "/")
		if !slices.ContainsFunc(document.Items, func(open plan.Item) bool { return open.ID == item }) && !slices.Contains(closed, item) {
			closed = append(closed, item)
		}
	}
	return closed, nil
}

// lanesOnClose names commit, where it closed a row of reference, the commit the lanes run at next, and starts a runner
// with start where none holds the lanes; it says where and when they run, nothing where no row closed. The commit is
// named before the lock is looked at, and a runner reads the commit due again once it lets the lock go (runLanesDue),
// so a close that finds the lock held is run by the runner that held it, or by one it started.
func lanesOnClose(root, reference, commit string, start func() error) (string, error) {
	closed, err := closedRows(root, reference, commit)
	if err != nil || len(closed) == 0 {
		return "", err
	}
	if err := processlock.EnsureStateDirectory(root); err != nil {
		return "", err
	}
	state := filepath.Join(root, processlock.StateDirectory)
	if err := jsonfile.Write(filepath.Join(state, lanesDueFile), lanesDue{Commit: commit, Rows: closed}, clioptions.PrivateFileMode); err != nil {
		return "", err
	}
	said := fmt.Sprintf("%s closed: the cycle's lanes run at %.12s in %s", strings.Join(closed, ", "), commit, filepath.Join(state, lanesCheckout))
	lock, err := processlock.Acquire(filepath.Join(state, lanesLock), clioptions.PrivateFileMode)
	if errors.Is(err, processlock.ErrBusy) {
		return said + " once the run going there ends, said in " + filepath.Join(state, lanesLogFile), nil
	}
	if err != nil {
		return "", err
	}
	if err := lock.Close(); err != nil {
		return "", err
	}
	return said + ", said in " + filepath.Join(state, lanesLogFile), start()
}

// startLanesRunner starts the lanes runner (runLanesDue) as a process that outlives the landing
// (processcontrol.StartDetached), built from the trunk's commit (buildChecker) into a directory of its own under the
// process state, in the environment a landing's tools run in (landingEnvironment), its output appended to the lanes log.
func startLanesRunner() error {
	state, err := filepath.Abs(filepath.Join(loopWorktree, processlock.StateDirectory))
	if err != nil {
		return err
	}
	// Windows keeps a running program's file: the programs of runners that ended are removed here, and one still
	// running is left for a later start to remove.
	built, err := filepath.Glob(filepath.Join(state, lanesRunnerPrefix+"*"))
	if err != nil {
		return err
	}
	for _, directory := range built {
		_ = os.RemoveAll(directory)
	}
	directory, err := os.MkdirTemp(state, lanesRunnerPrefix)
	if err != nil {
		return err
	}
	runner := filepath.Join(directory, "loop.exe")
	var said bytes.Buffer
	if refusal, err := buildChecker(loopWorktree, runner, "./cmd/loop", func(arguments ...string) (string, error) {
		err := runToolEnv(os.Environ(), nil, io.Discard, &said, arguments...)
		return strings.TrimSpace(said.String()), err
	}); err != nil {
		return fmt.Errorf("the lanes runner does not build: %s", refusal)
	}
	configured, err := configuredEnvironment()
	if err != nil {
		return err
	}
	log, err := os.OpenFile(filepath.Join(state, lanesLogFile), os.O_APPEND|os.O_CREATE|os.O_WRONLY, clioptions.PrivateFileMode)
	if err != nil {
		return err
	}
	trunk, err := filepath.Abs(loopWorktree)
	if err == nil {
		_, err = processcontrol.StartDetached(processcontrol.Command{
			Path: runner, Args: []string{"-lanes-due"}, Dir: trunk, Env: landingEnvironment(os.Environ(), configured), Stdout: log, Stderr: log,
		})
	}
	return errors.Join(err, log.Close())
}

// runLanesDue runs the cycle's lanes with run at the commit due (lanesDue), in the lanes checkout moved there, once. A
// commit named due while it ran is run by the runner next starts, built from the trunk (startLanesRunner), once this
// one lets the lanes go: a runner that ran each later commit itself supervised every run with its own build, and the
// one built at f46760944 ended the runs at d0eaf0f9a and 9b40546f8 on 2026-11-20 at the bound verdict-steady had
// replaced (lanes-runner-current). Where another runner holds the lanes it runs nothing: that runner starts the next
// once its run ends. A run that fails is said, and the next commit due still runs.
func runLanesDue(root string, output io.Writer, run func(checkout string) error, next func() error) error {
	state := filepath.Join(root, processlock.StateDirectory)
	checkout, err := filepath.Abs(filepath.Join(state, lanesCheckout))
	if err != nil {
		return err
	}
	lock, err := processlock.Acquire(filepath.Join(state, lanesLock), clioptions.PrivateFileMode)
	if errors.Is(err, processlock.ErrBusy) {
		fmt.Fprintln(output, "loop: another runner holds the lanes, and starts the next for the commit due")
		return nil
	}
	if err != nil {
		return err
	}
	due, began, err := lanesRecords(state)
	if err == nil && due.Commit != began.Commit {
		// The commit is begun once its checkout is set up, so a setup that fails leaves it due for the next runner
		// (SQA-496: begun before its setup, a commit whose local-models.json was missing read as run once repaired, and
		// no lane ran it).
		err = runLanesAt(root, checkout, due, lanesRunnerBuild(), output, run, func() error {
			return jsonfile.Write(filepath.Join(state, lanesBeganFile), due, clioptions.PrivateFileMode)
		})
	}
	if err = errors.Join(err, lock.Close()); err != nil {
		return err
	}
	// A close that named its commit while the lock was held left its run to the runner this one starts.
	if due, began, err := lanesRecords(state); err != nil || due.Commit == began.Commit {
		return err
	}
	fmt.Fprintln(output, "loop: a later commit is due; the runner built from the trunk runs it")
	return next()
}

// lanesRecords reads the commit due and the one a runner last began the lanes at, none where none began: a runner
// started as another let the lock go, after it ran the commit due, ran it again where it kept what it ran to itself.
func lanesRecords(state string) (due, began lanesDue, err error) {
	err = jsonfile.Decode(filepath.Join(state, lanesDueFile), &due)
	if beganErr := jsonfile.Decode(filepath.Join(state, lanesBeganFile), &began); !errors.Is(beganErr, fs.ErrNotExist) {
		err = errors.Join(err, beganErr)
	}
	return due, began, err
}

// lanesRunnerBuild is the build of the runner supervising the lanes, as its first line names it: the commit it was
// built at, with its tree's changes where the build stamped any, or an unstamped build. A runner built before the lanes'
// code changed runs that code still, so a run's verdict is its runner's build's (lanes-trunk: the runner built at
// 28390a169 ran three later commits under the hand-off and bound they had replaced).
func lanesRunnerBuild() string {
	info, _ := debug.ReadBuildInfo()
	return describeBuild(builtCommit, info)
}

// builtCommit is the commit buildChecker built this program from, in a checkout of that commit alone; empty for a build
// made otherwise.
var builtCommit string

// describeBuild names a build: the commit buildChecker built it from, clean, where built names one; otherwise the
// commit go stamped and whether the tree it found held changes; otherwise an unstamped build. A build buildChecker made
// is named by its commit alone, since go stamps it from the main checkout above its worktree (runner-from-commit).
func describeBuild(built string, info *debug.BuildInfo) string {
	if built != "" {
		return fmt.Sprintf("%.12s", built)
	}
	revision, modified := "", false
	if info != nil {
		for _, setting := range info.Settings {
			switch setting.Key {
			case "vcs.revision":
				revision = setting.Value
			case "vcs.modified":
				modified = setting.Value == "true"
			}
		}
	}
	switch {
	case revision == "":
		return "an unstamped build"
	case modified:
		return fmt.Sprintf("%.12s with its tree's changes", revision)
	}
	return fmt.Sprintf("%.12s", revision)
}

// runLanesAt moves checkout to the commit due with the data roots (moveWithRoots) and runs the lanes there with run,
// saying when they began, under the runner's build (lanesRunnerBuild), and how they ended; only a checkout that cannot
// be moved fails it. A run that yielded to a newer commit due between packages, as its record beside the commit due
// says, is said to have yielded, its passed packages kept, rather than to have failed (lanes-first/yield). begun, where
// given, records the run begun once the checkout is set up, before the lanes run.
func runLanesAt(root, checkout string, due lanesDue, runner string, output io.Writer, run func(checkout string) error, begun func() error) error {
	if err := moveWithRoots(root, checkout, due.Commit); err != nil {
		return err
	}
	if begun != nil {
		if err := begun(); err != nil {
			return err
		}
	}
	yieldedRecord := filepath.Join(root, processlock.StateDirectory, plan.LanesYieldedFile)
	if err := os.Remove(yieldedRecord); err != nil && !errors.Is(err, fs.ErrNotExist) {
		return err
	}
	began := time.Now()
	fmt.Fprintf(output, "loop: %s the cycle's lanes run at %.12s, which closed %s, under the runner built at %s\n",
		began.Format(time.DateTime), due.Commit, strings.Join(due.Rows, ", "), runner)
	verdict := "passed"
	if err := run(checkout); err != nil {
		verdict = "did not pass: " + err.Error()
		var yielded plan.LanesYielded
		if jsonfile.Decode(yieldedRecord, &yielded) == nil && yielded.Commit == due.Commit {
			verdict = fmt.Sprintf("yielded to %.12s, a newer commit due, as a lane turned to another package, its passed packages kept", yielded.To)
			// A run that failed before it yielded did not pass: its failures are said beside the yield (SQA-437).
			if len(yielded.Failures) > 0 {
				verdict = "did not pass: " + strings.Join(yielded.Failures, "; ") + "; then " + verdict
			}
		}
	}
	fmt.Fprintf(output, "loop: %s the lanes at %.12s %s after %s\n", time.Now().Format(time.DateTime), due.Commit, verdict, time.Since(began).Round(time.Second))
	return nil
}
