package main

import (
	"io"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	"overgo/internal/jsonfile"
	"overgo/internal/plan"
	"overgo/internal/processlock"
)

// TestTheHooksPrintTheRunsEnd holds the prompt hook and the stop gate's block
// to printing the newest measurement run's record, so a run's end and times are
// read and never reckoned: a run that ended says when, each phase's wall and
// the rows it filed; a running one its phase's expected end; and one whose
// process is gone with no end recorded, that it was cut short.
func TestTheHooksPrintTheRunsEnd(t *testing.T) {
	sessionCheckout(t)
	if err := processlock.EnsureStateDirectory("."); err != nil {
		t.Fatal(err)
	}
	keep := func(run plan.MeasurementRun) {
		t.Helper()
		if err := jsonfile.Write(plan.MeasurementRunPath("."), run, stateFileMode); err != nil {
			t.Fatal(err)
		}
	}
	captured := func(stream **os.File, hook func()) string {
		t.Helper()
		read, written, err := os.Pipe()
		if err != nil {
			t.Fatal(err)
		}
		saved := *stream
		*stream = written
		hook()
		*stream = saved
		_ = written.Close()
		printed, err := io.ReadAll(read)
		if err != nil {
			t.Fatal(err)
		}
		return string(printed)
	}
	prompted := func() string {
		return captured(&os.Stdout, func() { runPrompt(`{"session_id": "lead", "prompt": "progress?"}`) })
	}
	at := func(moment time.Time) string { return moment.Local().Format("Jan 2 15:04 MST") }

	began := time.Date(2026, 10, 30, 2, 21, 0, 0, time.UTC)
	lanes := plan.MeasurementPhase{Name: "lanes", Started: began, Ended: began.Add(5*time.Hour + 23*time.Minute), Passed: true}
	keep(plan.MeasurementRun{Commit: "bc1d47e13606fddb", Process: os.Getpid(), Started: began, Phases: []plan.MeasurementPhase{lanes}, Ended: began.Add(6 * time.Hour), Filed: 3})
	if printed := prompted(); !strings.Contains(printed, "began "+at(began)) || !strings.Contains(printed, "lanes passed in 5h23m") ||
		!strings.Contains(printed, "ended "+at(began.Add(6*time.Hour))) || !strings.Contains(printed, "filing 3 rows") {
		t.Fatalf("the prompt hook printed %q for a run that ended", printed)
	}

	cells := plan.MeasurementPhase{Name: "cells", Started: lanes.Ended, Expected: lanes.Ended.Add(2*time.Hour + 38*time.Minute)}
	keep(plan.MeasurementRun{Commit: "bc1d47e13606fddb", Process: os.Getpid(), Started: began, Phases: []plan.MeasurementPhase{lanes, cells}})
	want := "cells running since " + at(cells.Started) + ", expected to end " + at(cells.Expected)
	if printed := prompted(); !strings.Contains(printed, want) || !strings.Contains(printed, "the checkout free") {
		t.Fatalf("the prompt hook printed %q for a running run, want %q", printed, want)
	}
	if blocked := captured(&os.Stderr, func() { stopHook("lead") }); !strings.Contains(blocked, want) {
		t.Fatalf("the stop gate's block printed %q, want the run's %q", blocked, want)
	}

	finished := exec.CommandContext(t.Context(), "git", "--version")
	if err := finished.Run(); err != nil {
		t.Fatal(err)
	}
	keep(plan.MeasurementRun{Commit: "bc1d47e13606fddb", Process: finished.Process.Pid, Started: began, Phases: []plan.MeasurementPhase{lanes, cells}})
	if printed := prompted(); !strings.Contains(printed, "the run was cut short") {
		t.Fatalf("the prompt hook printed %q for a run whose process is gone", printed)
	}
}
