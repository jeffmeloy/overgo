package main

import (
	"errors"
	"io"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/loop"
	"overgo/internal/plan"
	"overgo/internal/recipe"
	"overgo/internal/scorecard"
	"overgo/internal/testutil"
)

// TestARunRecordsItsEnd runs two improvement cycles and reads the record each
// keeps of itself (plan.MeasurementRun): its commit and process, each phase as
// validate ran it with its verdict, the fill, the rows it filed and its end.
// The second run expects each phase to end after the wall the first recorded
// for it, and the record reads as the hooks print it.
func TestARunRecordsItsEnd(t *testing.T) {
	t.Parallel()
	repository := t.TempDir()
	gitFixture(t, repository, "init", "-q", "-b", "master")
	gitFixture(t, repository, "commit", "-q", "--allow-empty", "-m", "measured")
	measured := gitFixture(t, repository, "rev-parse", "HEAD")
	source := testutil.ArtifactID(t, artifact.KindEvidence, "loop-run-record")
	records := filepath.Join(t.TempDir(), "scorecards")
	published, accuracy := time.Date(2026, time.October, 30, 12, 0, 0, 0, time.UTC), 0.5
	run := &runRecorder{root: repository}
	improver := improvement{root: repository, plan: emptyPlan(t), records: records, run: run, fill: func(string) error { return nil }}
	improver.measure = func(_ string, filed func() error) error {
		return measurePhases(filepath.Join(t.TempDir(), cycleStateFile), "commit|store|obligations", io.Discard, filed, run, func(_ []string, arguments ...string) error {
			switch strings.Join(arguments, " ") {
			case "-run -cells":
				return errors.New("a cell refused its measurement")
			case "-scorecard -publish":
				published = published.Add(time.Hour)
				_, err := scorecard.Publish(records, published, scorecard.Record{Commit: measured, Rows: []scorecard.Row{
					scoredRow(t, source, "alpha", recipe.TaskInference, accuracy, 0, scorecard.Cell{Value: 1, Source: source}),
				}})
				return err
			}
			return nil
		})
	}
	phases := func(record plan.MeasurementRun) string {
		var names []string
		for _, phase := range record.Phases {
			verdict := "passed"
			if !phase.Passed {
				verdict = "failed"
			}
			names = append(names, phase.Name+" "+verdict)
		}
		return strings.Join(names, ", ")
	}
	if _, err := loop.Improve(improver); err == nil {
		t.Fatal("a cycle whose cells failed reported none")
	}
	first, kept := plan.ReadMeasurementRun(repository)
	if !kept || first.Commit != measured || first.Process == 0 || first.Ended.IsZero() || first.Filed != 0 {
		t.Fatalf("the first run's record = %+v (kept %t), want its commit, process and end, filing nothing", first, kept)
	}
	if got, want := phases(first), "preflight passed, lanes passed, cells failed, publication passed, fill passed"; got != want {
		t.Fatalf("the first run's phases = %q, want %q", got, want)
	}
	for _, phase := range first.Phases {
		if !phase.Expected.IsZero() {
			t.Fatalf("the first run expects %s to end with no earlier run to time it", phase.Name)
		}
	}
	if report := first.Report(); !strings.Contains(report, "cells did not pass") || !strings.Contains(report, "filing 0 rows") {
		t.Fatalf("the first run reads %q", report)
	}

	accuracy = 0.4
	if _, err := loop.Improve(improver); err == nil {
		t.Fatal("a cycle whose cells failed reported none")
	}
	second, _ := plan.ReadMeasurementRun(repository)
	if second.Filed != 1 || second.Ended.IsZero() || !second.Started.After(first.Started) {
		t.Fatalf("the second run's record = %+v, want the decline it filed and its end", second)
	}
	for _, phase := range second.Phases {
		if wall, timed := first.Wall(phase.Name); !timed || !phase.Expected.Equal(phase.Started.Add(wall)) {
			t.Fatalf("the second run expects %s to end %v, want its start and the first run's wall %v", phase.Name, phase.Expected, wall)
		}
	}
}
