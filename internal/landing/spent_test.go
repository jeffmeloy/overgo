package landing

import (
	"errors"
	"io"
	"io/fs"
	"path/filepath"
	"runtime"
	"sync"
	"testing"
	"time"

	"overgo/internal/jsonfile"
	"overgo/internal/processlock"
	"overgo/internal/quietmachine"
)

// firstWrite is a log that says when it is first written to.
type firstWrite struct{ written func() }

func (w firstWrite) Write(data []byte) (int, error) {
	w.written()
	return len(data), nil
}

// TestLandingRecordsItsWaitsByCause holds each run of the gate to a record,
// in its checkout's process state, of where its time went. A preflight that
// passes, one a stage refuses, a landing whose tree moves under it and a
// landing that lands each leave one: how the run ended, how long the tree was
// blocked, and the time of each stage that ran. The waits are recorded by what
// was waited for: a landing that finds a timed measurement running records
// how long it waited for it.
func TestLandingRecordsItsWaitsByCause(t *testing.T) {
	ctx := t.Context()
	fixture := newRowFixture(t, map[string]string{"a.txt": "base\n", "internal/p/p.go": "// Package p is the fixture.\npackage p\n"})
	fixture.commitPlan(rowPlan(0, [2]string{"do", "true"}, [2]string{"refused", "false"}, [2]string{"moved", "echo moved > a.txt"}))
	message := []byte("Change the fixture\n\nCause: the fixture needs a change.\nPredicted effect: none.\n")
	recorded := func() []spent {
		t.Helper()
		var runs []spent
		if err := jsonfile.Decode(filepath.Join(fixture.repo, processlock.StateDirectory, spentRecord), &runs); err != nil && !errors.Is(err, fs.ErrNotExist) {
			t.Fatal(err)
		}
		return runs
	}
	for _, run := range []struct {
		reference, outcome string
		preflight          bool
		stages             []string
	}{
		{"row/do", outcomePreflight, true, []string{"format", "fix", "vet", "rules", "invariants", "verify"}},
		{"row/refused", "verify", true, []string{"format", "fix", "vet", "rules", "invariants", "verify"}},
		{"row/moved", outcomeConflict, false, []string{"format", "fix", "vet", "rules", "invariants", "test", "lanes", "verify", "scorecard"}},
		{"row/do", outcomeLanded, false, []string{"format", "fix", "vet", "rules", "invariants", "test", "lanes", "verify", "scorecard"}},
	} {
		fixture.write("a.txt", "changed for "+run.outcome+"\n")
		before := len(recorded())
		_, err := Run(ctx, Options{Repo: fixture.repo, Ref: run.reference, Message: message, Log: io.Discard, Preflight: run.preflight})
		if refused := run.outcome != outcomePreflight && run.outcome != outcomeLanded; (err != nil) != refused {
			t.Fatalf("%s as %s: %v", run.reference, run.outcome, err)
		}
		runs := recorded()
		if len(runs) != before+1 {
			t.Fatalf("%s as %s left %d records where there were %d, want one more", run.reference, run.outcome, len(runs), before)
		}
		last := runs[len(runs)-1]
		blocked := last.Ended.Sub(last.Started)
		if last.Reference != run.reference || last.Outcome != run.outcome || blocked <= 0 || len(last.Stages) != len(run.stages) {
			t.Fatalf("the run of %s is recorded as %+v, want it to have ended as %q after the stages %v", run.reference, last, run.outcome, run.stages)
		}
		var staged time.Duration
		for _, stage := range run.stages {
			wall, ran := last.Stages[stage]
			if !ran {
				t.Fatalf("the run of %s as %s records the stages %v, want %s among them", run.reference, run.outcome, last.Stages, stage)
			}
			staged += wall
		}
		if staged <= 0 || staged > blocked {
			t.Fatalf("the stages of %s as %s took %s of a run that blocked the tree for %s", run.reference, run.outcome, staged, blocked)
		}
	}

	// A landing that finds a timed measurement running waits for it, and the
	// wait is its record's.
	measuring, _, err := quietmachine.Await(ctx, quietTestMachine, true, noWait)
	if err != nil {
		t.Fatal(err)
	}
	waiting := make(chan struct{})
	l := &landing{Options: Options{Log: firstWrite{sync.OnceFunc(func() { close(waiting) })}}}
	var group sync.WaitGroup
	group.Go(func() {
		<-waiting
		// The measurement ends once the clock the landing reads has moved: this
		// host's clock ticks, and a wait within one tick reads as none.
		for began := time.Now(); !time.Now().After(began); {
			runtime.Gosched()
		}
		if err := measuring(); err != nil {
			t.Error(err)
		}
	})
	release, err := l.holdQuiet(ctx, quietTestMachine)
	if err != nil {
		t.Fatal(err)
	}
	group.Wait()
	if err := release(); err != nil {
		t.Fatal(err)
	}
	if l.spent.Measurement <= 0 {
		t.Fatalf("a landing that waited for a timed measurement recorded a wait of %s", l.spent.Measurement)
	}
}
