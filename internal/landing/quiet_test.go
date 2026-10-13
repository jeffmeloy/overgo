package landing

import (
	"errors"
	"io"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"overgo/internal/quietmachine"
)

// quietTestMachine is the machine these tests share among themselves, not the host's own,
// so a cycle measuring on this machine never waits for a test.
const quietTestMachine = "overgo-test-quiet-machine"

// noWait is the notice of a wait the tests have no use for.
func noWait() {}

// TestATimedMeasurementAndALandingNeverRunTogether holds the quiet machine's two sides: landings
// share it, a timed measurement takes it alone, and each side finds the other busy until it
// releases, which wakes the one waiting. A cycle's timed cell and a landing's builds and
// tests are then never beside one another, so the cell reads the model's time. A side that
// finds the machine busy says so when it starts waiting, not when it stops.
func TestATimedMeasurementAndALandingNeverRunTogether(t *testing.T) {
	ctx := t.Context()
	first, waited, err := quietmachine.Await(ctx, quietTestMachine, false, noWait)
	if err != nil || waited != 0 {
		t.Fatalf("a landing on a free machine = (%v, %v), want it admitted at once", waited, err)
	}
	second, waited, err := quietmachine.Await(ctx, quietTestMachine, false, noWait)
	if err != nil || waited != 0 {
		t.Fatalf("a second landing = (%v, %v), want it to share the machine with the first", waited, err)
	}

	// A measurement waits for both landings and starts only once both have ended.
	var landings atomic.Int32
	landings.Store(2)
	waiting := make(chan struct{})
	var group sync.WaitGroup
	group.Go(func() {
		release, _, err := quietmachine.Await(ctx, quietTestMachine, true, func() { close(waiting) })
		if err != nil {
			t.Error(err)
			return
		}
		defer release()
		if running := landings.Load(); running != 0 {
			t.Errorf("a timed measurement started with %d landings running", running)
		}
	})
	<-waiting
	for _, release := range []func() error{first, second} {
		landings.Add(-1)
		if err := release(); err != nil {
			t.Fatal(err)
		}
	}
	group.Wait()

	// And the other way: no landing starts while a measurement runs.
	measuring, _, err := quietmachine.Await(ctx, quietTestMachine, true, noWait)
	if err != nil {
		t.Fatal(err)
	}
	var measured atomic.Bool
	measured.Store(true)
	waiting = make(chan struct{})
	group.Go(func() {
		release, _, err := quietmachine.Await(ctx, quietTestMachine, false, func() { close(waiting) })
		if err != nil {
			t.Error(err)
			return
		}
		defer release()
		if measured.Load() {
			t.Error("a landing started while a timed measurement was running")
		}
	})
	<-waiting
	measured.Store(false)
	if err := measuring(); err != nil {
		t.Fatal(err)
	}
	group.Wait()
}

// TestAMeasurementUnderALandingDoesNotWaitForIt holds the commands a landing starts to run
// their measurements at once: the tests of cmd/validate run cells' fixtures inside the gate,
// which holds the machine for them, and one that waited would wait for its own parent.
func TestAMeasurementUnderALandingDoesNotWaitForIt(t *testing.T) {
	t.Setenv(quietmachine.HeldEnvironment, "1")
	want := errors.New("measured")
	if reading, err := Measuring(t.Context(), io.Discard, func() error { return want }); !errors.Is(err, want) || reading.Waited != 0 || len(reading.Competing) != 0 {
		t.Fatalf("a measurement under a landing = (%+v, %v), want its own result without waiting and with nothing read of the machine", reading, err)
	}
}

// TestTimedWorkExcludesCompetingWork holds what a timed measurement reads of the machine it runs
// on, beyond the landing it waits for. The process list is given, never the real one, since the
// test's own binary and its go command are in the real one.
//
// Read from a process list, the work that competes is a build tool, a test binary and a go
// command that has no child. A go command that waits on a child is not work, so a server started
// with go run is not, and neither is the measurement's own run, which is its ancestors and what it
// started. A list that cannot be read is reported as what was seen and never as a quiet machine.
//
// A measurement reads the machine as it starts and as it ends. Work seen at either end makes the
// reading contended and is named once, in order. No work seen leaves the reading clean. The wait
// for a landing is returned apart, and is zero on a free machine. The measurement's own failure is
// returned beside the reading.
func TestTimedWorkExcludesCompetingWork(t *testing.T) {
	t.Parallel()
	const machine, self = quietTestMachine + "-competing", 13
	running := []Process{
		{"System Idle Process", 0, 0}, {"explorer.exe", 1, 0},
		// The measurement's own run: a go command ran the loop, which ran the go command of this process.
		{"go.exe", 10, 1}, {"loop.exe", 11, 10}, {"go.exe", 12, 11}, {"validate.exe", self, 12},
		// What this process started: a go command that waits on a test binary.
		{"go.exe", 60, self}, {"recipe.test.exe", 61, 60},
		// A server started with go run, and an editor's language server.
		{"go.exe", 50, 1}, {"server.exe", 51, 50}, {"gopls.exe", 70, 1},
	}
	// Another session's work: a build's tools under a go command that waits on them, a test binary
	// under another, and a go command with no child yet.
	working := append(slices.Clone(running),
		Process{"go.exe", 20, 1}, Process{"compile.exe", 21, 20}, Process{"compile.exe", 23, 20}, Process{"link.exe", 22, 20},
		Process{"go.exe", 30, 1}, Process{"scorecard.test.exe", 31, 30}, Process{"GO.EXE", 40, 1})
	if got := competing(running, self); len(got) != 0 {
		t.Fatalf("with only its own run, a waiting go command and a server beside it the measurement saw %q competing, want none", got)
	}
	busy := []string{"GO.EXE", "compile.exe", "link.exe", "scorecard.test.exe"}
	if got := competing(working, self); !slices.Equal(got, busy) {
		t.Fatalf("beside another session's build and test the measurement saw %q competing, want %q", got, busy)
	}
	// Seen from the other session's test binary, this process's own test binary competes.
	if got := competing(working, 31); !slices.Contains(got, "recipe.test.exe") || slices.Contains(got, "scorecard.test.exe") {
		t.Fatalf("the other session's test binary saw %q competing, want this run's test binary and not itself", got)
	}
	read := func(listed []Process, err error) func() ([]Process, error) {
		return func() ([]Process, error) { return listed, err }
	}
	if got := seenBeside(read(working, nil), self); !slices.Equal(got, busy) {
		t.Fatalf("a reading of the list gave %q, want %q", got, busy)
	}
	if got := seenBeside(read(nil, errors.New("access denied")), self); len(got) != 1 || !strings.Contains(got[0], "access denied") {
		t.Fatalf("a list that cannot be read gave %q, want the failure named as what was seen", got)
	}

	ctx, measured := t.Context(), errors.New("the measurement's own failure")
	// seen answers each reading of the machine in turn with the next of the lists given.
	seen := func(readings ...[]string) func() []string {
		return func() []string {
			next := readings[0]
			readings = readings[1:]
			return next
		}
	}
	for _, reads := range []struct {
		name         string
		first, last  []string
		want         []string
		failure      error
		wantContends bool
	}{
		{"no work seen", nil, nil, nil, nil, false},
		{"no work seen, and the measurement fails", nil, nil, nil, measured, false},
		{"work seen as it starts", []string{"compile.exe"}, nil, []string{"compile.exe"}, nil, true},
		{"work seen as it ends", nil, []string{"go.exe"}, []string{"go.exe"}, measured, true},
		{"work seen at both ends", []string{"link.exe", "compile.exe"}, []string{"compile.exe", "a.test.exe"}, []string{"a.test.exe", "compile.exe", "link.exe"}, nil, true},
	} {
		var said strings.Builder
		reading, err := timed(ctx, machine, seen(reads.first, reads.last), &said, func() error { return reads.failure })
		if !errors.Is(err, reads.failure) || (err == nil) != (reads.failure == nil) {
			t.Errorf("%s: the measurement returned %v, want its own result %v", reads.name, err, reads.failure)
		}
		if !slices.Equal(reading.Competing, reads.want) || reading.Waited != 0 {
			t.Errorf("%s: the reading is %+v, want %q competing and no wait on a free machine", reads.name, reading, reads.want)
		}
		if strings.Contains(said.String(), "contended") != reads.wantContends {
			t.Errorf("%s: the output says %q, want it to say contended only when work was seen", reads.name, said.String())
		}
	}

	// A landing holds the machine: the measurement waits, runs only once the landing has ended, and
	// returns its wait apart from what it measured.
	held, _, err := quietmachine.Await(ctx, machine, false, noWait)
	if err != nil {
		t.Fatal(err)
	}
	var landed, ran atomic.Bool
	written := make(chan struct{})
	output := firstWrite{written: sync.OnceFunc(func() { close(written) })}
	var reading Reading
	var group sync.WaitGroup
	group.Go(func() {
		reading, err = timed(ctx, machine, seen(nil, nil), output, func() error {
			ran.Store(landed.Load())
			return nil
		})
	})
	<-written
	// The wait is read from the clock, which may not have moved since the measurement began to wait.
	for began := time.Now(); time.Since(began) <= 0; {
	}
	landed.Store(true)
	if releaseErr := held(); releaseErr != nil {
		t.Fatal(releaseErr)
	}
	group.Wait()
	if err != nil || !ran.Load() || reading.Waited <= 0 || len(reading.Competing) != 0 {
		t.Fatalf("behind a landing the measurement returned (%+v, %v) and ran after the landing ended: %v; want it to run after, with its wait returned and nothing competing", reading, err, ran.Load())
	}
}
