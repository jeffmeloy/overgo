package landing

import (
	"errors"
	"io"
	"sync"
	"sync/atomic"
	"testing"
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
	first, waited, err := awaitQuiet(ctx, quietTestMachine, false, noWait)
	if err != nil || waited != 0 {
		t.Fatalf("a landing on a free machine = (%v, %v), want it admitted at once", waited, err)
	}
	second, waited, err := awaitQuiet(ctx, quietTestMachine, false, noWait)
	if err != nil || waited != 0 {
		t.Fatalf("a second landing = (%v, %v), want it to share the machine with the first", waited, err)
	}

	// A measurement waits for both landings and starts only once both have ended.
	var landings atomic.Int32
	landings.Store(2)
	waiting := make(chan struct{})
	var group sync.WaitGroup
	group.Go(func() {
		release, _, err := awaitQuiet(ctx, quietTestMachine, true, func() { close(waiting) })
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
	measuring, _, err := awaitQuiet(ctx, quietTestMachine, true, noWait)
	if err != nil {
		t.Fatal(err)
	}
	var measured atomic.Bool
	measured.Store(true)
	waiting = make(chan struct{})
	group.Go(func() {
		release, _, err := awaitQuiet(ctx, quietTestMachine, false, func() { close(waiting) })
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
	t.Setenv(underLandingEnvironment, "1")
	want := errors.New("measured")
	if err := Measuring(t.Context(), io.Discard, func() error { return want }); !errors.Is(err, want) {
		t.Fatalf("a measurement under a landing = %v, want its own result without waiting", err)
	}
}
