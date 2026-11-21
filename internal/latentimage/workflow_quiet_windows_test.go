//go:build windows

package latentimage

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"overgo/internal/processcontrol"
	"overgo/internal/quietmachine"
	"overgo/internal/testcheck"
)

// timedQuietly runs timing as a timing of the workflow runs (moss-first-run/quiet): on the quiet machine named alone,
// waiting for its shared holders (a commit's checks, overgo's workloads) to end and saying on output that it waits, and
// inside hold, the turns the timing takes after the machine (the card's, then the test turn every lane's step and a
// commit's tests take), so no wait of its own is timed and nothing of overgo's runs beside it on the host or the card.
// The machine comes first, as a landing and a workload take it before the card's turn, so no holder of a turn waits
// for the timing while the timing waits for that holder. A timing its holder started (quietmachine.HeldEnvironment)
// takes neither the machine nor the turns, which its holder holds for it: a process riding its holder's test turn that
// gave it up (landing.HoldTestTurnEach) would wait for the turn's waiters, a commit among them that waits for its
// holder. It returns how long it waited for the machine.
func timedQuietly(ctx context.Context, machine string, output io.Writer, hold func(work func() error) error, timing func() error) (waited time.Duration, err error) {
	if os.Getenv(quietmachine.HeldEnvironment) != "" {
		return 0, timing()
	}
	release, waited, err := quietmachine.Await(ctx, machine, true, func() {
		fmt.Fprintln(output, "the workflow's timing waits for the commit or the workload running on this machine to finish")
	})
	if err != nil {
		return 0, fmt.Errorf("the quiet machine: %w", err)
	}
	defer func() { err = errors.Join(err, release()) }()
	return waited, hold(timing)
}

// spoken is an output that says when it is first written to: first closes then.
type spoken struct {
	first chan struct{}
	said  func()
}

// newSpoken is an output not yet written to.
func newSpoken() *spoken {
	first := make(chan struct{})
	return &spoken{first: first, said: sync.OnceFunc(func() { close(first) })}
}

func (s *spoken) Write(line []byte) (int, error) {
	s.said()
	return len(line), nil
}

// TestAWorkflowTimingHoldsTheMachineQuietAlone holds a timing of the workflow to the quiet machine (moss-first-run/quiet):
// while a commit's checks hold the machine shared, the timing waits, says so, and runs nothing; once they let it go, it
// takes the machine alone, then the turns its hold takes, in their order, and times inside them, a shared holder
// refused while it times; it lets the machine go as it ends, and says how long it waited. A timing its holder started
// takes neither the machine nor the turns, running at once beside the shared holder. Two timings of the first run read the text to image section
// at 215.6 s (19c8144b0) and 775.7 s (f3117025d) against 69.4 s, and on 2026-11-21 one waited 4m46s for the card inside
// that section's wall, its turn taken as the session's first context was.
func TestAWorkflowTimingHoldsTheMachineQuietAlone(t *testing.T) {
	// Serial: it clears the mark of a held machine in the process environment, which a commit's checks set for every
	// command they run, this test's binary among them.
	t.Setenv(quietmachine.HeldEnvironment, "")
	ctx := t.Context()
	machine := fmt.Sprintf("overgo-test-workflow-quiet-%d", os.Getpid())
	checks, _, err := quietmachine.Await(ctx, machine, false, func() {})
	testcheck.NoError(t, err)
	var taken []string
	shared := make(chan error, 1)
	hold := func(name string, inner func(work func() error) error) func(work func() error) error {
		return func(work func() error) error {
			taken = append(taken, name)
			return inner(work)
		}
	}
	turns := hold("card", hold("tests", func(work func() error) error { return work() }))
	timing := func() error {
		taken = append(taken, "timing")
		release, err := processcontrol.ShareResource(machine)
		if err == nil {
			err = errors.Join(errors.New("a shared holder was admitted beside the timing"), release())
		} else if errors.Is(err, processcontrol.ErrResourceBusy) {
			err = nil
		}
		shared <- err
		return nil
	}
	output := newSpoken()
	done := make(chan error, 1)
	var waited time.Duration
	go func() {
		var err error
		waited, err = timedQuietly(ctx, machine, output, turns, timing)
		done <- err
	}()
	select {
	case <-output.first:
	case err := <-done:
		t.Fatalf("the timing ran beside a commit's checks holding the machine shared (%v); want it to wait for them", err)
	}
	if len(taken) != 0 {
		t.Fatalf("the timing took %v while it waited for the machine; want nothing taken before the machine", taken)
	}
	testcheck.NoError(t, checks())
	testcheck.NoError(t, <-done)
	if err := <-shared; err != nil {
		t.Errorf("while the timing held the machine, a shared claim: %v; want it refused", err)
	}
	if want := []string{"card", "tests", "timing"}; !slices.Equal(taken, want) {
		t.Errorf("the timing took %v; want the machine, then %v", taken, want)
	}
	// The wait said it began (output.first); its length may read 0 on a clock that steps by half a millisecond.
	t.Logf("the timing waited %v for the commit's checks", waited)
	release := testcheck.Must(processcontrol.ShareResource(machine))(t)
	// A timing its holder started rides its holder's side and turns, and waits for no one.
	t.Setenv(quietmachine.HeldEnvironment, "1")
	taken = nil
	ridden := &strings.Builder{}
	if _, err := timedQuietly(ctx, machine, ridden, turns, func() error {
		taken = append(taken, "timing")
		return nil
	}); err != nil || ridden.Len() != 0 || !slices.Equal(taken, []string{"timing"}) {
		t.Errorf("a timing its holder started, beside a shared holder, took %v, said %q, and answered %v; want it run at once in its holder's turns", taken, ridden, err)
	}
	testcheck.NoError(t, release())
}
