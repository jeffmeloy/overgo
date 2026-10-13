package quietmachine

import (
	"context"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

// workloadTestMachine is the machine this test holds, not the host's own, so
// a timed measurement on this machine never waits for a test.
const workloadTestMachine = "overgo-test-workload-machine"

// said is an output a test reads while another goroutine writes it, which
// says when it is first written to.
type said struct {
	lock    sync.Mutex
	text    strings.Builder
	written func()
}

func (s *said) Write(line []byte) (int, error) {
	s.lock.Lock()
	defer s.lock.Unlock()
	s.written()
	return s.text.Write(line)
}

func (s *said) String() string {
	s.lock.Lock()
	defer s.lock.Unlock()
	return s.text.String()
}

// TestOvergoWorkloadsShareTheQuietMachine holds the side of the quiet machine
// that overgo's own workloads take while they work.
//
// Two workloads run beside one another, and neither waits. While a timed
// measurement holds the machine alone, a workload waits, says so as its wait
// starts, runs only once the measurement has released the machine, and says
// how long it waited. A timed measurement in turn waits for a workload to
// end. The work's own failure is returned.
//
// A command whose holder has the machine for it takes no side: it runs at
// once while the machine is held alone, which is how the command a timed
// measurement measures runs under that measurement. A workload whose wait is
// cancelled does not run its work.
func TestOvergoWorkloadsShareTheQuietMachine(t *testing.T) {
	// Serial: it sets and clears the mark of a held machine in the process environment. A commit's
	// checks set that mark for every command they run, this test's binary among them.
	t.Setenv(HeldEnvironment, "")
	ctx := t.Context()
	quiet := func() *said { return &said{written: func() {}} }

	// Two workloads beside one another: the second starts while the first is inside its work.
	inside, leave := make(chan struct{}), make(chan struct{})
	var group sync.WaitGroup
	group.Go(func() {
		if err := sharing(ctx, workloadTestMachine, "first", quiet(), func() error {
			close(inside)
			<-leave
			return nil
		}); err != nil {
			t.Error(err)
		}
	})
	<-inside
	failure := errors.New("the work's own failure")
	beside := quiet()
	if err := sharing(ctx, workloadTestMachine, "second", beside, func() error { return failure }); !errors.Is(err, failure) || beside.String() != "" {
		t.Fatalf("a workload beside another returned %v and said %q, want its own failure and no wait", err, beside.String())
	}
	close(leave)
	group.Wait()

	// A timed measurement holds the machine alone: a workload waits for it.
	measuring, waited, err := Await(ctx, workloadTestMachine, true, func() {})
	if err != nil || waited != 0 {
		t.Fatalf("a measurement on a free machine = (%v, %v), want it admitted at once", waited, err)
	}
	var measured, ran atomic.Bool
	measured.Store(true)
	waiting := make(chan struct{})
	output := &said{written: sync.OnceFunc(func() { close(waiting) })}
	group.Go(func() {
		if err := sharing(ctx, workloadTestMachine, "train", output, func() error {
			ran.Store(!measured.Load())
			return nil
		}); err != nil {
			t.Error(err)
		}
	})
	<-waiting
	if got := output.String(); !strings.Contains(got, "train: waiting for a timed measurement") {
		t.Fatalf("a workload behind a measurement said %q as its wait started, want it to name itself and what it waits for", got)
	}

	// The command the measurement measures takes no side, and runs while the machine is held alone.
	t.Setenv(HeldEnvironment, "1")
	held := quiet()
	if err := sharing(ctx, workloadTestMachine, "recipe", held, func() error { return failure }); !errors.Is(err, failure) || held.String() != "" {
		t.Fatalf("a command whose holder has the machine returned %v and said %q, want its own result at once", err, held.String())
	}
	t.Setenv(HeldEnvironment, "")

	// A workload whose wait is cancelled does not run.
	cancelled, cancel := context.WithCancelCause(ctx)
	stopped := errors.New("the operator stopped it")
	cancel(stopped)
	if err := sharing(cancelled, workloadTestMachine, "evaluate", quiet(), func() error {
		t.Error("a workload whose wait was cancelled ran its work")
		return nil
	}); !errors.Is(err, stopped) {
		t.Fatalf("a workload whose wait was cancelled returned %v, want the cancellation", err)
	}

	measured.Store(false)
	if err := measuring(); err != nil {
		t.Fatal(err)
	}
	group.Wait()
	if !ran.Load() || !strings.Contains(output.String(), "train: waited ") {
		t.Fatalf("after the measurement the workload ran: %t, and said %q; want it to run once the machine was released and to say how long it waited", ran.Load(), output.String())
	}

	// And the other way: a measurement waits for a workload to end.
	inside, leave = make(chan struct{}), make(chan struct{})
	var working atomic.Bool
	group.Go(func() {
		if err := sharing(ctx, workloadTestMachine, "evaluate", quiet(), func() error {
			working.Store(true)
			close(inside)
			<-leave
			working.Store(false)
			return nil
		}); err != nil {
			t.Error(err)
		}
	})
	<-inside
	waiting = make(chan struct{})
	group.Go(func() {
		release, _, err := Await(ctx, workloadTestMachine, true, func() { close(waiting) })
		if err != nil {
			t.Error(err)
			return
		}
		if working.Load() {
			t.Error("a timed measurement started while a workload was working")
		}
		if err := release(); err != nil {
			t.Error(err)
		}
	})
	<-waiting
	close(leave)
	group.Wait()
}
