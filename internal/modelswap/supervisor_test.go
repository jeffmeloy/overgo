package modelswap

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type fakeProcess struct {
	url     string
	stopped atomic.Bool
	// done closes on the first Stop; a test waits on it for the idle stop.
	done     chan struct{}
	stopDone func()
}

func TestModelSwapCanceledDrainAndArtifactIdentity(t *testing.T) {
	launcher := &fakeLauncher{}
	supervisor, err := New(launcher, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer supervisor.Close()
	first := Servable{Name: "shared", Model: "first", Location: "first/weights"}
	firstLease, err := supervisor.Acquire(t.Context(), first)
	if err != nil {
		t.Fatal(err)
	}
	release := firstLease.Release
	defer release()
	original := launcher.processes[first.Name]
	ctx, cancel := context.WithCancelCause(t.Context())
	done := make(chan error, 1)
	next := Servable{Name: first.Name, Model: "second", Location: "second/weights"}
	go func() {
		acquired, err := supervisor.Acquire(ctx, next)
		if acquired != nil {
			acquired.Release()
		}
		done <- err
	}()
	// A replacement that did not wait for the active request would answer
	// with success, which the cancelled result below refuses.
	cancel(context.Canceled)
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled swap: %v", err)
	}
	if current, ok := supervisor.Status(); !ok || current != first || original.stopped.Load() {
		t.Fatal("canceled swap disturbed the active model")
	}
	release()
	leaseNext, err := supervisor.Acquire(t.Context(), next)
	if err != nil {
		t.Fatal(err)
	}
	leaseNext.Release()
	if len(launcher.launched) != 2 || !original.stopped.Load() {
		t.Fatal("same display name hid changed artifact identity")
	}
}

func (p *fakeProcess) URL() string                     { return p.url }
func (p *fakeProcess) Ready(ctx context.Context) error { return ctx.Err() }
func (p *fakeProcess) Stop() error {
	p.stopped.Store(true)
	p.stopDone()
	return nil
}

type fakeLauncher struct {
	mu        sync.Mutex
	launched  []string
	processes map[string]*fakeProcess
	fail      map[string]error
}

func (l *fakeLauncher) Launch(_ context.Context, servable Servable) (Process, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if err := l.fail[servable.Name]; err != nil {
		return nil, err
	}
	l.launched = append(l.launched, servable.Name)
	done := make(chan struct{})
	process := &fakeProcess{url: "http://upstream/" + servable.Name, done: done, stopDone: sync.OnceFunc(func() { close(done) })}
	if l.processes == nil {
		l.processes = map[string]*fakeProcess{}
	}
	l.processes[servable.Name] = process
	return process, nil
}

// TestModelSwapSupervisor pins the exclusive lifecycle: one child at a
// time, reuse without relaunch, swap stops the old child only after
// its in-flight requests drain (a queued caller for the new model
// waits), a stale release cannot touch the next model, and the idle
// timeout stops an unreferenced child.
func TestModelSwapSupervisor(t *testing.T) {
	ctx := t.Context()
	launcher := &fakeLauncher{}
	supervisor, err := New(launcher, 50*time.Millisecond)
	if err != nil {
		t.Fatal(err)
	}
	defer supervisor.Close()

	firstLease, err := supervisor.Acquire(ctx, Servable{Name: "alpha"})
	if err != nil || firstLease.Target != "http://upstream/alpha" {
		t.Fatalf("acquire alpha = (%+v, %v)", firstLease, err)
	}
	releaseFirst := firstLease.Release
	againLease, err := supervisor.Acquire(ctx, Servable{Name: "alpha"})
	if err != nil || againLease.Target != firstLease.Target || len(launcher.launched) != 1 {
		t.Fatalf("reacquire relaunched: lease=%+v launches=%v err=%v", againLease, launcher.launched, err)
	}
	againLease.Release()

	// A swap request queues until alpha's in-flight request releases,
	// and only then does alpha stop and beta start.
	swapDone := make(chan string, 1)
	go func() {
		betaLease, err := supervisor.Acquire(ctx, Servable{Name: "beta"})
		if err != nil {
			swapDone <- "error: " + err.Error()
			return
		}
		betaLease.Release()
		swapDone <- betaLease.Target
	}()
	if launcher.processes["alpha"].stopped.Load() {
		t.Fatal("alpha stopped while a request was in flight")
	}
	releaseFirst()
	if url := <-swapDone; url != "http://upstream/beta" {
		t.Fatalf("swap landed on %q", url)
	}
	if !launcher.processes["alpha"].stopped.Load() {
		t.Fatal("alpha survived the swap")
	}
	// The stale alpha release is a no-op against beta.
	releaseFirst()
	if _, running := supervisor.Status(); !running {
		t.Fatal("stale release stopped the swapped-in child")
	}

	// The idle timeout stops the unreferenced child; its stop is the signal.
	<-launcher.processes["beta"].done
	if _, running := supervisor.Status(); running {
		t.Fatal("status reports a stopped child")
	}

	// A launch failure propagates and leaves nothing running.
	launcher.mu.Lock()
	launcher.fail = map[string]error{"broken": errors.New("no such model")}
	launcher.mu.Unlock()
	if _, err := supervisor.Acquire(ctx, Servable{Name: "broken"}); err == nil {
		t.Fatal("failed launch acquired")
	}
	if _, running := supervisor.Status(); running {
		t.Fatal("failed launch left a child registered")
	}

	// A never-released acquire (an endless stream through the proxy)
	// cannot veto a swap: past the drain grace the old child stops
	// anyway and the requested model launches.
	supervisor.mu.Lock()
	supervisor.grace = 50 * time.Millisecond
	supervisor.mu.Unlock()
	if _, err := supervisor.Acquire(ctx, Servable{Name: "gamma"}); err != nil {
		t.Fatal(err)
	}
	// The gamma acquire above is deliberately never released.
	if deltaLease, err := supervisor.Acquire(ctx, Servable{Name: "delta"}); err != nil {
		t.Fatalf("bounded drain did not admit the swap: %v", err)
	} else {
		deltaLease.Release()
	}
	if !launcher.processes["gamma"].stopped.Load() {
		t.Fatal("grace expiry left the streamed-on child running")
	}
	if running, ok := supervisor.Status(); !ok || running.Name != "delta" {
		t.Fatalf("running after bounded drain = (%+v, %t)", running, ok)
	}
}

// TestSwapReportsItsPhasesAndDoesNotWaitOnAHeldOpenStream holds a swap to what a person
// choosing a model waits for: the old child's requests that are finishing, not the
// subscription the page keeps open for its own life, however long the drain grace is;
// the request that caused the swap is told what each part of it cost; and a request
// that did not swap is told nothing.
func TestSwapReportsItsPhasesAndDoesNotWaitOnAHeldOpenStream(t *testing.T) {
	ctx := t.Context()
	supervisor, err := New(&fakeLauncher{}, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer supervisor.Close()
	supervisor.mu.Lock()
	supervisor.grace = time.Hour
	supervisor.mu.Unlock()
	stream, err := supervisor.Acquire(ctx, Servable{Name: "alpha"})
	if err != nil {
		t.Fatal(err)
	}
	stream.HoldOpen()
	stream.HoldOpen()
	finishing, err := supervisor.Acquire(ctx, Servable{Name: "alpha"})
	if err != nil || finishing.Swap != (SwapTimes{}) {
		t.Fatalf("a request that did not swap = (%+v, %v), want no phases", finishing, err)
	}
	// A swap asked for while a request is finishing waits for it: cancelled, it ends cancelled, not swapped.
	waiting, cancel := context.WithCancelCause(ctx)
	cancelled := make(chan error, 1)
	go func() {
		lease, err := supervisor.Acquire(waiting, Servable{Name: "beta"})
		if lease != nil {
			lease.Release()
		}
		cancelled <- err
	}()
	cancel(context.Canceled)
	if err := <-cancelled; !errors.Is(err, context.Canceled) {
		t.Fatalf("a swap did not wait for the request that was finishing: %v", err)
	}
	// Once that request finishes the swap does not wait for the held-open stream, whatever the grace.
	swapped := make(chan *Lease, 1)
	go func() {
		lease, err := supervisor.Acquire(ctx, Servable{Name: "beta"})
		if err != nil {
			t.Error(err)
		}
		swapped <- lease
	}()
	finishing.Release()
	lease := <-swapped
	if lease == nil || lease.Target != "http://upstream/beta" || !lease.Swap.Launched {
		t.Fatalf("the swap landed on %+v", lease)
	}
	lease.Release()
	// A hold after the release does nothing: the next swap is not told a stale stream is held.
	stream.Release()
	stream.HoldOpen()
	if got := (SwapTimes{Drain: 1500 * time.Millisecond, Launch: 2 * time.Millisecond}).String(); got != "drain=1500;stop=0;launch=2;ready=0" {
		t.Fatalf("the swap's header value is %q", got)
	}
}
