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
}

func (p *fakeProcess) URL() string                     { return p.url }
func (p *fakeProcess) Ready(ctx context.Context) error { return ctx.Err() }
func (p *fakeProcess) Stop() error                     { p.stopped.Store(true); return nil }

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
	process := &fakeProcess{url: "http://upstream/" + servable.Name}
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

	first, releaseFirst, err := supervisor.Acquire(ctx, Servable{Name: "alpha"})
	if err != nil || first != "http://upstream/alpha" {
		t.Fatalf("acquire alpha = (%q, %v)", first, err)
	}
	again, releaseAgain, err := supervisor.Acquire(ctx, Servable{Name: "alpha"})
	if err != nil || again != first || len(launcher.launched) != 1 {
		t.Fatalf("reacquire relaunched: url=%q launches=%v err=%v", again, launcher.launched, err)
	}
	releaseAgain()

	// A swap request queues until alpha's in-flight request releases,
	// and only then does alpha stop and beta start.
	swapDone := make(chan string, 1)
	go func() {
		url, releaseBeta, err := supervisor.Acquire(ctx, Servable{Name: "beta"})
		if err != nil {
			swapDone <- "error: " + err.Error()
			return
		}
		releaseBeta()
		swapDone <- url
	}()
	select {
	case early := <-swapDone:
		t.Fatalf("swap completed before the in-flight request drained: %s", early)
	case <-time.After(30 * time.Millisecond):
	}
	if launcher.processes["alpha"].stopped.Load() {
		t.Fatal("alpha stopped while a request was in flight")
	}
	releaseFirst()
	select {
	case url := <-swapDone:
		if url != "http://upstream/beta" {
			t.Fatalf("swap landed on %q", url)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("swap never completed after the drain")
	}
	if !launcher.processes["alpha"].stopped.Load() {
		t.Fatal("alpha survived the swap")
	}
	// The stale alpha release is a no-op against beta.
	releaseFirst()
	if _, running := supervisor.Status(); !running {
		t.Fatal("stale release stopped the swapped-in child")
	}

	// The idle timeout stops the unreferenced child.
	deadline := time.Now().Add(2 * time.Second)
	for {
		if launcher.processes["beta"].stopped.Load() {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("idle beta never stopped")
		}
		time.Sleep(10 * time.Millisecond)
	}
	if _, running := supervisor.Status(); running {
		t.Fatal("status reports a stopped child")
	}

	// A launch failure propagates and leaves nothing running.
	launcher.mu.Lock()
	launcher.fail = map[string]error{"broken": errors.New("no such model")}
	launcher.mu.Unlock()
	if _, _, err := supervisor.Acquire(ctx, Servable{Name: "broken"}); err == nil {
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
	if _, _, err := supervisor.Acquire(ctx, Servable{Name: "gamma"}); err != nil {
		t.Fatal(err)
	}
	// The gamma acquire above is deliberately never released.
	if _, releaseDelta, err := supervisor.Acquire(ctx, Servable{Name: "delta"}); err != nil {
		t.Fatalf("bounded drain did not admit the swap: %v", err)
	} else {
		releaseDelta()
	}
	if !launcher.processes["gamma"].stopped.Load() {
		t.Fatal("grace expiry left the streamed-on child running")
	}
	if running, ok := supervisor.Status(); !ok || running.Name != "delta" {
		t.Fatalf("running after bounded drain = (%+v, %t)", running, ok)
	}
}
