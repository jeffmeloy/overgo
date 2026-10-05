// Package modelswap serves many models from one endpoint by swapping
// which child server runs -- llama-swap is the reference for the
// lifecycle, overgo authorities are the configuration: the store's
// servable catalog says what can run, and one child runs at a time
// because the store's writer lock admits one serving process.
package modelswap

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

// Servable names one launchable model: the display name requests
// route by, the on-disk artifact, and its store identity.
type Servable struct {
	Name     string
	Location string
	Model    string
}

// Process is one running child server.
type Process interface {
	// URL is the child's base address.
	URL() string
	// Ready blocks until the child answers health or the context ends.
	Ready(ctx context.Context) error
	// Stop terminates the child.
	Stop() error
}

// Launcher starts a child server for one servable.
type Launcher interface {
	Launch(ctx context.Context, servable Servable) (Process, error)
}

type upstream struct {
	servable Servable
	process  Process
	refs     int
	// held counts the refs that are subscriptions a page holds open for its own life: they keep the child
	// resident, but a swap does not wait for them.
	held     int
	idleStop *time.Timer
}

// SwapTimes is what each part of a swap cost the request that caused it: the wait for the old child's
// requests to drain, stopping it, launching the new one and its becoming ready. All are zero when the
// child asked for was already running.
type SwapTimes struct {
	// Launched is true for the request that launched a child, whether or not it replaced one.
	Launched                   bool
	Drain, Stop, Launch, Ready time.Duration
}

// String is the swap as a header value: each part in milliseconds.
func (swap SwapTimes) String() string {
	return fmt.Sprintf("drain=%d;stop=%d;launch=%d;ready=%d", swap.Drain.Milliseconds(), swap.Stop.Milliseconds(), swap.Launch.Milliseconds(), swap.Ready.Milliseconds())
}

// Lease is one request's hold on the running child.
type Lease struct {
	// Target is the child's base address.
	Target string
	// Swap is what the swap this request caused cost.
	Swap SwapTimes
	// held and released are guarded by the supervisor's lock.
	held, released bool
	release, hold  func()
}

// Release ends the hold. It may be called more than once.
func (lease *Lease) Release() { lease.release() }

// HoldOpen marks the request as a subscription the page keeps open for as long as it listens: it still
// keeps the child resident, but a swap no longer waits for it to end.
func (lease *Lease) HoldOpen() { lease.hold() }

// drainGrace bounds how long a swap waits for the old child's
// in-flight requests: the workbench holds endless streams (the
// activity stream stays open for the life of the page), and an
// endless stream must not veto the model the user asked for. Past the
// grace the old child stops and lingering streams sever.
const drainGrace = 15 * time.Second

// Supervisor owns the exclusive child lifecycle: Acquire hands out the
// running child's address for the requested model, swapping when a
// different model is asked for -- in-flight requests on the old child
// drain (bounded by the grace) before it stops, callers for the new
// model queue on the swap, and an idle child past the timeout stops so
// the device frees.
type Supervisor struct {
	mu       sync.Mutex
	drained  *sync.Cond
	launcher Launcher
	idle     time.Duration
	grace    time.Duration
	current  *upstream
	closed   bool
}

// New binds a supervisor to its launcher. A zero idle timeout keeps an
// idle child resident until the next swap or Close.
func New(launcher Launcher, idle time.Duration) (*Supervisor, error) {
	if launcher == nil {
		return nil, errors.New("model swap: a launcher is required")
	}
	supervisor := &Supervisor{launcher: launcher, idle: idle, grace: drainGrace}
	supervisor.drained = sync.NewCond(&supervisor.mu)
	return supervisor, nil
}

// Acquire returns the lease on the running child for the servable,
// launching or swapping as needed, which the caller MUST release when its
// request completes: releases drive draining and the idle clock. Callers
// block while a different model drains and starts.
func (s *Supervisor) Acquire(ctx context.Context, servable Servable) (*Lease, error) {
	if ctx == nil {
		return nil, errors.New("model swap: nil acquire context")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	wake := context.AfterFunc(ctx, func() {
		s.mu.Lock()
		defer s.mu.Unlock()
		s.drained.Broadcast()
	})
	defer wake()
	var deadline *time.Timer
	graceOver := false
	defer func() {
		if deadline != nil {
			deadline.Stop()
		}
	}()
	var swap SwapTimes
	draining := time.Now()
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if s.closed {
			return nil, errors.New("model swap: supervisor is closed")
		}
		if s.current != nil && s.current.servable == servable {
			return s.leaseLocked(swap), nil
		}
		if s.current == nil {
			break
		}
		// A different model is running: wait for its in-flight requests
		// to drain, then stop it and fall through to launch. The drain is
		// bounded -- endless streams held open on the old child that do not
		// declare themselves held open must not veto the swap, so past the
		// grace the old child stops anyway. The streams that do declare it
		// (the workbench's activity stream) are not waited for at all.
		if s.current.refs-s.current.held > 0 && !graceOver {
			if deadline == nil {
				deadline = time.AfterFunc(s.grace, func() {
					s.mu.Lock()
					graceOver = true
					s.mu.Unlock()
					s.drained.Broadcast()
				})
			}
			s.drained.Wait()
			continue
		}
		swap.Drain = time.Since(draining)
		stopping := time.Now()
		s.stopCurrentLocked()
		swap.Stop = time.Since(stopping)
		break
	}
	swap.Launched = true
	launching := time.Now()
	process, err := s.launcher.Launch(ctx, servable)
	if err != nil {
		return nil, fmt.Errorf("model swap: launch %q: %w", servable.Name, err)
	}
	swap.Launch = time.Since(launching)
	readying := time.Now()
	if err := process.Ready(ctx); err != nil {
		_ = process.Stop()
		return nil, fmt.Errorf("model swap: %q did not become ready: %w", servable.Name, err)
	}
	swap.Ready = time.Since(readying)
	s.current = &upstream{servable: servable, process: process}
	return s.leaseLocked(swap), nil
}

// leaseLocked retains the running child for one request and hands back its lease.
func (s *Supervisor) leaseLocked(swap SwapTimes) *Lease {
	owner := s.current
	lease := &Lease{Target: s.retainLocked(), Swap: swap}
	lease.release = s.releaseFunc(owner, lease)
	lease.hold = s.holdFunc(owner, lease)
	return lease
}

// Status reports the running child, if any.
func (s *Supervisor) Status() (Servable, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.current == nil {
		return Servable{}, false
	}
	return s.current.servable, true
}

// Close stops the child and refuses further acquires.
func (s *Supervisor) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closed = true
	s.stopCurrentLocked()
	s.drained.Broadcast()
	return nil
}

func (s *Supervisor) retainLocked() string {
	s.current.refs++
	if s.current.idleStop != nil {
		s.current.idleStop.Stop()
		s.current.idleStop = nil
	}
	return s.current.process.URL()
}

// releaseFunc returns the one-shot release for a specific upstream:
// a release after that upstream was already swapped away is a no-op,
// so a slow caller cannot decrement the NEXT model's requests.
func (s *Supervisor) releaseFunc(owner *upstream, lease *Lease) func() {
	var once sync.Once
	return func() {
		once.Do(func() {
			s.mu.Lock()
			defer s.mu.Unlock()
			lease.released = true
			if s.current != owner {
				return
			}
			owner.refs--
			if lease.held {
				owner.held--
			}
			// A swap waits only for the requests that are not held open, so it wakes when the last of those ends.
			if owner.refs-owner.held == 0 {
				s.drained.Broadcast()
			}
			if owner.refs > 0 {
				return
			}
			if s.idle > 0 {
				owner.idleStop = time.AfterFunc(s.idle, func() { s.stopIfIdle(owner) })
			}
		})
	}
}

// holdFunc returns the hold-open for a lease on a specific upstream: it moves the
// lease's ref from the ones a swap waits for to the ones it does not, and wakes a
// drain that was waiting on it. A hold after the release, or after the upstream
// was swapped away, does nothing.
func (s *Supervisor) holdFunc(owner *upstream, lease *Lease) func() {
	return func() {
		s.mu.Lock()
		defer s.mu.Unlock()
		if s.current != owner || lease.released || lease.held {
			return
		}
		lease.held = true
		owner.held++
		s.drained.Broadcast()
	}
}

// stopIfIdle stops the upstream only when it is still current and
// still unreferenced: an acquire between the timer firing and the lock
// keeps the child alive.
func (s *Supervisor) stopIfIdle(owner *upstream) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.current != owner || owner.refs > 0 {
		return
	}
	s.stopCurrentLocked()
}

func (s *Supervisor) stopCurrentLocked() {
	if s.current == nil {
		return
	}
	if s.current.idleStop != nil {
		s.current.idleStop.Stop()
	}
	_ = s.current.process.Stop()
	s.current = nil
}
