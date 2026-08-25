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
	idleStop *time.Timer
}

// Supervisor owns the exclusive child lifecycle: Acquire hands out the
// running child's address for the requested model, swapping when a
// different model is asked for -- in-flight requests on the old child
// drain before it stops, callers for the new model queue on the swap,
// and an idle child past the timeout stops so the device frees.
type Supervisor struct {
	mu       sync.Mutex
	drained  *sync.Cond
	launcher Launcher
	idle     time.Duration
	current  *upstream
	closed   bool
}

// New binds a supervisor to its launcher. A zero idle timeout keeps an
// idle child resident until the next swap or Close.
func New(launcher Launcher, idle time.Duration) (*Supervisor, error) {
	if launcher == nil {
		return nil, errors.New("model swap: a launcher is required")
	}
	supervisor := &Supervisor{launcher: launcher, idle: idle}
	supervisor.drained = sync.NewCond(&supervisor.mu)
	return supervisor, nil
}

// Acquire returns the running child's address for the servable,
// launching or swapping as needed, and a release the caller MUST call
// when its request completes: releases drive draining and the idle
// clock. Callers block while a different model drains and starts.
func (s *Supervisor) Acquire(ctx context.Context, servable Servable) (string, func(), error) {
	if ctx == nil {
		return "", nil, errors.New("model swap: nil acquire context")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for {
		if s.closed {
			return "", nil, errors.New("model swap: supervisor is closed")
		}
		if s.current != nil && s.current.servable.Name == servable.Name {
			return s.retainLocked(), s.releaseFunc(s.current), nil
		}
		if s.current == nil {
			break
		}
		// A different model is running: wait for its in-flight requests
		// to drain, then stop it and fall through to launch.
		if s.current.refs > 0 {
			s.drained.Wait()
			continue
		}
		s.stopCurrentLocked()
		break
	}
	process, err := s.launcher.Launch(ctx, servable)
	if err != nil {
		return "", nil, fmt.Errorf("model swap: launch %q: %w", servable.Name, err)
	}
	if err := process.Ready(ctx); err != nil {
		_ = process.Stop()
		return "", nil, fmt.Errorf("model swap: %q did not become ready: %w", servable.Name, err)
	}
	s.current = &upstream{servable: servable, process: process}
	return s.retainLocked(), s.releaseFunc(s.current), nil
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
func (s *Supervisor) releaseFunc(owner *upstream) func() {
	var once sync.Once
	return func() {
		once.Do(func() {
			s.mu.Lock()
			defer s.mu.Unlock()
			if s.current != owner {
				return
			}
			owner.refs--
			if owner.refs > 0 {
				return
			}
			s.drained.Broadcast()
			if s.idle > 0 {
				owner.idleStop = time.AfterFunc(s.idle, func() { s.stopIfIdle(owner) })
			}
		})
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
