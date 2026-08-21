package capabilityruntime

import (
	"context"
	"errors"
	"runtime"
	"testing"
	"time"
)

func TestBoundedSessionAdmission(t *testing.T) {
	t.Run("queue bound and cancellation", func(t *testing.T) {
		director, err := NewResidentModelSessionDirector("test", "host", 1, struct{}{})
		if err != nil {
			t.Fatal(err)
		}
		defer director.Close(context.Background())
		held, ok := director.TryLease(-1)
		if !ok {
			t.Fatal("initial lease was not admitted")
		}
		ctx, cancel := context.WithCancel(context.Background())
		parked := make(chan error, 1)
		go func() {
			_, err := director.Lease(ctx, -1)
			parked <- err
		}()
		waitForParkedSession(t, director)
		if _, err := director.Lease(context.Background(), -1); !errors.Is(err, ErrAdmissionQueueFull) {
			t.Fatalf("overflow error = %v", err)
		}
		cancel()
		if err := <-parked; !errors.Is(err, context.Canceled) {
			t.Fatalf("cancelled waiter error = %v", err)
		}
		if err := held.Release(); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("requested slot and shutdown", func(t *testing.T) {
		director, err := NewResidentModelSessionDirector("test", "host", 2, struct{}{})
		if err != nil {
			t.Fatal(err)
		}
		held, ok := director.TryLease(1)
		if !ok {
			t.Fatal("requested slot lease was not admitted")
		}
		parked := make(chan *SessionLease[struct{}], 1)
		parkedErr := make(chan error, 1)
		go func() {
			lease, err := director.Lease(context.Background(), 1)
			parked <- lease
			parkedErr <- err
		}()
		waitForParkedSession(t, director)
		other, ok := director.TryLease(0)
		if !ok {
			t.Fatal("unrequested slot was not independently available")
		}
		if err := held.Release(); err != nil {
			t.Fatal(err)
		}
		resumed := <-parked
		if err := <-parkedErr; err != nil || resumed == nil || resumed.ID != 1 {
			t.Fatalf("resumed requested lease = (%v, %v)", resumed, err)
		}
		shutdownCtx, cancel := context.WithCancel(context.Background())
		cancel()
		if _, err := director.Lease(shutdownCtx, -1); !errors.Is(err, context.Canceled) {
			t.Fatalf("pre-cancelled lease error = %v", err)
		}
		if err := other.Release(); err != nil {
			t.Fatal(err)
		}
		if err := resumed.Release(); err != nil {
			t.Fatal(err)
		}
		if err := director.Close(context.Background()); err != nil {
			t.Fatal(err)
		}
		if _, err := director.Lease(context.Background(), -1); !errors.Is(err, ErrSessionUnavailable) {
			t.Fatalf("closed director error = %v", err)
		}
	})

	t.Run("shutdown wakes parked request", func(t *testing.T) {
		director, err := NewResidentModelSessionDirector("test", "host", 1, struct{}{})
		if err != nil {
			t.Fatal(err)
		}
		held, ok := director.TryLease(-1)
		if !ok {
			t.Fatal("initial lease was not admitted")
		}
		parked := make(chan error, 1)
		go func() {
			_, err := director.Lease(context.Background(), -1)
			parked <- err
		}()
		waitForParkedSession(t, director)
		closed := make(chan error, 1)
		go func() { closed <- director.Close(context.Background()) }()
		if err := <-parked; !errors.Is(err, ErrSessionUnavailable) {
			t.Fatalf("shutdown waiter error = %v", err)
		}
		if err := held.Release(); err != nil {
			t.Fatal(err)
		}
		if err := <-closed; err != nil {
			t.Fatal(err)
		}
	})
}

func waitForParkedSession[Input, Model, Output any](t *testing.T, director *ModelSessionDirector[Input, Model, Output]) {
	t.Helper()
	timer := time.NewTimer(time.Second)
	defer timer.Stop()
	for director.Snapshot().Waiting == 0 {
		select {
		case <-timer.C:
			t.Fatal("session did not enter admission parking")
		default:
			runtime.Gosched()
		}
	}
}
