//go:build windows

package processcontrol

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
)

// TestResourceAdmissionWaitsForRelease holds the admission wait to the
// resource's own signals: a claim refused for contention is made again when
// a holder releases, and the release itself wakes the waiter.
func TestResourceAdmissionWaitsForRelease(t *testing.T) {
	name := "overgo-test-await-" + t.Name()
	var released atomic.Bool
	done := make(chan error, 1)
	go func() {
		done <- AwaitResource(t.Context(), func() error {
			if !released.Load() {
				return &ResourceBusyError{Name: name}
			}
			return nil
		})
	}()
	hold, err := ShareResource(name)
	if err != nil {
		t.Fatal(err)
	}
	released.Store(true)
	if err := hold(); err != nil {
		t.Fatal(err)
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}

// TestResourceAdmissionWakesOnHolderExit holds the other signal: an
// exclusive holder that exits without releasing wakes the waiter through
// its process handle, and the claim then succeeds.
func TestResourceAdmissionWakesOnHolderExit(t *testing.T) {
	name := sharedAdmissionName(t)
	owner := holdSharedAdmission(t, sharedAdmissionProbe{Name: name, Exclusive: true})
	attempts := 0
	contended := make(chan struct{})
	done := make(chan error, 1)
	go func() {
		done <- AwaitResource(t.Context(), func() error {
			attempts++
			if attempts == 1 {
				defer close(contended)
			}
			return ClaimResource(name)
		})
	}()
	<-contended
	// Close stdin only after the first claim observes the live holder.
	if err := owner.input.Close(); err != nil {
		t.Fatal(err)
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if attempts < 2 {
		t.Fatalf("attempts = %d", attempts)
	}
}

// TestResourceAdmissionCancellationAndFailure holds the wait's ends: the
// caller's cancellation carries its cause beside the contention, a failure
// other than contention returns at once, and a cancelled caller claims nothing.
func TestResourceAdmissionCancellationAndFailure(t *testing.T) {
	name := "overgo-test-await-" + t.Name()
	ended := errors.New("the caller ended the wait")
	ctx, cancel := context.WithCancelCause(t.Context())
	err := AwaitResource(ctx, func() error {
		cancel(ended)
		return &ResourceBusyError{Name: name}
	})
	if !errors.Is(err, ended) || !errors.Is(err, ErrResourceBusy) {
		t.Fatalf("lost cause or contention: %v", err)
	}
	fatal := errors.New("device unavailable")
	calls := 0
	err = AwaitResource(t.Context(), func() error { calls++; return fatal })
	if !errors.Is(err, fatal) || calls != 1 {
		t.Fatalf("non-contention failure retried: calls=%d error=%v", calls, err)
	}
	canceled, stop := context.WithCancelCause(t.Context())
	stop(context.Canceled)
	err = AwaitResource(canceled, func() error { t.Error("claim after cancellation"); return nil })
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation lost: %v", err)
	}
}
