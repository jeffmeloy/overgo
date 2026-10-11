//go:build windows

package processcontrol

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
)

// TestResourceAdmissionWaitsForRelease holds the admission wait to its
// holders' releases. Two other processes hold the resource shared. The
// first releases after the waiter's armed claim was refused and before its
// wait begins, the gap the arming is there for; the second still holds, so
// that wait has a holder to wait on and ends only on the release's signal,
// on cancellation or on the second holder's exit. The waiter claims again
// and is refused, and the second holder then answers its own release: it
// had not exited, so the signal ended the wait. The hold is granted at the
// fourth claim exactly.
func TestResourceAdmissionWaitsForRelease(t *testing.T) {
	name := sharedAdmissionName(t)
	one := holdSharedAdmission(t, sharedAdmissionProbe{Name: name})
	two := holdSharedAdmission(t, sharedAdmissionProbe{Name: name})
	attempts := 0
	armed, released, woken := make(chan struct{}), make(chan struct{}), make(chan struct{})
	done := make(chan error, 1)
	var held func() error
	go func() {
		done <- AwaitResource(t.Context(), func() error {
			var err error
			held, err = HoldResource(name)
			switch attempts++; attempts {
			case 2:
				// Refused with the waiter armed: the wait does not begin
				// until the first holder has released.
				close(armed)
				select {
				case <-released:
				case <-t.Context().Done():
				}
			case 3:
				// Made only once that release has ended the wait.
				close(woken)
			}
			return err
		})
	}()
	reached := func(gate chan struct{}) {
		t.Helper()
		select {
		case <-gate:
		case err := <-done:
			t.Fatalf("the wait ended after %d claims: %v", attempts, err)
		}
	}
	reached(armed)
	one.release(t)
	close(released)
	reached(woken)
	// The second holder answers, so it was running when the wait ended: the
	// first release ended that wait, not a holder's exit.
	two.release(t)
	// Four on every schedule: two claims arm the wait and each release
	// prompts one. The count does not prove the waiter slept: one that
	// polled would exceed it only if a poll landed before the second release.
	if err := <-done; err != nil || attempts != 4 || held == nil {
		t.Fatalf("the hold after %d claims, want 4 (two to arm the wait, one for each release): %v", attempts, err)
	}
	if err := held(); err != nil {
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

// TestResourceAdmissionCancellationAndFailure holds one end of the wait: a
// caller that ends it with a cause of its own gets that cause beside the
// contention. A failure other than contention returning at once, and a
// cancelled caller claiming nothing, are held on every platform by
// TestDeterministicCancellationContracts.
func TestResourceAdmissionCancellationAndFailure(t *testing.T) {
	name := sharedAdmissionName(t)
	ended := errors.New("the caller ended the wait")
	ctx, cancel := context.WithCancelCause(t.Context())
	err := AwaitResource(ctx, func() error {
		cancel(ended)
		return &ResourceBusyError{Name: name}
	})
	if !errors.Is(err, ended) || !errors.Is(err, ErrResourceBusy) {
		t.Fatalf("lost cause or contention: %v", err)
	}
}

// TestResourceWaitNamesItsHolders holds a wait that ends on its caller's
// deadline to naming the processes that kept the resource: a device step
// timed out behind a foreign holder reports that holder's image and id,
// once, beside its cause and the contention.
func TestResourceWaitNamesItsHolders(t *testing.T) {
	name := sharedAdmissionName(t)
	owner := holdSharedAdmission(t, sharedAdmissionProbe{Name: name, Exclusive: true})
	ended := errors.New("the device step's deadline")
	ctx, cancel := context.WithCancelCause(t.Context())
	attempts := 0
	err := AwaitResource(ctx, func() error {
		err := ClaimResource(name)
		// The first refusal opens the waiter; the second ends the caller's
		// deadline, so the wait it enters ends behind the live holder.
		if attempts++; attempts > 1 {
			cancel(ended)
		}
		return err
	})
	holder := fmt.Sprintf(".exe (pid %d)", owner.command.Process.Pid)
	if !errors.Is(err, ended) || !errors.Is(err, ErrResourceBusy) || strings.Count(err.Error(), holder) != 1 {
		t.Fatalf("wait ended without naming its holder once (want %q): %v", holder, err)
	}
}
