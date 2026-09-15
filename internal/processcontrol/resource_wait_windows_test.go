//go:build windows

package processcontrol

import (
	"bufio"
	"context"
	"errors"
	"os"
	"os/exec"
	"strings"
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
		done <- AwaitResource(t.Context(), name, func() error {
			if !released.Load() {
				return ErrResourceBusy
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
	name := "overgo-test-await-exit-" + t.Name()
	owner := exec.Command(os.Args[0], "-test.run=^TestResourceContention$")
	owner.Env = append(os.Environ(), "OVERGO_TEST_RESOURCE_CLAIM="+name, "OVERGO_TEST_RESOURCE_OWNER=1")
	stdin, err := owner.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	stdout, err := owner.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := owner.Start(); err != nil {
		t.Fatal(err)
	}
	defer owner.Wait()
	reader := bufio.NewReader(stdout)
	if line, err := reader.ReadString('\n'); err != nil || !strings.HasPrefix(line, "reserved") {
		t.Fatalf("owner did not reserve: %q %v", line, err)
	}
	attempts := 0
	done := make(chan error, 1)
	go func() {
		done <- AwaitResource(t.Context(), name, func() error {
			attempts++
			return ClaimResource(name)
		})
	}()
	// The owner exits when its stdin closes; nothing else signals the waiter.
	if err := stdin.Close(); err != nil {
		t.Fatal(err)
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if attempts < 1 {
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
	err := AwaitResource(ctx, name, func() error {
		cancel(ended)
		return ErrResourceBusy
	})
	if !errors.Is(err, ended) || !errors.Is(err, ErrResourceBusy) {
		t.Fatalf("lost cause or contention: %v", err)
	}
	fatal := errors.New("device unavailable")
	calls := 0
	err = AwaitResource(t.Context(), name, func() error { calls++; return fatal })
	if !errors.Is(err, fatal) || calls != 1 {
		t.Fatalf("non-contention failure retried: calls=%d error=%v", calls, err)
	}
	canceled, stop := context.WithCancelCause(t.Context())
	stop(context.Canceled)
	err = AwaitResource(canceled, name, func() error { t.Error("claim after cancellation"); return nil })
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation lost: %v", err)
	}
}
