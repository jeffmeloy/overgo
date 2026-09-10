package processcontrol

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"
)

func TestResourceAdmissionWaitsForRelease(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		var released atomic.Bool
		done := make(chan error, 1)
		go func() {
			done <- AwaitResource(t.Context(), func() error {
				if !released.Load() {
					return ErrResourceBusy
				}
				return nil
			})
		}()
		synctest.Wait()
		select {
		case err := <-done:
			t.Fatalf("finished before owner release: %v", err)
		default:
		}
		released.Store(true)
		if err := <-done; err != nil {
			t.Fatal(err)
		}
	})
}

func TestResourceAdmissionCancellationAndFailure(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ctx, cancel := context.WithTimeoutCause(t.Context(), time.Second, context.DeadlineExceeded)
		defer cancel()
		err := AwaitResource(ctx, func() error { return ErrResourceBusy })
		if !errors.Is(err, context.DeadlineExceeded) || !errors.Is(err, ErrResourceBusy) {
			t.Fatalf("lost deadline or contention: %v", err)
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
	})
}
