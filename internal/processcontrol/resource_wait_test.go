package processcontrol

import (
	"context"
	"errors"
	"testing"
)

// TestDeterministicCancellationContracts holds AwaitResource's cancellation and
// admission semantics with explicit claim state instead of holders, sleeps or a
// polling clock: a context cancelled before admission never claims, a claim that
// is granted or fails hard returns at once, a contention that names no resource
// is refused rather than waited on, a claim granted after one contention
// succeeds without a surviving wait, and a cancellation while waiting on a named
// holder ends the wait with the cause. Real holder-release and process-exit
// wakeups keep their platform integration tests.
func TestDeterministicCancellationContracts(t *testing.T) {
	t.Parallel()

	t.Run("cancelled before admission never claims", func(t *testing.T) {
		ctx, cancel := context.WithCancelCause(t.Context())
		cancel(nil)
		claims := 0
		err := AwaitResource(ctx, func() error { claims++; return nil })
		if !errors.Is(err, context.Canceled) || claims != 0 {
			t.Fatalf("cancelled admission: err=%v claims=%d", err, claims)
		}
	})

	t.Run("granted claim returns once", func(t *testing.T) {
		claims := 0
		if err := AwaitResource(t.Context(), func() error { claims++; return nil }); err != nil || claims != 1 {
			t.Fatalf("granted claim: err=%v claims=%d", err, claims)
		}
	})

	t.Run("hard failure returns without retry", func(t *testing.T) {
		boom := errors.New("hard failure")
		claims := 0
		err := AwaitResource(t.Context(), func() error { claims++; return boom })
		if !errors.Is(err, boom) || claims != 1 {
			t.Fatalf("hard failure: err=%v claims=%d", err, claims)
		}
	})

	t.Run("contention naming no resource is refused", func(t *testing.T) {
		err := AwaitResource(t.Context(), func() error { return ErrResourceBusy })
		if err == nil || !errors.Is(err, ErrResourceBusy) || errors.Is(err, context.Canceled) {
			t.Fatalf("unnamed contention: err=%v", err)
		}
	})

	t.Run("grant after one contention succeeds", func(t *testing.T) {
		claims := 0
		err := AwaitResource(t.Context(), func() error {
			claims++
			if claims == 1 {
				return &ResourceBusyError{Name: "overgo-test-det-grant"}
			}
			return nil
		})
		if err != nil || claims < 2 {
			t.Fatalf("grant after contention: err=%v claims=%d", err, claims)
		}
	})

	t.Run("cancellation while waiting ends with the cause", func(t *testing.T) {
		ctx, cancel := context.WithCancelCause(t.Context())
		claims := 0
		err := AwaitResource(ctx, func() error {
			claims++
			if claims >= 2 {
				cancel(nil)
			}
			return &ResourceBusyError{Name: "overgo-test-det-wait"}
		})
		if !errors.Is(err, context.Canceled) || !errors.Is(err, ErrResourceBusy) || claims < 2 {
			t.Fatalf("cancel while waiting: err=%v claims=%d", err, claims)
		}
	})
}
