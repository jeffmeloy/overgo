//go:build windows

package processcontrol

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
)

// TestRecoveryDrills replays the device collisions landings met, on a
// resource named for the test, with no GPU: a served model's context, or a
// foreign gate's test stage, holds the shared device claim that every CUDA
// context takes (driver.ContextCreate calls ShareResource), and a guard run
// holds the device exclusively as ReserveDevice does. The recovery is a
// wait on the claim's holders rather than a run beside them or a refusal;
// its signals are the wait's admission once the holder leaves and, for a
// wait its caller ends, the holder it names.
func TestRecoveryDrills(t *testing.T) {
	for _, drill := range []struct {
		name string
		run  func(t *testing.T, name string)
	}{
		{
			// A lane's measurement, which takes the device exclusively
			// (cudatest.MeasurementProcess), waits on a context another
			// process holds instead of timing itself beside it, names that
			// process when its caller stops waiting, and is admitted once
			// the context's claim is released.
			name: "an exclusive reservation waits on a context's shared claim",
			run: func(t *testing.T, name string) {
				served := holdSharedAdmission(t, sharedAdmissionProbe{Name: name})
				stopped := errors.New("the lane stopped its device test")
				ctx, cancel := context.WithCancelCause(t.Context())
				attempts := 0
				err := AwaitResource(ctx, func() error {
					err := ClaimResource(name)
					// The first refusal opens the wait; the caller stops waiting
					// at the second, so the wait ends behind the live context.
					if attempts++; attempts > 1 {
						cancel(stopped)
					}
					return err
				})
				holder := fmt.Sprintf(".exe (pid %d)", served.command.Process.Pid)
				if !errors.Is(err, stopped) || !errors.Is(err, ErrResourceBusy) || !strings.Contains(err.Error(), holder) {
					t.Fatalf("the reservation ended without naming the context's process (want %q): %v", holder, err)
				}
				served.release(t)
				assertSharedAdmission(t, name, true, false)
			},
		},
		{
			// A context's claim made through the admission wait, as the swap
			// launcher makes a served model's, waits out a guard run's
			// exclusive hold instead of failing with the contention.
			name: "a context's claim waits out an exclusive holder",
			run: func(t *testing.T, name string) {
				guard := holdSharedAdmission(t, sharedAdmissionProbe{Name: name, Exclusive: true})
				contended := make(chan struct{})
				done := make(chan error, 1)
				attempts := 0
				var release func() error
				go func() {
					done <- AwaitResource(t.Context(), func() error {
						if attempts++; attempts == 1 {
							defer close(contended)
						}
						var err error
						release, err = ShareResource(name)
						return err
					})
				}()
				<-contended
				// The guard run ends only after the context met its hold.
				if err := guard.input.Close(); err != nil {
					t.Fatal(err)
				}
				if err := <-done; err != nil || attempts < 2 {
					t.Fatalf("the context's claim after %d attempts: %v", attempts, err)
				}
				if err := release(); err != nil {
					t.Fatal(err)
				}
			},
		},
	} {
		t.Run(drill.name, func(t *testing.T) { drill.run(t, sharedAdmissionName(t)) })
	}
}
