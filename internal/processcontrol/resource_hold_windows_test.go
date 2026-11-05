//go:build windows

package processcontrol

import (
	"errors"
	"testing"

	"overgo/internal/testcheck"
)

// TestHoldResourceAdmitsOneHolderAlone holds the releasable exclusive side of a resource:
// while it is held, another hold and a shared admission see typed contention; once released
// (a second release is harmless) the resource admits shared holders, which keep another hold
// out until they release, and a waiter on the resource is woken by that release.
func TestHoldResourceAdmitsOneHolderAlone(t *testing.T) {
	name := sharedAdmissionName(t)
	release, err := HoldResource(name)
	testcheck.NoError(t, err)
	if _, err := HoldResource(name); !errors.Is(err, ErrResourceBusy) {
		t.Errorf("a second hold: %v, want typed contention", err)
	}
	if _, err := ShareResource(name); !errors.Is(err, ErrResourceBusy) {
		t.Errorf("a shared admission beside a hold: %v, want typed contention", err)
	}
	testcheck.NoError(t, errors.Join(release(), release()))

	shared, err := ShareResource(name)
	if err != nil {
		t.Fatalf("a shared admission after the release: %v", err)
	}
	if _, err := HoldResource(name); !errors.Is(err, ErrResourceBusy) {
		t.Errorf("a hold beside a shared admission: %v, want typed contention", err)
	}
	var held func() error
	done := make(chan error, 1)
	go func() {
		done <- AwaitResource(t.Context(), func() error {
			var err error
			held, err = HoldResource(name)
			return err
		})
	}()
	testcheck.NoError(t, shared())
	if err := <-done; err != nil || held == nil {
		t.Fatalf("a hold waiting for the shared admission = %v, want it granted once the admission releases", err)
	}
	testcheck.NoError(t, held())
}
