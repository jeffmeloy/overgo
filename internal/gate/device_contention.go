package gate

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"overgo/internal/cuda/driver"
	"overgo/internal/processcontrol"
	"overgo/internal/testevidence"
)

// testRunner runs packages as runGoTestsAdmitted does; the device batch
// takes it so a test can stand in for go test.
type testRunner func(ctx context.Context, packages []string, short bool, observe func(string, bool) error, leased bool) (testevidence.GoTestReport, error)

// runHostTests runs host packages as a testRunner; host packages hold no
// lease, so leased is ignored.
func (g *gateContext) runHostTests(ctx context.Context, packages []string, short bool, observe func(string, bool) error, _ bool) (testevidence.GoTestReport, error) {
	return g.runGoTests(ctx, packages, short, observe)
}

// testDevice names the device a refused package waits on: the one the
// batch lease admitted, or, for a host batch that held no lease, device
// zero read from the driver, the ordinal every trainer's optimizer stepper
// opens (device.New(0)).
func (g *gateContext) testDevice() (string, error) {
	if g.deviceResource != "" {
		return g.deviceResource, nil
	}
	library, err := driver.Open()
	if err != nil {
		return "", err
	}
	defer library.Close()
	if err := library.Init(); err != nil {
		return "", err
	}
	info, err := library.DeviceInfo(0)
	if err != nil {
		return "", err
	}
	g.deviceResource = info.UUID
	return info.UUID, nil
}

// runContendedBatch runs one batch (a device batch under the shared lease)
// and, when its only failures were refused exclusive claims, runs each
// refused package again alone and outside the lease: a package whose code
// claims the device exclusively is refused beneath any holder's lease, its
// siblings' and the gate's own, and admits once no other process holds the
// device. Host batches run through it too, because a host package can still
// open the device: on Windows every trainer's optimizer stepper does. The
// wait for a foreign holder ends when that holder releases the device or
// exits; the caller's context bounds the wait and each run alike.
func (g *gateContext) runContendedBatch(ctx context.Context, batch []string, short bool, observe func(string, bool) error, run testRunner) (testevidence.GoTestReport, error) {
	report, err := run(ctx, batch, short, observe, true)
	if err == nil || !report.ContentionOnly() {
		return report, err
	}
	contended := report.Contended
	g.note(fmt.Sprintf("device contention: %d package(s) refused an exclusive device claim [%s]; each runs again alone, outside any lease, once the device's holder releases",
		len(contended), strings.Join(contended, ",")))
	resource, err := g.testDevice()
	if err != nil {
		return report, fmt.Errorf("device contention: the refused device cannot be named to wait on: %w", err)
	}
	for _, pkg := range contended {
		attempts := 0
		err := processcontrol.AwaitResource(ctx, func() error {
			attempts++
			report, err = run(ctx, []string{pkg}, short, observe, false)
			if err != nil && report.ContentionOnly() {
				return &processcontrol.ResourceBusyError{Name: resource}
			}
			return err
		})
		if err != nil {
			if errors.Is(err, processcontrol.ErrResourceBusy) || ctx.Err() != nil {
				return report, fmt.Errorf("device contention: %s refused its exclusive claim %d time(s): %w", pkg, attempts, errors.Join(processcontrol.ErrResourceBusy, err, context.Cause(ctx)))
			}
			return report, err
		}
		g.note(fmt.Sprintf("device contention: %s admitted alone after %d attempt(s)", pkg, attempts))
	}
	return report, nil
}
