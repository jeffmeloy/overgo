package landing

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"time"

	"overgo/internal/processcontrol"
)

// quietMachine names the host-wide resource that keeps a timed measurement off a
// busy machine. A landing holds it shared for as long as it runs, builds and tests
// beside other landings; a timed measurement holds it alone while it measures, so
// what it reads is the model's time and not a compile's, and a landing waits for
// the measurement in progress and not for the cycle that runs it.
const quietMachine = "overgo-quiet-machine"

// underLandingEnvironment marks the commands a landing starts: their landing
// holds the quiet machine already, so a measurement one of them runs, as the
// tests of cmd/validate do, does not wait for the landing that is running it.
const underLandingEnvironment = "OVERGO_UNDER_LANDING"

// awaitQuiet takes one side of the quiet machine, waiting for the other side's
// holders to release it: alone for a timed measurement, shared for a landing. It
// calls waiting once, when the machine is found busy, so the wait is said when it
// starts and not only when it ends, and returns how long the wait took, zero when
// the machine was free.
func awaitQuiet(ctx context.Context, name string, alone bool, waiting func()) (release func() error, waited time.Duration, err error) {
	started, attempts := time.Now(), 0
	err = processcontrol.AwaitResource(ctx, func() error {
		attempts++
		var claimed error
		if alone {
			release, claimed = processcontrol.HoldResource(name)
		} else {
			release, claimed = processcontrol.ShareResource(name)
		}
		if attempts == 1 && errors.Is(claimed, processcontrol.ErrResourceBusy) {
			waiting()
		}
		return claimed
	})
	if err != nil {
		return nil, 0, err
	}
	if attempts > 1 {
		waited = time.Since(started).Round(time.Millisecond)
	}
	return release, waited, nil
}

// holdQuiet is a landing's side: it runs until the landing ends.
func (l *landing) holdQuiet(ctx context.Context) (release func() error, err error) {
	release, waited, err := awaitQuiet(ctx, quietMachine, false, func() {
		fmt.Fprintln(l.Log, "land: waiting for a timed measurement of the cycle to finish")
	})
	if err != nil {
		return nil, fmt.Errorf("the quiet machine: %w", err)
	}
	if waited > 0 {
		fmt.Fprintf(l.Log, "land: waited %s for a timed measurement to finish\n", waited)
		l.measured = append(l.measured, "timed measurement wait "+waited.String())
	}
	return release, nil
}

// Measuring runs one timed measurement with no landing running beside it, and
// no landing starts until it ends: the cycle's cells that read first output,
// throughput or load wait here for the landing in progress, and say so on
// output. A command a landing started runs it at once, the landing holding the
// machine for it.
func Measuring(ctx context.Context, output io.Writer, run func() error) (err error) {
	if os.Getenv(underLandingEnvironment) != "" {
		return run()
	}
	release, waited, err := awaitQuiet(ctx, quietMachine, true, func() {
		fmt.Fprintln(output, "measuring: waiting for the landing running on this machine to finish")
	})
	if err != nil {
		return fmt.Errorf("the quiet machine: %w", err)
	}
	defer func() { err = errors.Join(err, release()) }()
	if waited > 0 {
		fmt.Fprintf(output, "measuring: waited %s for the landing running on this machine\n", waited)
	}
	return run()
}
