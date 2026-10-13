package landing

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"slices"
	"strings"
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
		waited = time.Since(started)
	}
	return release, waited, nil
}

// holdQuiet is a landing's side of the machine named: it runs until the
// landing ends, and the wait for it is the landing's to record.
func (l *landing) holdQuiet(ctx context.Context, machine string) (release func() error, err error) {
	release, waited, err := awaitQuiet(ctx, machine, false, func() {
		fmt.Fprintln(l.Log, "land: waiting for a timed measurement of the cycle to finish")
	})
	if err != nil {
		return nil, fmt.Errorf("the quiet machine: %w", err)
	}
	if l.spent.Measurement = waited; waited > 0 {
		fmt.Fprintf(l.Log, "land: waited %s for a timed measurement to finish\n", waited.Round(time.Millisecond))
		l.measured = append(l.measured, "timed measurement wait "+waited.Round(time.Millisecond).String())
	}
	return release, nil
}

// Reading is what a timed measurement found of the machine it ran on. Waited
// is how long it waited for a landing to end, which is no part of the time it
// measured. Competing names the work it saw beside it as it started or as it
// ended. A reading with competing work is contended, and its times are
// compared with nothing.
type Reading struct {
	Waited    time.Duration
	Competing []string
}

// Measuring runs one timed measurement with no landing running beside it, and
// no landing starts until it ends: the cycle's cells that read first output,
// throughput or load wait here for the landing in progress, and say so on
// output. A landing is not the only work a machine does, and a session's own
// build or test takes no side of the quiet machine, so the measurement also
// reads the machine's processes as it starts and as it ends and reports the
// build and test work it saw. A command a landing started runs it at once,
// the landing holding the machine for it, and reads no process.
func Measuring(ctx context.Context, output io.Writer, run func() error) (Reading, error) {
	if os.Getenv(underLandingEnvironment) != "" {
		return Reading{}, run()
	}
	return timed(ctx, quietMachine, func() []string { return seenBeside(Processes, os.Getpid()) }, output, run)
}

// seenBeside names the build and test work that one reading of the process
// list shows beside the process self and its own run. A list that cannot be
// read is named as what was seen, so an unread machine is never taken for a
// quiet one.
func seenBeside(read func() ([]Process, error), self int) []string {
	listed, err := read()
	if err != nil {
		return []string{"an unreadable process list (" + err.Error() + ")"}
	}
	return competing(listed, self)
}

// timed is Measuring on the machine named, with beside as its reading of the
// work that runs beside the measurement.
func timed(ctx context.Context, machine string, beside func() []string, output io.Writer, run func() error) (reading Reading, err error) {
	release, waited, err := awaitQuiet(ctx, machine, true, func() {
		fmt.Fprintln(output, "measuring: waiting for the landing running on this machine to finish")
	})
	if err != nil {
		return Reading{}, fmt.Errorf("the quiet machine: %w", err)
	}
	defer func() { err = errors.Join(err, release()) }()
	if reading.Waited = waited; waited > 0 {
		fmt.Fprintf(output, "measuring: waited %s for the landing running on this machine\n", waited.Round(time.Millisecond))
	}
	seen := beside()
	err = run()
	seen = append(seen, beside()...)
	slices.Sort(seen)
	if reading.Competing = slices.Compact(seen); len(reading.Competing) != 0 {
		fmt.Fprintf(output, "measuring: contended, %s ran beside this measurement\n", strings.Join(reading.Competing, ", "))
	}
	return reading, err
}
