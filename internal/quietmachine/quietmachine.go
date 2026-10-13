// Package quietmachine keeps a timed measurement off a busy machine. It holds
// the one host-wide resource whose two sides do that, the wait for a side,
// and the entry overgo's own workloads take the shared side by.
//
// The package is its own, and small, so that the commands that train,
// evaluate and serve a recipe can take a side without importing the commit
// checks, and so that no package a capability is identified by imports it:
// a change here moves no capability's identity and expires no answer.
package quietmachine

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"runtime"
	"time"

	"overgo/internal/processcontrol"
)

// Name names the host-wide resource. A commit's checks hold it shared for as
// long as they run, and so does each of overgo's own workloads while it works
// (Workload). A timed measurement holds it alone while it measures, so what
// it reads is the model's time and not a compile's or a training run's. Each
// side waits for the other's holders, and for no longer than they hold it.
const Name = "overgo-quiet-machine"

// HeldEnvironment marks the commands a holder of the quiet machine starts.
// Their holder has the machine for them, so they take no side of it. A
// commit's checks set it for every command they run, and a timed measurement
// sets it for the command it measures, which would otherwise wait for its own
// parent to release the machine.
const HeldEnvironment = "OVERGO_QUIET_MACHINE_HELD"

// holdingPlatform is the one system the quiet machine's holds exist on.
const holdingPlatform = "windows"

// Await takes one side of the quiet machine named, waiting for the other
// side's holders to release it: alone for a timed measurement, shared for a
// commit's checks or a workload. It calls waiting once, when the machine is
// found busy, so the wait is said when it starts and not only when it ends,
// and returns how long the wait took, zero when the machine was free.
func Await(ctx context.Context, name string, alone bool, waiting func()) (release func() error, waited time.Duration, err error) {
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

// Workload returns work as one of overgo's own workloads runs it: on the
// shared side of the quiet machine, so a timed measurement and the work are
// never beside one another. The command named waits for a timed measurement
// in progress and says so on output, and a timed measurement waits for the
// work to end. A command whose holder has the machine for it takes no side
// (HeldEnvironment), and neither does one on a system that has no holds.
func Workload(name string, output io.Writer, work func() error) func() error {
	return func() error { return sharing(context.Background(), Name, name, output, work) }
}

// sharing is Workload's run on the machine named.
func sharing(ctx context.Context, machine, name string, output io.Writer, work func() error) (err error) {
	if os.Getenv(HeldEnvironment) != "" || runtime.GOOS != holdingPlatform {
		return work()
	}
	release, waited, err := Await(ctx, machine, false, func() {
		fmt.Fprintf(output, "%s: waiting for a timed measurement on this machine to finish\n", name)
	})
	if err != nil {
		return fmt.Errorf("the quiet machine: %w", err)
	}
	defer func() { err = errors.Join(err, release()) }()
	if waited > 0 {
		fmt.Fprintf(output, "%s: waited %s for a timed measurement on this machine\n", name, waited.Round(time.Millisecond))
	}
	return work()
}
