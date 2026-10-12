//go:build windows

package main

import (
	"os"
	"slices"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/runrecord"
	"overgo/internal/trainingworkflow"
)

// heldDeviceBytes is the device memory the test's own process holds while a
// child trains: far more than the trainer of the fixture's tiny model holds.
const heldDeviceBytes = 256 << 20

// TestWorkbenchRewardRunTrainsInItsOwnProcessOnTheDevice holds a reward run a
// workbench starts to the graphics card and to a process of its own. The
// child is this test binary run as the training command with the device
// trainer the command is built with, while this process holds device memory
// of its own, as a server holds the model it serves.
//
// The run returns its step, and the model it published has moved from the
// one it began at. The session the child recorded is a device session, and
// the peak it recorded is above zero and below what this process holds: it
// is the trainer's alone. A trainer inside this process would have read this
// process's ledger, and recorded at least what this process holds.
func TestWorkbenchRewardRunTrainsInItsOwnProcessOnTheDevice(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	fixture := newRewardCommandFixture(t)
	t.Setenv(commandChildEnvironment, childDevice)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	held := device.NewAllocationSet(worker)
	defer held.Close(ctx)
	if _, err := held.Allocate(ctx, heldDeviceBytes); err != nil {
		t.Fatal(err)
	}
	if resident := driver.ProcessCurrentBytes(); resident < heldDeviceBytes {
		t.Fatalf("this process holds %d device bytes after allocating %d; want its own ledger to count them", resident, uint64(heldDeviceBytes))
	}

	request := fixture.request("trained")
	result, err := trainingworkflow.ExecuteInCommand(ctx, os.Args[0], fixture.storePath, request)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.GRPO) != 1 || !result.GRPO[0].Valid() || result.Checkpoint.Optimizer.Step != 1 {
		t.Fatalf("the run returned the steps %+v and a checkpoint at step %d; want the one step the device took", result.GRPO, result.Checkpoint.Optimizer.Step)
	}
	moved := false
	trained := modelWeights(t, request.OutputDirectory)
	for name, values := range modelWeights(t, fixture.model) {
		moved = moved || !slices.Equal(values, trained[name])
	}
	if !moved {
		t.Fatal("the model the run published holds the weights it began at; want the device's step to have moved them")
	}
	sessions := fixture.sessions(t)
	if len(sessions) != 1 || sessions[0].Outcome != runrecord.OutcomeSucceeded {
		t.Fatalf("after the run the store holds the training sessions %+v; want the one succeeded session the child recorded", sessions)
	}
	environment, err := runrecord.RequireEnvironment(ctx, fixture.store, sessions[0].Environment)
	if err != nil || environment.Device != "cuda0" {
		t.Fatalf("the session was recorded on %q (%v); want the device", environment.Device, err)
	}
	if peak := sessions[0].Resources.PeakDeviceBytes; peak == 0 || peak >= heldDeviceBytes {
		t.Fatalf("the session recorded a device peak of %d bytes while this process held %d; want the trainer's own, above zero and below what this process holds",
			peak, uint64(heldDeviceBytes))
	}
}
