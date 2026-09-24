//go:build windows

package devicemath

import (
	"context"
	"fmt"
	"overgo/internal/checked"
	"overgo/internal/cuda/driver"

	"math"
	"math/rand"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/hostmath"
	"overgo/internal/testutil"
)

func TestSoftmaxCrossEntropyBackwardResidentMatchesHost(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	const rows, classes = 11, 17
	logits := randSlice(rand.New(rand.NewSource(53)), rows*classes)
	targets, targetU32 := make([]int, rows), make([]uint32, rows)
	for row := range rows {
		targets[row] = (row*7 + 3) % classes
		targetU32[row] = uint32(targets[row])
	}
	want := make([]float32, len(logits))
	wantLoss := hostmath.SoftmaxCrossEntropy(want, logits, targets, rows, classes)
	logitsPtr, err := AllocResidentF32(worker, len(logits), logits)
	if err != nil {
		t.Fatal(err)
	}
	targetsPtr, err := AllocResidentU32(worker, targetU32)
	if err != nil {
		_ = FreeResident(worker, logitsPtr)
		t.Fatal(err)
	}
	lossesPtr, err := AllocResidentF32(worker, rows, nil)
	if err != nil {
		_ = FreeResident(worker, logitsPtr, targetsPtr)
		t.Fatal(err)
	}
	defer FreeResident(worker, logitsPtr, targetsPtr, lossesPtr)
	if err := SoftmaxCrossEntropyBackwardResident(worker, logitsPtr, targetsPtr, lossesPtr, rows, classes); err != nil {
		t.Fatal(err)
	}
	got, losses := make([]float32, len(logits)), make([]float32, rows)
	if err := ReadResident(worker, logitsPtr, ResidentSlice{Data: got}); err != nil {
		t.Fatal(err)
	}
	if err := ReadResident(worker, lossesPtr, ResidentSlice{Data: losses}); err != nil {
		t.Fatal(err)
	}
	var gotLoss float64
	for _, loss := range losses {
		gotLoss += float64(loss)
	}
	gradientDelta, lossDelta := testutil.MaxAbsDiff(got, want), math.Abs(gotLoss-wantLoss)
	t.Logf("resident/host CE gradient=%.3e loss=%.3e", gradientDelta, lossDelta)
	if gradientDelta > 2e-7 || lossDelta > 2e-6 {
		t.Fatalf("resident CE differs: gradient=%.3e loss=%.3e", gradientDelta, lossDelta)
	}
}

// AllocResidentU32 allocates and initializes persistent device indices.
func AllocResidentU32(worker *device.Worker, values []uint32) (driver.DevicePtr, error) {
	if worker == nil || len(values) == 0 {
		return 0, fmt.Errorf("AllocResidentU32: values absent")
	}
	bytes, ok := checked.Bytes(uint64(len(values)), 4)
	if !ok {
		return 0, fmt.Errorf("AllocResidentU32: size overflow")
	}
	var pointer driver.DevicePtr
	err := worker.Do(context.Background(), func(state *device.State) error {
		allocated, err := state.Driver.MemAlloc(bytes)
		if err != nil {
			return err
		}
		if err := state.Driver.MemcpyHtoD(allocated, driver.Bytes(values)); err != nil {
			_ = state.Driver.MemFree(allocated)
			return err
		}
		pointer = allocated
		return nil
	})
	return pointer, err
}

// SoftmaxCrossEntropyBackwardResident replaces logits with mean CE gradients.
func SoftmaxCrossEntropyBackwardResident(
	worker *device.Worker,
	logits, targets, losses driver.DevicePtr,
	rows, classes int,
) error {
	if worker == nil || logits == 0 || targets == 0 || losses == 0 || rows <= 0 || classes <= 0 || uint64(rows) > math.MaxUint32 || uint64(classes) > math.MaxUint32 {
		return fmt.Errorf("SoftmaxCrossEntropyBackwardResident: invalid buffer or geometry")
	}
	return WithResidentOps(worker, func(ops *ResidentOps) error {
		return ops.SoftmaxCrossEntropy(logits, targets, losses, rows, classes)
	})
}
