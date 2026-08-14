//go:build windows

package devicemath

import (
	"math"
	"math/rand"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/hostmath"
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
	gradientDelta, lossDelta := maxAbsDiff(got, want), math.Abs(gotLoss-wantLoss)
	t.Logf("resident/host CE gradient=%.3e loss=%.3e", gradientDelta, lossDelta)
	if gradientDelta > 2e-7 || lossDelta > 2e-6 {
		t.Fatalf("resident CE differs: gradient=%.3e loss=%.3e", gradientDelta, lossDelta)
	}
}
