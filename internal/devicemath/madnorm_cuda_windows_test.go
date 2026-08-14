//go:build windows

package devicemath

import (
	"math"
	"math/rand"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/hostmath"
)

func TestMADNormDeviceMatchesHost(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	const rows, width = 7, 8
	const epsilon = 1e-8
	rng := rand.New(rand.NewSource(7))
	input, incoming := randSlice(rng, rows*width), randSlice(rng, rows*width)
	wantForward, err := hostmath.MADNorm(input, rows, width, epsilon)
	if err != nil {
		t.Fatal(err)
	}
	wantBackward, err := hostmath.MADNormBackward(incoming, input, rows, width, epsilon)
	if err != nil {
		t.Fatal(err)
	}
	gotForward, err := MADNormForward(worker, input, rows, width, epsilon)
	if err != nil {
		t.Fatal(err)
	}
	gotBackward, err := MADNormBackward(worker, incoming, input, rows, width, epsilon)
	if err != nil {
		t.Fatal(err)
	}
	forwardDelta, backwardDelta := 0.0, 0.0
	for index := range input {
		forwardDelta = max(forwardDelta, math.Abs(float64(gotForward[index]-wantForward[index])))
		backwardDelta = max(backwardDelta, math.Abs(float64(gotBackward[index]-wantBackward[index])))
	}
	t.Logf("MADNorm CUDA/host forward=%.3e backward=%.3e", forwardDelta, backwardDelta)
	if forwardDelta > 1e-5 || backwardDelta > 1e-4 {
		t.Fatalf("MADNorm CUDA/host delta forward=%.3e backward=%.3e", forwardDelta, backwardDelta)
	}
}

func TestMADNormBackwardResidentMatchesShared(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	const rows, width = 9, 12
	const epsilon = 1e-8
	rng := rand.New(rand.NewSource(29))
	input, incoming := randSlice(rng, rows*width), randSlice(rng, rows*width)
	output, err := MADNormForward(worker, input, rows, width, epsilon)
	if err != nil {
		t.Fatal(err)
	}
	want, err := MADNormBackward(worker, incoming, input, rows, width, epsilon)
	if err != nil {
		t.Fatal(err)
	}
	var pointers []driver.DevicePtr
	allocate := func(initial []float32) driver.DevicePtr {
		t.Helper()
		pointer, err := AllocResidentF32(worker, len(input), initial)
		if err != nil {
			t.Fatal(err)
		}
		pointers = append(pointers, pointer)
		return pointer
	}
	defer func() {
		if err := FreeResident(worker, pointers...); err != nil {
			t.Error(err)
		}
	}()
	incomingPtr := allocate(incoming)
	inputPtr := allocate(input)
	outputPtr := allocate(output)
	gradientPtr := allocate(nil)
	if err := MADNormBackwardResident(worker, incomingPtr, inputPtr, outputPtr, gradientPtr, rows, width, epsilon); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, len(want))
	if err := ReadResident(worker, gradientPtr, ResidentSlice{Data: got}); err != nil {
		t.Fatal(err)
	}
	delta := maxAbsDiff(got, want)
	t.Logf("resident/shared MADNorm VJP=%.3e", delta)
	if delta != 0 {
		t.Fatalf("resident MADNorm VJP differs: %.3e", delta)
	}
}
