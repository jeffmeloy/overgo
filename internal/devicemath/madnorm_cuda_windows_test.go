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
