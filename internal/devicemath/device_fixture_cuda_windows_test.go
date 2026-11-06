//go:build windows

package devicemath

import (
	"math"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/testcheck"
)

// residentAllocation allocates resident F32 buffers on worker, each kept in pointers for the test to free.
func residentAllocation(t *testing.T, worker *device.Worker, pointers *[]driver.DevicePtr) func(count int, initial []float32) driver.DevicePtr {
	return func(count int, initial []float32) driver.DevicePtr {
		t.Helper()
		pointer := testcheck.Must(AllocResidentF32(worker, count, initial))(t)
		*pointers = append(*pointers, pointer)
		return pointer
	}
}

// finiteDifference returns the most the central difference of loss, eps either side of each entry of param, differs
// from analytic, and logs it under name.
func finiteDifference(t *testing.T, name string, param, analytic []float32, eps float64, loss func() float64) float64 {
	var maxDiff float64
	for i := range param {
		orig := param[i]
		param[i] = orig + float32(eps)
		lp := loss()
		param[i] = orig - float32(eps)
		lm := loss()
		param[i] = orig
		if diff := math.Abs((lp-lm)/(2*eps) - float64(analytic[i])); diff > maxDiff {
			maxDiff = diff
		}
	}
	t.Logf("%s: max |grad-fd| %.3e", name, maxDiff)
	return maxDiff
}
