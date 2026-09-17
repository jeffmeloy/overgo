package hostmath

import (
	"errors"
	"runtime"
	"slices"
	"testing"
)

// Scheduling partitions outputs, never the accumulation for one output.
func TestDispatchSchedulingPreservesResults(t *testing.T) {
	previous, processors := dispatchOnce, runtime.GOMAXPROCS(2)
	t.Cleanup(func() { dispatchOnce = previous; runtime.GOMAXPROCS(processors) })
	for _, failed := range []bool{false, true} {
		dispatchOnce = func() (dispatchCalibration, error) {
			if failed {
				return dispatchCalibration{}, errors.New("fixture counter unavailable")
			}
			return dispatchCalibration{perTaskNs: 1, macNs: [macRateCount]float64{1, 1}}, nil
		}
		for _, width := range []int{1, 3, 17} {
			const rows, input, frames, kernel = 3, 5, 7, 3
			values := func(n int) []float32 {
				result := make([]float32, n)
				for index := range result {
					result[index] = float32(index%7-3) / 4
				}
				return result
			}
			x, weights := values(rows*input), values(width*input)
			want, got := make([]float32, rows*width), make([]float32, rows*width)
			linearCols(want, x, weights, rows, input, width, 0, width)
			Linear(got, x, weights, rows, input, width)
			if !slices.Equal(got, want) {
				t.Fatalf("linear changed: failed=%t width=%d", failed, width)
			}
			x, weights, bias := values(input*frames), values(width*input*kernel), values(width)
			want = make([]float32, width*frames)
			causalConv1dChannels(want, x, weights, bias, input, frames, frames, kernel, 1, kernel-1, 0, width)
			got = CausalConv1d(x, input, frames, weights, bias, width, kernel, 1)
			if !slices.Equal(got, want) {
				t.Fatalf("convolution changed: failed=%t width=%d", failed, width)
			}
			if workers := dispatchWorkers(width, rows*input, macF32); (workers == 1) != (failed || width == 1) {
				t.Fatalf("scheduling fixture missed its path: failed=%t width=%d workers=%d", failed, width, workers)
			}
		}
	}
}
