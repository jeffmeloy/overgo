//go:build windows

package optimizer

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/hostoptimizer"
)

// TestDeviceNewtonSchulzMatchesHost gates the fp32 device Newton-Schulz against
// the fp64 host oracle across tall/wide/square shapes within tolerance.
func TestDeviceNewtonSchulzMatchesHost(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	var ops *deviceOps
	if err := worker.Do(context.Background(), func(state *device.State) error { var err error; ops, err = newDeviceOps(state); return err }); err != nil {
		t.Fatal(err)
	}
	defer worker.Do(context.Background(), func(_ *device.State) error { ops.close(); return nil })

	cases := []struct {
		name       string
		rows, cols int
	}{
		{"square", 32, 32},
		{"tall", 64, 16},
		{"wide", 16, 64},
		{"small", 8, 8},
	}

	rng := rand.New(rand.NewSource(1))
	for _, tc := range cases {
		var buffers nsBuffers
		if err := worker.Do(context.Background(), func(_ *device.State) error {
			var err error
			buffers, err = ops.allocBuffers(tc.rows*tc.cols, gramDim(tc.rows, tc.cols)*gramDim(tc.rows, tc.cols))
			return err
		}); err != nil {
			t.Fatal(err)
		}
		defer worker.Do(context.Background(), func(_ *device.State) error { buffers.free(ops.lib); return nil })
		for _, distribution := range []string{"ordinary", "rank-one", "zero", "below-guard", "threshold-below", "threshold-above", "underflow"} {
			input := make([]float32, tc.rows*tc.cols)
			switch distribution {
			case "ordinary":
				for i := range input {
					input[i] = float32(rng.NormFloat64())
				}
			case "rank-one":
				for i := range input {
					input[i] = float32(i%tc.cols + 1)
				}
			case "below-guard":
				input[0] = float32(hostoptimizer.NewtonSchulzFrobeniusGuard / 2)
			case "threshold-below", "threshold-above":
				guard := hostoptimizer.NewtonSchulzFrobeniusGuard
				value := float32(guard)
				below := distribution == "threshold-below"
				// The old branch consumes the rounded FP32 reduction, not a FP64 sum.
				for (math.Sqrt(float64(value*value)) < guard) != below {
					toward := float32(math.Inf(1))
					if below {
						toward = 0
					}
					value = math.Nextafter32(value, toward)
				}
				input[0] = value
			case "underflow":
				input[0] = math.SmallestNonzeroFloat32
			}
			for _, tf32 := range []bool{false, true} {
				schedule := "8+2-FP32"
				if tf32 {
					schedule = "4+2-TF32"
				}
				t.Run(tc.name+"/"+distribution+"/"+schedule, func(t *testing.T) {
					got, squared, err := deviceNewtonSchulz(worker, ops, buffers, input, tc.rows, tc.cols, tf32)
					if err != nil {
						t.Fatal(err)
					}
					guarded := math.Sqrt(float64(squared)) < hostoptimizer.NewtonSchulzFrobeniusGuard
					for i, v := range got {
						if guarded && math.Float32bits(v) != math.Float32bits(input[i]) {
							t.Fatalf("guarded direction[%d]=%g, input %g", i, v, input[i])
						}
						if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
							t.Fatalf("nonfinite direction[%d]=%g", i, v)
						}
					}
					if distribution == "threshold-above" && slices.Equal(got, input) {
						t.Fatal("nonguarded direction was not normalized")
					}
					if !tf32 && distribution == "ordinary" {
						host := make([]float64, len(input))
						for i, v := range input {
							host[i] = float64(v)
						}
						hostoptimizer.NewtonSchulz(host, tc.rows, tc.cols)
						// Retain the established FP32 versus FP64 host-oracle bound.
						const tolerance = 1e-5
						for i, v := range got {
							if math.IsNaN(float64(v)) || math.Abs(float64(v)-host[i]) > tolerance {
								t.Fatalf("host-oracle direction[%d]=%g, GPU %g", i, host[i], v)
							}
						}
					}
					if guarded {
						t.Logf("FP32 squared norm=%08x guard=true; multiplier undefined", math.Float32bits(squared))
					} else {
						multiplier := float32(1 / math.Sqrt(float64(squared)))
						t.Logf("FP32 squared norm=%08x guard=false; reference multiplier=%08x", math.Float32bits(squared), math.Float32bits(multiplier))
					}
				})
			}
		}
	}
}

// deviceNewtonSchulz compares schedules with retained handles and shape-sized scratch.
func deviceNewtonSchulz(worker *device.Worker, ops *deviceOps, buffers nsBuffers, input []float32, rows, cols int, tf32 bool) ([]float32, float32, error) {
	if rows <= 0 || cols <= 0 || len(input) != rows*cols {
		return nil, 0, fmt.Errorf("deviceNewtonSchulz: len %d != %d*%d", len(input), rows, cols)
	}
	if uint64(len(input)) > math.MaxUint32 {
		return nil, 0, fmt.Errorf("deviceNewtonSchulz: matrix too large for a 32-bit element count")
	}
	normSquared := make([]float32, 1)
	result := make([]float32, len(input))
	copy(result, input)
	err := worker.Do(context.Background(), func(_ *device.State) error {
		if err := ops.lib.MemcpyHtoD(buffers.dX, driver.Bytes(result)); err != nil {
			return err
		}
		handle, iterations := ops.handle, hostoptimizer.NewtonSchulzStage1Iterations
		if tf32 {
			handle, iterations = ops.tf32, residentNewtonSchulzStage1Iterations
		}
		final, err := ops.runNewtonSchulz(buffers, rows, cols, handle, iterations)
		if err != nil {
			return err
		}
		if err := ops.lib.StreamSynchronize(ops.stream); err != nil {
			return err
		}
		if err := ops.lib.MemcpyDtoH(driver.Bytes(normSquared), buffers.dNorm); err != nil {
			return err
		}
		return ops.lib.MemcpyDtoH(driver.Bytes(result), final)
	})
	if err != nil {
		return nil, 0, err
	}
	return result, normSquared[0], nil
}
