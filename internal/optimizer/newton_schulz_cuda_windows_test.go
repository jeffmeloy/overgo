//go:build windows

package optimizer

import (
	"context"
	"fmt"
	"math"
	"math/rand"
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
		t.Run(tc.name, func(t *testing.T) {
			n := tc.rows * tc.cols
			host := make([]float64, n)
			dev := make([]float32, n)
			for i := range host {
				v := rng.NormFloat64()
				host[i] = v
				dev[i] = float32(v)
			}
			hostoptimizer.NewtonSchulz(host, tc.rows, tc.cols)

			got, err := deviceNewtonSchulz(worker, dev, tc.rows, tc.cols)
			if err != nil {
				t.Fatal(err)
			}
			var maxAbs float64
			for i := range host {
				if d := math.Abs(float64(got[i]) - host[i]); d > maxAbs {
					maxAbs = d
				}
			}
			t.Logf("%s (%dx%d): max abs device(fp32)-vs-host(fp64) = %.3e", tc.name, tc.rows, tc.cols, maxAbs)
			// Observed ~1e-7..7e-7 (fp32 precision; Newton-Schulz is contractive
			// so error does not accumulate). 1e-5 keeps a wide margin while
			// still catching a real regression.
			const tolerance = 1e-5
			if maxAbs > tolerance {
				t.Fatalf("%s: max abs error %.3e > %.1e", tc.name, maxAbs, tolerance)
			}
		})
	}
}

// deviceNewtonSchulz runs the Muon Newton-Schulz iteration on the GPU in fp32,
// reproducing host newtonSchulz (fp64 oracle) within tolerance. Every matmul is
// a cuBLAS GEMM; the c0*I term folds into the final matmul
// (X*(c0*I + R) = c0*X + X*R), so no diagonal kernel is needed -- only cuBLAS
// plus the existing ops_f32 scale/add. The Frobenius norm reuses a GEMM
// dot-product. Buffers/handles are per-call (correctness-first; the device Muon
// step reuses the same primitives on resident state).
func deviceNewtonSchulz(worker *device.Worker, input []float32, rows, cols int) ([]float32, error) {
	if rows <= 0 || cols <= 0 || len(input) != rows*cols {
		return nil, fmt.Errorf("deviceNewtonSchulz: len %d != %d*%d", len(input), rows, cols)
	}
	if uint64(len(input)) > math.MaxUint32 {
		return nil, fmt.Errorf("deviceNewtonSchulz: matrix too large for a 32-bit element count")
	}
	result := make([]float32, len(input))
	copy(result, input)
	err := worker.Do(context.Background(), func(state *device.State) error {
		ops, err := newDeviceOps(state)
		if err != nil {
			return err
		}
		defer ops.close()

		n := rows * cols
		gramN := gramDim(rows, cols)
		gramN *= gramN
		buffers, err := ops.allocBuffers(n, gramN)
		if err != nil {
			return err
		}
		defer buffers.free(ops.lib)

		if err := ops.lib.MemcpyHtoD(buffers.dX, driver.Bytes(result)); err != nil {
			return err
		}
		final, err := ops.newtonSchulz(buffers, rows, cols)
		if err != nil {
			return err
		}
		if err := ops.lib.StreamSynchronize(ops.stream); err != nil {
			return err
		}
		return ops.lib.MemcpyDtoH(driver.Bytes(result), final)
	})
	if err != nil {
		return nil, err
	}
	return result, nil
}
