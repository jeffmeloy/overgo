//go:build windows

package optimizer

import (
	"math"
	"math/rand"
	"testing"

	"overgo/internal/cuda/device"
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
