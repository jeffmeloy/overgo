//go:build windows

package optimizer

import (
	"math"
	"math/rand"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
)

// TestDeviceMuonMatrixStepMatchesHost gates one device (fp32) Muon matrix update
// against the host optimizer's Step (fp64 Newton-Schulz) for square/tall/wide
// groups: both the updated weights and the momentum must match within tolerance.
func TestDeviceMuonMatrixStepMatchesHost(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	const mu, baseLR = 0.9, 0.1
	cases := []struct {
		name       string
		rows, cols int
	}{
		{"square", 32, 32},
		{"tall", 64, 16},
		{"wide", 16, 64},
	}
	rng := rand.New(rand.NewSource(2))
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			n := tc.rows * tc.cols
			w := make([]float32, n)
			g := make([]float32, n)
			for i := range w {
				w[i] = float32(rng.NormFloat64())
				g[i] = float32(rng.NormFloat64())
			}

			// Host reference: one Muon matrix group, one Step.
			hostW := make([]float32, n)
			copy(hostW, w)
			hostG := make([]float32, n)
			copy(hostG, g)
			plan, err := CompilePlan(n, []GroupSpec{{Name: "w", Start: 0, End: n, Rows: tc.rows, Cols: tc.cols}})
			if err != nil {
				t.Fatal(err)
			}
			opt, err := New(hostW, hostG, plan, Config{BaseLearningRate: baseLR, Momentum: mu, Schedule: ScheduleConstant})
			if err != nil {
				t.Fatal(err)
			}
			opt.Step()

			// Device: same inputs, zero initial momentum.
			devW := make([]float32, n)
			copy(devW, w)
			devG := make([]float32, n)
			copy(devG, g)
			devM := make([]float32, n)
			if err := deviceMuonMatrixStep(worker, devW, devG, devM, tc.rows, tc.cols, mu, baseLR); err != nil {
				t.Fatal(err)
			}

			var maxW, maxM float64
			for i := 0; i < n; i++ {
				if d := math.Abs(float64(devW[i]) - float64(hostW[i])); d > maxW {
					maxW = d
				}
				if d := math.Abs(float64(devM[i]) - opt.momentum[i]); d > maxM {
					maxM = d
				}
			}
			t.Logf("%s (%dx%d): max abs weight %.3e momentum %.3e", tc.name, tc.rows, tc.cols, maxW, maxM)
			// Observed weight ~2.4e-7 (fp32 NS precision), momentum exact.
			const tolerance = 1e-5
			if maxW > tolerance || maxM > tolerance {
				t.Fatalf("%s: weight %.3e / momentum %.3e > %.1e", tc.name, maxW, maxM, tolerance)
			}
		})
	}
}
