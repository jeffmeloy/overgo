//go:build windows

package devicemath

import (
	"math/rand"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/hostmath"
	"overgo/internal/testutil"
)

// TestL2NormBackwardDeviceMatchesHost pins L2NormBackwardDevice against the
// hostmath.L2NormBackward f64 golden. Two cases exercise both branches of the
// VJP: eps=1e-6 keeps every row unclamped (norm>eps, full inv/inv^3 term); a
// large eps clamps rows (norm<=eps -> dX=inv*dY, no dot term).
func TestL2NormBackwardDeviceMatchesHost(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	cases := []struct {
		name        string
		rows, width int
		scale       float32
		eps         float64
	}{
		{"unclamped", 12, 8, 1.0, 1e-6},
		{"clamped", 10, 6, 0.3, 1.5},
		{"wide", 5, 33, 0.7, 1e-6},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(101 + tc.rows*7 + tc.width)))
			rs := func(n int) []float32 {
				s := make([]float32, n)
				for i := range s {
					s[i] = float32(rng.NormFloat64()) * tc.scale
				}
				return s
			}
			x := rs(tc.rows * tc.width)
			dY := rs(tc.rows * tc.width)

			want := hostmath.L2NormBackward(x, dY, tc.rows, tc.width, tc.eps)
			got, err := L2NormBackwardDevice(worker, x, dY, tc.rows, tc.width, tc.eps)
			if err != nil {
				t.Fatal(err)
			}
			diff := testutil.MaxAbsDiff(got, want)
			t.Logf("L2NormBackward %-9s max|device-host| %.3e", tc.name, diff)
			const tol = 1e-5
			if diff > tol {
				t.Errorf("L2NormBackward %s: max|device-host| %.3e > %.1e", tc.name, diff, tol)
			}
		})
	}
}
