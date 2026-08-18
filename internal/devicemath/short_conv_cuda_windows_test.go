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

// TestShortConvBackwardDeviceMatchesHost pins ShortConvBackwardDevice against the
// hostmath.ShortConvBackward f64 golden for dX, dW and dBias. Covers the biased
// path and the nil-bias path (kernel has_bias==0 -> no bias add, dBias nil).
func TestShortConvBackwardDeviceMatchesHost(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	cases := []struct {
		name             string
		channels, tokens int
		k                int
		bias             bool
	}{
		{"biased", 16, 7, 3, true},
		{"nobias", 16, 7, 3, false},
		{"k4_wide", 24, 9, 4, true},
		{"single_tap", 8, 5, 1, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(211 + tc.channels + tc.k*13 + tc.tokens)))
			rs := func(n int) []float32 {
				s := make([]float32, n)
				for i := range s {
					s[i] = float32(rng.NormFloat64()) * 0.4
				}
				return s
			}
			x := rs(tc.channels * tc.tokens)
			dY := rs(tc.channels * tc.tokens)
			w := rs(tc.channels * tc.k)
			var bias []float32
			if tc.bias {
				bias = rs(tc.channels)
			}

			wantDX, wantDW, wantDBias := hostmath.ShortConvBackward(x, dY, tc.channels, tc.tokens, w, bias, tc.k)
			gotDX, gotDW, gotDBias, err := ShortConvBackwardDevice(worker, x, dY, tc.channels, tc.tokens, w, bias, tc.k)
			if err != nil {
				t.Fatal(err)
			}
			const tol = 1e-5
			dxDiff := testutil.MaxAbsDiff(gotDX, wantDX)
			dwDiff := testutil.MaxAbsDiff(gotDW, wantDW)
			t.Logf("ShortConvBackward %-10s dX %.3e dW %.3e", tc.name, dxDiff, dwDiff)
			if dxDiff > tol {
				t.Errorf("ShortConvBackward %s: dX %.3e > %.1e", tc.name, dxDiff, tol)
			}
			if dwDiff > tol {
				t.Errorf("ShortConvBackward %s: dW %.3e > %.1e", tc.name, dwDiff, tol)
			}
			if tc.bias {
				if gotDBias == nil {
					t.Fatalf("ShortConvBackward %s: expected non-nil dBias", tc.name)
				}
				dbDiff := testutil.MaxAbsDiff(gotDBias, wantDBias)
				t.Logf("ShortConvBackward %-10s dBias %.3e", tc.name, dbDiff)
				if dbDiff > tol {
					t.Errorf("ShortConvBackward %s: dBias %.3e > %.1e", tc.name, dbDiff, tol)
				}
			} else if gotDBias != nil {
				t.Errorf("ShortConvBackward %s: expected nil dBias for nil bias", tc.name)
			}
		})
	}
}
