//go:build windows

package devicemath

import (
	"math"
	"math/rand"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
)

// hostSoftmaxRows returns row-wise softmax of s[rows,d] in float64.
func hostSoftmaxRows(s []float32, rows, d int) []float64 {
	p := make([]float64, rows*d)
	for r := range rows {
		mx := math.Inf(-1)
		for i := range d {
			if v := float64(s[r*d+i]); v > mx {
				mx = v
			}
		}
		var sum float64
		for i := range d {
			e := math.Exp(float64(s[r*d+i]) - mx)
			p[r*d+i] = e
			sum += e
		}
		for i := range d {
			p[r*d+i] /= sum
		}
	}
	return p
}

// TestSoftmaxBackwardGradCheck verifies device SoftmaxBackward against float64
// central finite differences of L = sum(dp ⊙ softmax(s)) w.r.t. s.
func TestSoftmaxBackwardGradCheck(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	const rows, d = 8, 12
	rng := rand.New(rand.NewSource(8))
	s := randSlice(rng, rows*d)
	dp := randSlice(rng, rows*d)

	pF64 := hostSoftmaxRows(s, rows, d)
	p := make([]float32, rows*d)
	for i := range pF64 {
		p[i] = float32(pF64[i])
	}
	ds, err := SoftmaxBackward(worker, p, dp, rows, d)
	if err != nil {
		t.Fatal(err)
	}

	loss := func() float64 {
		pp := hostSoftmaxRows(s, rows, d)
		var l float64
		for i := range pp {
			l += float64(dp[i]) * pp[i]
		}
		return l
	}
	const eps = 1e-3
	var maxDiff float64
	for i := range s {
		orig := s[i]
		s[i] = orig + float32(eps)
		lp := loss()
		s[i] = orig - float32(eps)
		lm := loss()
		s[i] = orig
		if diff := math.Abs((lp-lm)/(2*eps) - float64(ds[i])); diff > maxDiff {
			maxDiff = diff
		}
	}
	t.Logf("softmax backward: max |ds-fd| %.3e", maxDiff)
	const tolerance = 1e-3
	if maxDiff > tolerance {
		t.Fatalf("ds vs fd %.3e > %.1e", maxDiff, tolerance)
	}
}
