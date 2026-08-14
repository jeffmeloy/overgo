//go:build windows

package devicemath

import (
	"math"
	"math/rand"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
)

// hostAttention computes single-head causal attention in float64, returning the
// causal-masked softmax p (fp32, as the device backward consumes it) and the
// output.
func hostAttention(q, k, v []float32, seq, hd int, scale float64) (p []float32, out []float64) {
	p = make([]float32, seq*seq)
	out = make([]float64, seq*hd)
	row := make([]float64, seq)
	for i := 0; i < seq; i++ {
		mx := math.Inf(-1)
		for j := 0; j <= i; j++ {
			var dot float64
			for h := 0; h < hd; h++ {
				dot += float64(q[i*hd+h]) * float64(k[j*hd+h])
			}
			row[j] = dot * scale
			if row[j] > mx {
				mx = row[j]
			}
		}
		var sum float64
		for j := 0; j <= i; j++ {
			row[j] = math.Exp(row[j] - mx)
			sum += row[j]
		}
		for j := 0; j <= i; j++ {
			pij := row[j] / sum
			p[i*seq+j] = float32(pij)
			for h := 0; h < hd; h++ {
				out[i*hd+h] += pij * float64(v[j*hd+h])
			}
		}
	}
	return p, out
}

// TestAttentionCoreBackwardGradCheck verifies device AttentionCoreBackward
// (dQ/dK/dV) against float64 central finite differences of L = sum(dOut ⊙ out).
func TestAttentionCoreBackwardGradCheck(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	const seq, hd = 6, 4
	scale := 1 / math.Sqrt(float64(hd))
	rng := rand.New(rand.NewSource(9))
	q := randSlice(rng, seq*hd)
	k := randSlice(rng, seq*hd)
	v := randSlice(rng, seq*hd)
	dOut := randSlice(rng, seq*hd)

	p, _ := hostAttention(q, k, v, seq, hd, scale)
	grads, err := AttentionCoreBackward(worker, q, k, v, p, dOut, seq, hd, scale)
	if err != nil {
		t.Fatal(err)
	}

	loss := func() float64 {
		_, out := hostAttention(q, k, v, seq, hd, scale)
		var l float64
		for i := range out {
			l += float64(dOut[i]) * out[i]
		}
		return l
	}
	const eps = 1e-3
	gradCheck := func(name string, param, analytic []float32) float64 {
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
	const tolerance = 3e-3
	worstScore := 0.0
	for query := range seq {
		dp := make([]float64, seq)
		var expectation float64
		for key := 0; key <= query; key++ {
			for channel := range hd {
				dp[key] += float64(dOut[query*hd+channel]) * float64(v[key*hd+channel])
			}
			expectation += float64(p[query*seq+key]) * dp[key]
		}
		for key := range seq {
			want := float64(p[query*seq+key]) * (dp[key] - expectation)
			worstScore = max(worstScore, math.Abs(float64(grads.DScores[query*seq+key])-want))
		}
	}
	t.Logf("dScores: max |device-host| %.3e", worstScore)
	worst := gradCheck("dQ", q, grads.DQ)
	worst = math.Max(worst, gradCheck("dK", k, grads.DK))
	worst = math.Max(worst, gradCheck("dV", v, grads.DV))
	worst = math.Max(worst, worstScore)
	if worst > tolerance {
		t.Fatalf("worst grad-check %.3e > %.1e", worst, tolerance)
	}
}
