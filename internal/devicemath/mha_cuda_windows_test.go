//go:build windows

package devicemath

import (
	"math"
	"math/rand"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
)

// hostMHA computes multi-head causal GQA attention in float64, returning the
// per-head causal softmax p [nh,seq,seq] (fp32) and the output [seq, nh*hd].
func hostMHA(q, k, v []float32, seq, nh, nkv, hd int, scale float64) (p []float32, out []float64) {
	p = make([]float32, nh*seq*seq)
	out = make([]float64, seq*nh*hd)
	headsPerKV := nh / nkv
	row := make([]float64, seq)
	for h := range nh {
		kvh := h / headsPerKV
		for i := range seq {
			mx := math.Inf(-1)
			for j := 0; j <= i; j++ {
				var dot float64
				for c := range hd {
					dot += float64(q[i*nh*hd+h*hd+c]) * float64(k[j*nkv*hd+kvh*hd+c])
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
				p[h*seq*seq+i*seq+j] = float32(pij)
				for c := range hd {
					out[i*nh*hd+h*hd+c] += pij * float64(v[j*nkv*hd+kvh*hd+c])
				}
			}
		}
	}
	return p, out
}

// TestMultiHeadAttentionBackwardGradCheck verifies device
// MultiHeadAttentionBackward (GQA) against float64 finite differences of
// L = sum(dOut ⊙ out) for Q, K, V.
func TestMultiHeadAttentionBackwardGradCheck(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	const seq, nh, nkv, hd = 5, 4, 2, 4 // GQA: 4 query heads, 2 KV heads
	scale := 1 / math.Sqrt(float64(hd))
	rng := rand.New(rand.NewSource(10))
	q := randSlice(rng, seq*nh*hd)
	k := randSlice(rng, seq*nkv*hd)
	v := randSlice(rng, seq*nkv*hd)
	dOut := randSlice(rng, seq*nh*hd)

	p, _ := hostMHA(q, k, v, seq, nh, nkv, hd, scale)
	dQ, dK, dV, err := MultiHeadAttentionBackward(worker, q, k, v, p, dOut, seq, nh, nkv, hd, scale)
	if err != nil {
		t.Fatal(err)
	}

	loss := func() float64 {
		_, out := hostMHA(q, k, v, seq, nh, nkv, hd, scale)
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
	worst := gradCheck("dQ", q, dQ)
	worst = max(worst, gradCheck("dK", k, dK))
	worst = max(worst, gradCheck("dV", v, dV))
	if worst > tolerance {
		t.Fatalf("worst grad-check %.3e > %.1e", worst, tolerance)
	}
}
