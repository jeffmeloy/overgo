//go:build windows

package devicemath

import (
	"math"
	"math/rand"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/hostmath"
)

// hostCausalSoftmaxGQA computes the per-head causal softmax p[nh*seq*seq] the
// device MHA consumes, using exactly hostmath.CausalAttentionBackward's scoring
// (scale=1, GQA kv=h/group, keys 0..qi). p[h*seq*seq + qi*seq + m].
func hostCausalSoftmaxGQA(q, k []float32, seq, nh, nkv, hd int) []float32 {
	group := nh / nkv
	p := make([]float32, nh*seq*seq)
	row := make([]float64, seq)
	for h := range nh {
		kv := h / group
		for qi := range seq {
			nk := qi + 1
			mx := math.Inf(-1)
			for m := range nk {
				var dot float64
				for x := range hd {
					dot += float64(q[(qi*nh+h)*hd+x]) * float64(k[(m*nkv+kv)*hd+x])
				}
				row[m] = dot
				mx = max(mx, dot)
			}
			var sum float64
			for m := range nk {
				row[m] = math.Exp(row[m] - mx)
				sum += row[m]
			}
			for m := range nk {
				p[h*seq*seq+qi*seq+m] = float32(row[m] / sum)
			}
		}
	}
	return p
}

// TestMultiHeadAttentionBackwardMatchesHostCausal proves the device MHA (scale=1,
// host-computed causal softmax p) reproduces hostmath.CausalAttentionBackward
// exactly -- the convention (layout, GQA accumulation, scale) densecausal uses.
// This de-risks the densecausal-exact device layer backward.
func TestMultiHeadAttentionBackwardMatchesHostCausal(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	const seq, nh, nkv, hd = 6, 4, 2, 8
	rng := rand.New(rand.NewSource(21))
	q := randSlice(rng, seq*nh*hd)
	k := randSlice(rng, seq*nkv*hd)
	v := randSlice(rng, seq*nkv*hd)
	dOut := randSlice(rng, seq*nh*hd)

	p := hostCausalSoftmaxGQA(q, k, seq, nh, nkv, hd)
	dQ, dK, dV, err := MultiHeadAttentionBackward(worker, q, k, v, p, dOut, seq, nh, nkv, hd, 1.0)
	if err != nil {
		t.Fatal(err)
	}

	hdq := make([]float32, seq*nh*hd)
	hdk := make([]float32, seq*nkv*hd)
	hdv := make([]float32, seq*nkv*hd)
	hostmath.CausalAttentionBackward(hdq, hdk, hdv, q, k, v, dOut, seq, nh, nkv, hd)

	maxDiff := func(got, want []float32) float64 {
		var m float64
		for i := range got {
			if d := math.Abs(float64(got[i]) - float64(want[i])); d > m {
				m = d
			}
		}
		return m
	}
	dqD, dkD, dvD := maxDiff(dQ, hdq), maxDiff(dK, hdk), maxDiff(dV, hdv)
	t.Logf("device vs host CausalAttentionBackward: dQ %.3e dK %.3e dV %.3e", dqD, dkD, dvD)
	const tolerance = 1e-4
	if dqD > tolerance || dkD > tolerance || dvD > tolerance {
		t.Fatalf("dQ %.3e / dK %.3e / dV %.3e > %.1e", dqD, dkD, dvD, tolerance)
	}
}
