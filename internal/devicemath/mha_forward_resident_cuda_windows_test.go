//go:build windows

package devicemath

import (
	"math"
	"math/rand"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/hostmath"
	"overgo/internal/testcheck"
)

// TestMultiHeadAttentionForwardResidentMatchesHost checks the resident device
// attention-core forward against hostmath.CausalAttention (same GQA causal math,
// scale folded into q upstream so scores use scale 1).
func TestMultiHeadAttentionForwardResidentMatchesHost(t *testing.T) {
	cudatest.Require(t)
	worker := testcheck.Must(device.New(0))(t)
	defer worker.Close()

	const seq, nh, nkv, hd = 24, 8, 2, 16
	rng := rand.New(rand.NewSource(7))
	q := randSlice(rng, seq*nh*hd)
	k := randSlice(rng, seq*nkv*hd)
	v := randSlice(rng, seq*nkv*hd)

	got := testcheck.Must(MultiHeadAttentionForwardResident(worker, q, k, v, seq, nh, nkv, hd))(t)
	want := make([]float32, seq*nh*hd)
	hostmath.CausalAttention(want, q, k, v, seq, nh, nkv, hd)

	var maxDiff float64
	for i := range want {
		if d := math.Abs(float64(got[i]) - float64(want[i])); d > maxDiff {
			maxDiff = d
		}
	}
	t.Logf("attn-core forward: max |dev-host| %.3e", maxDiff)
	const tolerance = 1e-4
	if maxDiff > tolerance {
		t.Fatalf("attn-core forward %.3e > %.1e", maxDiff, tolerance)
	}
}
