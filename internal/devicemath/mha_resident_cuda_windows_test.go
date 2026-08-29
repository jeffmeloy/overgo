//go:build windows

package devicemath

import (
	"math"
	"math/rand"
	"testing"
	"time"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
)

// TestMultiHeadAttentionBackwardResidentMatchesPerOp checks the resident GQA
// attention backward against the per-op MultiHeadAttentionBackward (same math)
// and reports each one's wall time (one worker.Do vs nh), per SQA finding 2.
func TestMultiHeadAttentionBackwardResidentMatchesPerOp(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	const seq, nh, nkv, hd = 32, 8, 2, 16
	scale := 1 / math.Sqrt(float64(hd))
	rng := rand.New(rand.NewSource(88))
	q := randSlice(rng, seq*nh*hd)
	k := randSlice(rng, seq*nkv*hd)
	v := randSlice(rng, seq*nkv*hd)
	dOut := randSlice(rng, seq*nh*hd)
	p := hostCausalSoftmaxGQA(q, k, seq, nh, nkv, hd)

	dQp, dKp, dVp, err := MultiHeadAttentionBackward(worker, q, k, v, p, dOut, seq, nh, nkv, hd, scale)
	if err != nil {
		t.Fatal(err)
	}
	dQr, dKr, dVr, err := MultiHeadAttentionBackwardResident(worker, q, k, v, dOut, seq, nh, nkv, hd, scale)
	if err != nil {
		t.Fatal(err)
	}

	maxAbs := func(a, b []float32) float64 {
		var m float64
		for i := range a {
			if x := math.Abs(float64(a[i]) - float64(b[i])); x > m {
				m = x
			}
		}
		return m
	}
	const tolerance = 1e-4
	t.Logf("dQ %.3e dK %.3e dV %.3e", maxAbs(dQp, dQr), maxAbs(dKp, dKr), maxAbs(dVp, dVr))
	for _, c := range []struct {
		name string
		a, b []float32
	}{{"dQ", dQp, dQr}, {"dK", dKp, dKr}, {"dV", dVp, dVr}} {
		if v := maxAbs(c.a, c.b); v > tolerance {
			t.Fatalf("%s resident vs per-op %.3e > %.1e", c.name, v, tolerance)
		}
	}

	const iters = 20
	timeIt := func(fn func() error) time.Duration {
		if err := fn(); err != nil {
			t.Fatal(err)
		}
		start := time.Now()
		for range iters {
			if err := fn(); err != nil {
				t.Fatal(err)
			}
		}
		return time.Since(start) / iters
	}
	perOpT := timeIt(func() error {
		_, _, _, e := MultiHeadAttentionBackward(worker, q, k, v, p, dOut, seq, nh, nkv, hd, scale)
		return e
	})
	resT := timeIt(func() error {
		_, _, _, e := MultiHeadAttentionBackwardResident(worker, q, k, v, dOut, seq, nh, nkv, hd, scale)
		return e
	})
	t.Logf("MHA backward per-call: per-op %v, resident %v (%.2fx)", perOpT, resT, float64(perOpT)/float64(resT))
}
