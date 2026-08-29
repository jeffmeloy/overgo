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

// TestGatedMLPBackwardTResidentMatchesPerOp checks the resident SwiGLU MLP
// backward against the per-op GatedMLPBackwardT (must match, same math) and
// reports the wall-time of each so the residency win (one worker.Do vs four) is
// measured, per SQA findings 2/3.
func TestGatedMLPBackwardTResidentMatchesPerOp(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	const rows, d, inter = 32, 64, 128
	rng := rand.New(rand.NewSource(77))
	x := randSlice(rng, rows*d)
	wGate := randSlice(rng, inter*d)
	wUp := randSlice(rng, inter*d)
	wDown := randSlice(rng, d*inter)
	dY := randSlice(rng, rows*d)
	g, a, u, h := hostGatedMLPT32(x, wGate, wUp, wDown, rows, d, inter)

	perOp, err := GatedMLPBackwardT(worker, x, wGate, wUp, wDown, g, a, u, h, dY, rows, d, inter)
	if err != nil {
		t.Fatal(err)
	}
	res, err := GatedMLPBackwardTResident(worker, x, wGate, wUp, wDown, g, a, u, h, dY, rows, d, inter)
	if err != nil {
		t.Fatal(err)
	}

	maxAbs := func(p, q []float32) float64 {
		var m float64
		for i := range p {
			if v := math.Abs(float64(p[i]) - float64(q[i])); v > m {
				m = v
			}
		}
		return m
	}
	const tolerance = 1e-4
	for _, c := range []struct {
		name string
		a, b []float32
	}{{"dX", perOp.DX, res.DX}, {"dWgate", perOp.DWGate, res.DWGate}, {"dWup", perOp.DWUp, res.DWUp}, {"dWdown", perOp.DWDown, res.DWDown}} {
		if v := maxAbs(c.a, c.b); v > tolerance {
			t.Fatalf("%s resident vs per-op %.3e > %.1e", c.name, v, tolerance)
		}
	}

	const iters = 30
	timeIt := func(fn func() error) time.Duration {
		if err := fn(); err != nil { // warm up
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
		_, e := GatedMLPBackwardT(worker, x, wGate, wUp, wDown, g, a, u, h, dY, rows, d, inter)
		return e
	})
	resT := timeIt(func() error {
		_, e := GatedMLPBackwardTResident(worker, x, wGate, wUp, wDown, g, a, u, h, dY, rows, d, inter)
		return e
	})
	t.Logf("MLP backward per-call: per-op %v, resident %v (%.2fx)", perOpT, resT, float64(perOpT)/float64(resT))
}
