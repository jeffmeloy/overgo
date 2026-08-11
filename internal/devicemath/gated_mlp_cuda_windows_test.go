//go:build windows

package devicemath

import (
	"math"
	"math/rand"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
)

// matmulF64 computes row-major A·B in full float64 (A[m,k], B[k,n]).
func matmulF64(a, b []float32, m, k, n int) []float64 {
	out := make([]float64, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var s float64
			for p := 0; p < k; p++ {
				s += float64(a[i*k+p]) * float64(b[p*n+j])
			}
			out[i*n+j] = s
		}
	}
	return out
}

// gatedMLPLossF64 returns L = sum(dY ⊙ Y) for the SwiGLU block in float64 -- an
// accurate finite-difference reference (the device path is fp32).
func gatedMLPLossF64(x, wGate, wUp, wDown, dY []float32, rows, d, inter int) float64 {
	g := matmulF64(x, wGate, rows, d, inter)
	h := make([]float32, len(g))
	for i := range g {
		a := g[i] / (1 + math.Exp(-g[i])) // silu
		u := 0.0
		// u = (X·Wup)[i]
		r, c := i/inter, i%inter
		for p := 0; p < d; p++ {
			u += float64(x[r*d+p]) * float64(wUp[p*inter+c])
		}
		h[i] = float32(a * u)
	}
	y := matmulF64(h, wDown, rows, inter, d)
	var loss float64
	for i := range y {
		loss += float64(dY[i]) * y[i]
	}
	return loss
}

// hostGatedMLP32 returns the fp32 forward intermediates the device backward
// consumes (g, a, u, h all [rows,inter]).
func hostGatedMLP32(x, wGate, wUp, wDown []float32, rows, d, inter int) (g, a, u, h []float32) {
	gf := matmulF64(x, wGate, rows, d, inter)
	uf := matmulF64(x, wUp, rows, d, inter)
	g = make([]float32, len(gf))
	a = make([]float32, len(gf))
	u = make([]float32, len(gf))
	h = make([]float32, len(gf))
	for i := range gf {
		g[i] = float32(gf[i])
		a[i] = float32(gf[i] / (1 + math.Exp(-gf[i])))
		u[i] = float32(uf[i])
		h[i] = a[i] * u[i]
	}
	return g, a, u, h
}

// TestGatedMLPBackwardGradCheck verifies the composed device SwiGLU backward
// against float64 central finite differences of L = sum(dY ⊙ Y) for X and all
// three weight matrices.
func TestGatedMLPBackwardGradCheck(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	const rows, d, inter = 4, 5, 6
	rng := rand.New(rand.NewSource(6))
	x := randSlice(rng, rows*d)
	wGate := randSlice(rng, d*inter)
	wUp := randSlice(rng, d*inter)
	wDown := randSlice(rng, inter*d)
	dY := randSlice(rng, rows*d)

	g, a, u, h := hostGatedMLP32(x, wGate, wUp, wDown, rows, d, inter)
	grads, err := GatedMLPBackward(worker, x, wGate, wUp, wDown, g, a, u, h, dY, rows, d, inter)
	if err != nil {
		t.Fatal(err)
	}

	const eps = 1e-2 // large eps ok: float64 reference, low truncation for this smooth map
	gradCheck := func(name string, param, analytic []float32) float64 {
		var maxDiff float64
		for i := range param {
			orig := param[i]
			param[i] = orig + float32(eps)
			lp := gatedMLPLossF64(x, wGate, wUp, wDown, dY, rows, d, inter)
			param[i] = orig - float32(eps)
			lm := gatedMLPLossF64(x, wGate, wUp, wDown, dY, rows, d, inter)
			param[i] = orig
			if diff := math.Abs((lp-lm)/(2*eps) - float64(analytic[i])); diff > maxDiff {
				maxDiff = diff
			}
		}
		t.Logf("%s: max |grad-fd| %.3e", name, maxDiff)
		return maxDiff
	}
	const tolerance = 3e-3 // device fp32 across the composed ops
	worst := gradCheck("dX", x, grads.DX)
	worst = math.Max(worst, gradCheck("dWgate", wGate, grads.DWGate))
	worst = math.Max(worst, gradCheck("dWup", wUp, grads.DWUp))
	worst = math.Max(worst, gradCheck("dWdown", wDown, grads.DWDown))
	if worst > tolerance {
		t.Fatalf("worst grad-check %.3e > %.1e", worst, tolerance)
	}
}

// matmulTF64 computes row-major A·Bᵀ in float64 (A[m,k], B[n,k] -> [m,n]).
func matmulTF64(a, b []float32, m, k, n int) []float64 {
	out := make([]float64, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var s float64
			for p := 0; p < k; p++ {
				s += float64(a[i*k+p]) * float64(b[j*k+p])
			}
			out[i*n+j] = s
		}
	}
	return out
}

// gatedMLPLossTF64 returns L = sum(dY ⊙ Y) for the densecausal-convention SwiGLU
// (Y = X·Wᵀ everywhere) in float64.
func gatedMLPLossTF64(x, wGate, wUp, wDown, dY []float32, rows, d, inter int) float64 {
	g := matmulTF64(x, wGate, rows, d, inter)
	u := matmulTF64(x, wUp, rows, d, inter)
	h := make([]float32, len(g))
	for i := range g {
		a := g[i] / (1 + math.Exp(-g[i]))
		h[i] = float32(a * u[i])
	}
	y := matmulTF64(h, wDown, rows, inter, d)
	var loss float64
	for i := range y {
		loss += float64(dY[i]) * y[i]
	}
	return loss
}

// hostGatedMLPT32 returns the fp32 forward intermediates for the Wᵀ convention.
func hostGatedMLPT32(x, wGate, wUp, wDown []float32, rows, d, inter int) (g, a, u, h []float32) {
	gf := matmulTF64(x, wGate, rows, d, inter)
	uf := matmulTF64(x, wUp, rows, d, inter)
	g = make([]float32, len(gf))
	a = make([]float32, len(gf))
	u = make([]float32, len(gf))
	h = make([]float32, len(gf))
	for i := range gf {
		g[i] = float32(gf[i])
		a[i] = float32(gf[i] / (1 + math.Exp(-gf[i])))
		u[i] = float32(uf[i])
		h[i] = a[i] * u[i]
	}
	return g, a, u, h
}

// TestGatedMLPBackwardTGradCheck verifies GatedMLPBackwardT (densecausal Wᵀ
// convention) against float64 finite differences.
func TestGatedMLPBackwardTGradCheck(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	const rows, d, inter = 4, 5, 6
	rng := rand.New(rand.NewSource(66))
	x := randSlice(rng, rows*d)
	wGate := randSlice(rng, inter*d)
	wUp := randSlice(rng, inter*d)
	wDown := randSlice(rng, d*inter)
	dY := randSlice(rng, rows*d)

	g, a, u, h := hostGatedMLPT32(x, wGate, wUp, wDown, rows, d, inter)
	grads, err := GatedMLPBackwardT(worker, x, wGate, wUp, wDown, g, a, u, h, dY, rows, d, inter)
	if err != nil {
		t.Fatal(err)
	}
	const eps = 1e-2
	gradCheck := func(name string, param, analytic []float32) float64 {
		var maxDiff float64
		for i := range param {
			orig := param[i]
			param[i] = orig + float32(eps)
			lp := gatedMLPLossTF64(x, wGate, wUp, wDown, dY, rows, d, inter)
			param[i] = orig - float32(eps)
			lm := gatedMLPLossTF64(x, wGate, wUp, wDown, dY, rows, d, inter)
			param[i] = orig
			if diff := math.Abs((lp-lm)/(2*eps) - float64(analytic[i])); diff > maxDiff {
				maxDiff = diff
			}
		}
		t.Logf("%s: max |grad-fd| %.3e", name, maxDiff)
		return maxDiff
	}
	const tolerance = 3e-3
	worst := gradCheck("dX", x, grads.DX)
	worst = math.Max(worst, gradCheck("dWgate", wGate, grads.DWGate))
	worst = math.Max(worst, gradCheck("dWup", wUp, grads.DWUp))
	worst = math.Max(worst, gradCheck("dWdown", wDown, grads.DWDown))
	if worst > tolerance {
		t.Fatalf("worst grad-check %.3e > %.1e", worst, tolerance)
	}
}
