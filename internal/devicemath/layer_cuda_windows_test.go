//go:build windows

package devicemath

import (
	"math"
	"math/rand"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
)

func f64to32(v []float64) []float32 {
	out := make([]float32, len(v))
	for i := range v {
		out[i] = float32(v[i])
	}
	return out
}

func rmsNormForward(x, w []float32, rows, d int, eps float64) []float32 {
	out := make([]float32, rows*d)
	for r := 0; r < rows; r++ {
		var ss float64
		for i := 0; i < d; i++ {
			v := float64(x[r*d+i])
			ss += v * v
		}
		inv := 1.0 / math.Sqrt(ss/float64(d)+eps)
		for i := 0; i < d; i++ {
			out[r*d+i] = float32(float64(x[r*d+i]) * inv * float64(w[i]))
		}
	}
	return out
}

// hostLayer computes the pre-norm transformer layer, returning the fp32 forward
// cache (as the device backward consumes) and the float64 output.
func hostLayer(x []float32, w LayerWeights, dims LayerDims) (c LayerCache, y []float64) {
	seq, d, nh, nkv, hd, inter := dims.Seq, dims.D, dims.NH, dims.NKV, dims.HD, dims.Inter
	c.N1 = rmsNormForward(x, w.Norm1, seq, d, dims.Eps)
	c.Q = f64to32(matmulF64(c.N1, w.WQ, seq, d, nh*hd))
	c.K = f64to32(matmulF64(c.N1, w.WK, seq, d, nkv*hd))
	c.V = f64to32(matmulF64(c.N1, w.WV, seq, d, nkv*hd))
	var attn []float64
	c.P, attn = hostMHA(c.Q, c.K, c.V, seq, nh, nkv, hd, dims.Scale)
	c.Attn = f64to32(attn)
	o := matmulF64(c.Attn, w.WO, seq, nh*hd, d)
	c.X2 = make([]float32, seq*d)
	for i := range c.X2 {
		c.X2[i] = float32(float64(x[i]) + o[i])
	}
	c.N2 = rmsNormForward(c.X2, w.Norm2, seq, d, dims.Eps)
	gF := matmulF64(c.N2, w.WGate, seq, d, inter)
	uF := matmulF64(c.N2, w.WUp, seq, d, inter)
	c.G = f64to32(gF)
	c.U = f64to32(uF)
	c.A = make([]float32, seq*inter)
	c.H = make([]float32, seq*inter)
	for i := range gF {
		c.A[i] = float32(hostSiLU(gF[i]))
		c.H[i] = c.A[i] * c.U[i]
	}
	mlp := matmulF64(c.H, w.WDown, seq, inter, d)
	y = make([]float64, seq*d)
	for i := range y {
		y[i] = float64(c.X2[i]) + mlp[i]
	}
	return c, y
}

// TestLayerBackwardGradCheck verifies the composed full-layer device backward
// against float64 finite differences of L = sum(dy ⊙ y) for the input and every
// weight -- the end-to-end proof that the device operators compose into a
// transformer-layer VJP.
func TestLayerBackwardGradCheck(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	dims := LayerDims{Seq: 4, D: 8, NH: 2, NKV: 1, HD: 4, Inter: 12, Eps: 1e-6}
	dims.Scale = 1 / math.Sqrt(float64(dims.HD))
	rng := rand.New(rand.NewSource(11))
	x := randSlice(rng, dims.Seq*dims.D)
	w := LayerWeights{
		Norm1: randSlice(rng, dims.D), Norm2: randSlice(rng, dims.D),
		WQ: randSlice(rng, dims.D*dims.NH*dims.HD), WK: randSlice(rng, dims.D*dims.NKV*dims.HD),
		WV: randSlice(rng, dims.D*dims.NKV*dims.HD), WO: randSlice(rng, dims.NH*dims.HD*dims.D),
		WGate: randSlice(rng, dims.D*dims.Inter), WUp: randSlice(rng, dims.D*dims.Inter),
		WDown: randSlice(rng, dims.Inter*dims.D),
	}
	dy := randSlice(rng, dims.Seq*dims.D)

	cache, _ := hostLayer(x, w, dims)
	grads, err := LayerBackward(worker, x, w, cache, dy, dims)
	if err != nil {
		t.Fatal(err)
	}

	loss := func() float64 {
		_, y := hostLayer(x, w, dims)
		var l float64
		for i := range y {
			l += float64(dy[i]) * y[i]
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
	worst := 0.0
	for _, ch := range []struct {
		name            string
		param, analytic []float32
	}{
		{"dx", x, grads.DX},
		{"dNorm1", w.Norm1, grads.DNorm1}, {"dNorm2", w.Norm2, grads.DNorm2},
		{"dWQ", w.WQ, grads.DWQ}, {"dWK", w.WK, grads.DWK}, {"dWV", w.WV, grads.DWV},
		{"dWO", w.WO, grads.DWO},
		{"dWGate", w.WGate, grads.DWGate}, {"dWUp", w.WUp, grads.DWUp}, {"dWDown", w.WDown, grads.DWDown},
	} {
		worst = math.Max(worst, gradCheck(ch.name, ch.param, ch.analytic))
	}
	const tolerance = 1e-2
	if worst > tolerance {
		t.Fatalf("worst grad-check %.3e > %.1e", worst, tolerance)
	}
}
