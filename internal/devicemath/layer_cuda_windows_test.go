//go:build windows

package devicemath

import (
	"fmt"

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
	for r := range rows {
		var ss float64
		for i := range d {
			v := float64(x[r*d+i])
			ss += v * v
		}
		inv := 1.0 / math.Sqrt(ss/float64(d)+eps)
		for i := range d {
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
		worst = max(worst, gradCheck(ch.name, ch.param, ch.analytic))
	}
	const tolerance = 1e-2
	if worst > tolerance {
		t.Fatalf("worst grad-check %.3e > %.1e", worst, tolerance)
	}
}

// LayerBackward composes the verified device operators into the VJP of one
// pre-norm transformer layer:
//
//	n1 = RMSNorm(x, Norm1); Q,K,V = n1·{WQ,WK,WV}; attn = MHA(Q,K,V);
//	o = attn·WO; x2 = x + o; n2 = RMSNorm(x2, Norm2);
//	mlp = SwiGLU(n2, WGate,WUp,WDown); y = x2 + mlp.
//
// Given dy and the saved cache, it returns dx and every weight gradient. The two
// residuals split the gradient: dx2 = dy + RMSNorm2-path; dx = dx2 + RMSNorm1-path.
func LayerBackward(worker *device.Worker, x []float32, w LayerWeights, c LayerCache, dy []float32, dims LayerDims) (LayerGrads, error) {
	seq, d, nh, nkv, hd, inter := dims.Seq, dims.D, dims.NH, dims.NKV, dims.HD, dims.Inter
	if len(x) != seq*d || len(dy) != seq*d {
		return LayerGrads{}, fmt.Errorf("LayerBackward: x/dy shape (seq=%d d=%d)", seq, d)
	}

	// y = x2 + mlp: dy flows to x2 (residual) and to the MLP.
	mlpG, err := GatedMLPBackward(worker, c.N2, w.WGate, w.WUp, w.WDown, c.G, c.A, c.U, c.H, dy, seq, d, inter)
	if err != nil {
		return LayerGrads{}, err
	}
	// n2 = RMSNorm(x2): dn2 -> dx2 (norm path) + weight grad.
	dx2Norm, dNorm2, err := RMSNormBackward(worker, c.X2, w.Norm2, mlpG.DX, seq, d, dims.Eps)
	if err != nil {
		return LayerGrads{}, err
	}
	dx2 := addInto(dy, dx2Norm)

	// x2 = x + o: dx2 flows to o (=> WO/attn) and to x (residual).
	dattn, dWO, err := LinearBackward(worker, c.Attn, w.WO, dx2, seq, nh*hd, d)
	if err != nil {
		return LayerGrads{}, err
	}
	dQ, dK, dV, err := MultiHeadAttentionBackward(worker, c.Q, c.K, c.V, c.P, dattn, seq, nh, nkv, hd, dims.Scale)
	if err != nil {
		return LayerGrads{}, err
	}
	dn1q, dWQ, err := LinearBackward(worker, c.N1, w.WQ, dQ, seq, d, nh*hd)
	if err != nil {
		return LayerGrads{}, err
	}
	dn1k, dWK, err := LinearBackward(worker, c.N1, w.WK, dK, seq, d, nkv*hd)
	if err != nil {
		return LayerGrads{}, err
	}
	dn1v, dWV, err := LinearBackward(worker, c.N1, w.WV, dV, seq, d, nkv*hd)
	if err != nil {
		return LayerGrads{}, err
	}
	dn1 := addInto(addInto(dn1q, dn1k), dn1v)

	// n1 = RMSNorm(x): dn1 -> dx (norm path) + weight grad.
	dxNorm1, dNorm1, err := RMSNormBackward(worker, x, w.Norm1, dn1, seq, d, dims.Eps)
	if err != nil {
		return LayerGrads{}, err
	}
	dx := addInto(dx2, dxNorm1) // residual (x2=x+o) + norm-1 path

	return LayerGrads{
		DX: dx, DNorm1: dNorm1, DNorm2: dNorm2,
		DWQ: dWQ, DWK: dWK, DWV: dWV, DWO: dWO,
		DWGate: mlpG.DWGate, DWUp: mlpG.DWUp, DWDown: mlpG.DWDown,
	}, nil
}

// LayerDims describes a pre-norm transformer layer's geometry.
type LayerDims struct {
	Seq, D, NH, NKV, HD, Inter int
	Scale, Eps                 float64
}

// LayerWeights holds a layer's parameters.
type LayerWeights struct {
	Norm1, Norm2      []float32 // [d]
	WQ, WK, WV, WO    []float32 // [d,nh*hd] [d,nkv*hd] [d,nkv*hd] [nh*hd,d]
	WGate, WUp, WDown []float32 // [d,inter] [d,inter] [inter,d]
}

// LayerCache holds the forward intermediates the backward consumes.
type LayerCache struct {
	N1, N2     []float32 // [seq,d]
	Q, K, V    []float32 // [seq,nh*hd] [seq,nkv*hd] [seq,nkv*hd]
	P          []float32 // [nh,seq,seq] causal softmax
	Attn       []float32 // [seq,nh*hd]
	X2         []float32 // [seq,d] (x + attention output)
	G, A, U, H []float32 // [seq,inter]
}

// LayerGrads holds the input and parameter gradients of a layer.
type LayerGrads struct {
	DX                   []float32
	DNorm1, DNorm2       []float32
	DWQ, DWK, DWV, DWO   []float32
	DWGate, DWUp, DWDown []float32
}

func addInto(a, b []float32) []float32 {
	out := make([]float32, len(a))
	for i := range a {
		out[i] = a[i] + b[i]
	}
	return out
}
