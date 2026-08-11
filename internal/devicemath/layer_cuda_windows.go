//go:build windows

package devicemath

import (
	"fmt"

	"overgo/internal/cuda/device"
)

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
