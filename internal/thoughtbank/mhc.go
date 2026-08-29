package thoughtbank

import (
	"fmt"
	"math"

	"overgo/internal/hostmath"
)

// Manifold-constrained hyper-connections (mHC).
//
// A hyper-connection widens the residual stream by a factor n_hc and replaces
// the scalar residual add with a learned mixing of the streams:
//
//	X_{l+1} = B_l X_l + C_l F_l(A_l X_l)
//
// A, B and C are generated per token from the current state. What makes the
// variant "manifold-constrained" is that B is projected onto the Birkhoff
// polytope (doubly stochastic) instead of being free: that bounds its spectral
// norm at 1, so stacking layers cannot compound the residual norm. A and C are
// bounded by sigmoid instead, C at twice the range so a stream can be
// amplified as well as attenuated.
//
// The constraint placement is not decorative and is reproduced exactly:
//   - A and C take their dynamic part through tanh(alpha) * projection.
//   - B takes tanh(alpha) * tanh(projection). The EXTRA inner tanh is the one
//     that matters: B's logits feed exp() inside Sinkhorn, so an unbounded
//     projection there drives both the value and its gradient to infinity as
//     the generating weights grow. A and C never reach an exponential.
//
// The static biases are initialised so the whole block starts near identity:
// S_res is a flattened identity matrix and S_post is 1/n_hc.

// HyperConnectionWeights is one mHC block's parameters, in the reference's
// layout. Projection weights are [out, in] row-major, matching a torch
// nn.Linear(bias=False).
type HyperConnectionWeights struct {
	NHC    int
	DModel int
	// Sinkhorn sweeps; the reference ships 20.
	SinkhornIters int

	WPre  []float32 // [n_hc, n_hc*d]
	WRes  []float32 // [n_hc*n_hc, n_hc*d]
	WPost []float32 // [n_hc, n_hc*d]

	SPre  []float32 // [n_hc]
	SRes  []float32 // [n_hc*n_hc]
	SPost []float32 // [n_hc]

	AlphaPre  float32
	AlphaRes  float32
	AlphaPost float32

	NormWeight []float32 // [n_hc*d] RMSNorm gain over the FLATTENED streams
	NormEps    float64
}

// HyperConnectionLayerFn is the wrapped sub-layer: it maps rows x d to
// rows x d in place-independent fashion (input may not be written).
type HyperConnectionLayerFn func(in []float32, rows, d int) []float32

// Validate reports a parameter bundle whose shapes do not agree with NHC/DModel.
func (w *HyperConnectionWeights) Validate() error {
	if w.NHC < 1 || w.DModel < 1 {
		return fmt.Errorf("mHC: n_hc=%d d_model=%d must be positive", w.NHC, w.DModel)
	}
	flat := w.NHC * w.DModel
	for _, c := range []struct {
		name string
		got  int
		want int
	}{
		{"W_pre", len(w.WPre), w.NHC * flat},
		{"W_res", len(w.WRes), w.NHC * w.NHC * flat},
		{"W_post", len(w.WPost), w.NHC * flat},
		{"S_pre", len(w.SPre), w.NHC},
		{"S_res", len(w.SRes), w.NHC * w.NHC},
		{"S_post", len(w.SPost), w.NHC},
		{"norm", len(w.NormWeight), flat},
	} {
		if c.got != c.want {
			return fmt.Errorf("mHC: %s has %d values, want %d", c.name, c.got, c.want)
		}
	}
	return nil
}

// HyperConnectionForward applies one mHC-wrapped sub-layer.
//
// x holds rows x n_hc x d values (row-major, stream-major within a row) and is
// not modified. The result has the same shape.
func HyperConnectionForward(x []float32, rows int, w *HyperConnectionWeights, layer HyperConnectionLayerFn) ([]float32, error) {
	if err := w.Validate(); err != nil {
		return nil, err
	}
	n, d := w.NHC, w.DModel
	flat := n * d
	if len(x) != rows*flat {
		return nil, fmt.Errorf("mHC: input has %d values, want %d", len(x), rows*flat)
	}

	// Parameter generation reads the NORMALISED flattened state; the residual
	// update below reads the raw state. Both are needed, so the norm cannot be
	// done in place.
	xHat := rmsNormNew(x, w.NormWeight, rows, flat, w.NormEps)

	aPre := float64(tanh32(w.AlphaPre))
	aRes := float64(tanh32(w.AlphaRes))
	aPost := float64(tanh32(w.AlphaPost))

	aGate := make([]float32, rows*n)     // sigmoid(A_raw)
	cGate := make([]float32, rows*n)     // 2*sigmoid(C_raw)
	bLogits := make([]float32, rows*n*n) // pre-Sinkhorn
	for r := range rows {
		hat := xHat[r*flat : (r+1)*flat]
		for i := range n {
			aGate[r*n+i] = float32(sigmoid(aPre*dot(w.WPre[i*flat:(i+1)*flat], hat) + float64(w.SPre[i])))
			cGate[r*n+i] = float32(2.0 * sigmoid(aPost*dot(w.WPost[i*flat:(i+1)*flat], hat)+float64(w.SPost[i])))
		}
		for i := range n * n {
			// Inner tanh before the alpha gate: see the type comment.
			inner := math.Tanh(dot(w.WRes[i*flat:(i+1)*flat], hat))
			bLogits[r*n*n+i] = float32(aRes*inner + float64(w.SRes[i]))
		}
	}

	hostmath.SinkhornFromLogitsInPlace(bLogits, rows, n, w.SinkhornIters)

	// h_in = sum_n A[n] * X[n]  -- collapse the streams into the sub-layer input.
	hIn := make([]float32, rows*d)
	for r := range rows {
		out := hIn[r*d : (r+1)*d]
		for s := range n {
			g := float64(aGate[r*n+s])
			if g == 0 {
				continue
			}
			stream := x[r*flat+s*d : r*flat+(s+1)*d]
			for i, v := range stream {
				out[i] += float32(g * float64(v))
			}
		}
	}

	hOut := layer(hIn, rows, d)
	if len(hOut) != rows*d {
		return nil, fmt.Errorf("mHC: sub-layer returned %d values, want %d", len(hOut), rows*d)
	}

	// X_new = B_ds X + C (x) h_out
	res := make([]float32, rows*flat)
	for r := range rows {
		bm := bLogits[r*n*n : (r+1)*n*n]
		for i := range n {
			dst := res[r*flat+i*d : r*flat+(i+1)*d]
			for j := range n {
				coeff := float64(bm[i*n+j])
				if coeff == 0 {
					continue
				}
				src := x[r*flat+j*d : r*flat+(j+1)*d]
				for k, v := range src {
					dst[k] += float32(coeff * float64(v))
				}
			}
			c := float64(cGate[r*n+i])
			row := hOut[r*d : (r+1)*d]
			for k, v := range row {
				dst[k] += float32(c * float64(v))
			}
		}
	}
	return res, nil
}

func hyperConnectionVector(x []float32, weights *HyperConnectionWeights, layer HyperConnectionLayerFn) ([]float32, error) {
	return HyperConnectionForward(x, len(x)/(weights.NHC*weights.DModel), weights, layer)
}
