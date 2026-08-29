package thoughtbank

import (
	"fmt"
	"math"

	"overgo/internal/hostmath"
)

// Analytic gradient of the mHC block, with the sub-layer's gradient arriving as
// an INPUT rather than being computed here.
//
// That split is what makes the piece verifiable alone: the sub-layer is a
// callback the block knows nothing about, so a backward that tried to descend
// into it would be testing two things at once and could not attribute a
// disagreement to either.
//
// x reaches the output through FIVE paths, and every one accumulates into dX:
//
//	x -> rmsNorm -> W_pre  -> aGate -> h_in   -> (sub-layer) -> out
//	x -> rmsNorm -> W_res  -> B     -> B @ X  -> out
//	x -> rmsNorm -> W_post -> cGate -> out
//	x ->                      aGate * X       -> h_in -> (sub-layer) -> out
//	x ->                      B @ X           -> out
//
// The last two are the raw state; the first three go through the normalised
// copy. Assigning rather than accumulating anywhere here silently drops a path.
type HyperConnectionGradients struct {
	DX          []float32 // [rows, n_hc*d]
	DHOut       []float32 // [rows, d]: what the sub-layer's own backward consumes
	DWPre       []float32
	DWRes       []float32
	DWPost      []float32
	DSPre       []float32
	DSRes       []float32
	DSPost      []float32
	DAlphaPre   float32
	DAlphaRes   float32
	DAlphaPost  float32
	DNormWeight []float32
}

// HyperConnectionSubBackwardFn maps the gradient at the sub-layer's OUTPUT to
// the gradient at its INPUT. It is a callback for the same reason the forward's
// layer is: h_in feeds the sub-layer and dh_in can only come back through it, so
// the dependency is genuinely circular.
type HyperConnectionSubBackwardFn func(dHOut []float32, rows, d int) []float32

// HyperConnectionBackward computes the block's gradients. hOut is the sub-layer
// output the forward produced, and dRes is the gradient arriving at the block's
// output; the returned DHOut is what the sub-layer's backward should be called
// with.
func HyperConnectionBackward(x, hOut, dRes []float32, rows int,
	w *HyperConnectionWeights, subBackward HyperConnectionSubBackwardFn) (*HyperConnectionGradients, error) {
	if err := w.Validate(); err != nil {
		return nil, err
	}
	n, d := w.NHC, w.DModel
	flat := n * d
	if len(x) != rows*flat {
		return nil, fmt.Errorf("mHC backward: x has %d values, want %d", len(x), rows*flat)
	}
	if len(hOut) != rows*d {
		return nil, fmt.Errorf("mHC backward: hOut has %d values, want %d", len(hOut), rows*d)
	}
	if len(dRes) != rows*flat {
		return nil, fmt.Errorf("mHC backward: dRes has %d values, want %d", len(dRes), rows*flat)
	}

	// Recompute the forward exactly, including the normalised copy the parameter
	// generation reads and the raw state the residual reads.
	xHat := rmsNormNew(x, w.NormWeight, rows, flat, w.NormEps)
	aPre := float64(tanh32(w.AlphaPre))
	aRes := float64(tanh32(w.AlphaRes))
	aPost := float64(tanh32(w.AlphaPost))

	aGate := make([]float64, rows*n)
	cGate := make([]float64, rows*n)
	preZ := make([]float64, rows*n) // pre-sigmoid, for its derivative
	postZ := make([]float64, rows*n)
	inner := make([]float64, rows*n*n) // tanh(W_res . xHat), before the alpha gate
	bLogits := make([]float32, rows*n*n)
	for r := range rows {
		hat := xHat[r*flat : (r+1)*flat]
		for i := range n {
			preZ[r*n+i] = aPre*dot(w.WPre[i*flat:(i+1)*flat], hat) + float64(w.SPre[i])
			aGate[r*n+i] = sigmoid(preZ[r*n+i])
			postZ[r*n+i] = aPost*dot(w.WPost[i*flat:(i+1)*flat], hat) + float64(w.SPost[i])
			cGate[r*n+i] = 2.0 * sigmoid(postZ[r*n+i])
		}
		for i := range n * n {
			inner[r*n*n+i] = math.Tanh(dot(w.WRes[i*flat:(i+1)*flat], hat))
			bLogits[r*n*n+i] = float32(aRes*inner[r*n*n+i] + float64(w.SRes[i]))
		}
	}
	bDS := make([]float32, len(bLogits))
	copy(bDS, bLogits)
	hostmath.SinkhornFromLogitsInPlace(bDS, rows, n, w.SinkhornIters)

	g := &HyperConnectionGradients{
		DX:          make([]float32, rows*flat),
		DHOut:       make([]float32, rows*d),
		DWPre:       make([]float32, len(w.WPre)),
		DWRes:       make([]float32, len(w.WRes)),
		DWPost:      make([]float32, len(w.WPost)),
		DSPre:       make([]float32, len(w.SPre)),
		DSRes:       make([]float32, len(w.SRes)),
		DSPost:      make([]float32, len(w.SPost)),
		DNormWeight: make([]float32, len(w.NormWeight)),
	}
	dXHat := make([]float32, rows*flat)
	dBDS := make([]float64, rows*n*n)
	dAGate := make([]float64, rows*n)
	dCGate := make([]float64, rows*n)
	var dAlphaPreAcc, dAlphaResAcc, dAlphaPostAcc float64

	// res[r,i,:] = sum_j B[i,j] X[r,j,:] + cGate[r,i] hOut[r,:]
	for r := range rows {
		bm := bDS[r*n*n : (r+1)*n*n]
		for i := range n {
			dst := dRes[r*flat+i*d : r*flat+(i+1)*d]
			for j := range n {
				src := x[r*flat+j*d : r*flat+(j+1)*d]
				dsrc := g.DX[r*flat+j*d : r*flat+(j+1)*d]
				coeff := float64(bm[i*n+j])
				var acc float64
				for k, dv := range dst {
					acc += float64(dv) * float64(src[k])
					dsrc[k] += float32(coeff * float64(dv))
				}
				dBDS[r*n*n+i*n+j] += acc
			}
			row := hOut[r*d : (r+1)*d]
			c := cGate[r*n+i]
			var acc float64
			for k, dv := range dst {
				acc += float64(dv) * float64(row[k])
				g.DHOut[r*d+k] += float32(c * float64(dv))
			}
			dCGate[r*n+i] += acc
		}
	}

	// Now the sub-layer: dh_out is complete, so ask for dh_in and close the two
	// paths that run through it -- aGate, and the raw stream the collapse reads.
	//
	//	h_in[r,:] = sum_s aGate[r,s] * X[r,s,:]
	//	  d/d aGate[r,s] = dh_in . X[r,s,:]
	//	  d/d X[r,s,k]   = dh_in[k] * aGate[r,s]
	dHIn := subBackward(g.DHOut, rows, d)
	if len(dHIn) != rows*d {
		return nil, fmt.Errorf("mHC backward: sub-layer returned %d values for dHIn, want %d",
			len(dHIn), rows*d)
	}
	for r := range rows {
		din := dHIn[r*d : (r+1)*d]
		for s := range n {
			src := x[r*flat+s*d : r*flat+(s+1)*d]
			dsrc := g.DX[r*flat+s*d : r*flat+(s+1)*d]
			gate := aGate[r*n+s]
			var acc float64
			for k, dv := range din {
				acc += float64(dv) * float64(src[k])
				dsrc[k] += float32(gate * float64(dv))
			}
			dAGate[r*n+s] += acc
		}
	}

	// Sinkhorn: dB_ds -> dLogits, then through the alpha gate and the inner tanh.
	dBDS32 := make([]float32, len(dBDS))
	for i, v := range dBDS {
		dBDS32[i] = float32(v)
	}
	dLogits := hostmath.SinkhornFromLogitsBackward(bLogits, dBDS32, rows, n, w.SinkhornIters)

	for r := range rows {
		hat := xHat[r*flat : (r+1)*flat]
		dhat := dXHat[r*flat : (r+1)*flat]
		for i := range n * n {
			dl := float64(dLogits[r*n*n+i])
			if dl == 0 {
				continue
			}
			g.DSRes[i] += float32(dl)
			in := inner[r*n*n+i]
			dAlphaResAcc += dl * in
			dInner := dl * aRes * (1 - in*in)
			wrow := w.WRes[i*flat : (i+1)*flat]
			drow := g.DWRes[i*flat : (i+1)*flat]
			for j := range flat {
				drow[j] += float32(dInner * float64(hat[j]))
				dhat[j] += float32(dInner * float64(wrow[j]))
			}
		}
		for i := range n {
			// cGate = 2*sigmoid(z): the 2 is in the forward and must be here.
			s := sigmoid(postZ[r*n+i])
			dz := dCGate[r*n+i] * 2 * s * (1 - s)
			g.DSPost[i] += float32(dz)
			// postZ = aPost * (W_post . hat) + s_post, so d/d aPost is the dot.
			dAlphaPostAcc += dz * dot(w.WPost[i*flat:(i+1)*flat], hat)
			wrow := w.WPost[i*flat : (i+1)*flat]
			drow := g.DWPost[i*flat : (i+1)*flat]
			for j := range flat {
				drow[j] += float32(dz * aPost * float64(hat[j]))
				dhat[j] += float32(dz * aPost * float64(wrow[j]))
			}

			sa := sigmoid(preZ[r*n+i])
			dza := dAGate[r*n+i] * sa * (1 - sa)
			g.DSPre[i] += float32(dza)
			dAlphaPreAcc += dza * dot(w.WPre[i*flat:(i+1)*flat], hat)
			wrowA := w.WPre[i*flat : (i+1)*flat]
			drowA := g.DWPre[i*flat : (i+1)*flat]
			for j := range flat {
				drowA[j] += float32(dza * aPre * float64(hat[j]))
				dhat[j] += float32(dza * aPre * float64(wrowA[j]))
			}
		}
	}

	// The alpha scalars are tanh'd, so their gradient carries 1 - tanh^2.
	g.DAlphaPre = float32(dAlphaPreAcc * (1 - aPre*aPre))
	g.DAlphaRes = float32(dAlphaResAcc * (1 - aRes*aRes))
	g.DAlphaPost = float32(dAlphaPostAcc * (1 - aPost*aPost))

	// Through the norm, ACCUMULATING onto the raw-state paths already written.
	// The hostmath owner writes dx (addDX=true here) and dscale.
	hostmath.RMSNormBackward(g.DX, g.DNormWeight, x, w.NormWeight, dXHat, rows, flat, w.NormEps, true)
	return g, nil
}
