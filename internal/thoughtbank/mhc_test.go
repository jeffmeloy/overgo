package thoughtbank

import (
	"fmt"
	"overgo/internal/hostmath"

	"math"
	"math/rand"
	"testing"
)

// The sub-layer is the IDENTITY, so h_out = h_in and its backward is the
// identity too: it isolates the block's own five paths into dX, which is what
// this backward owns.
func identitySubLayer(hIn []float32, rows, d int) []float32 {
	out := make([]float32, len(hIn))
	copy(out, hIn)
	return out
}

func identitySubBackward(dHOut []float32, rows, d int) []float32 {
	out := make([]float32, len(dHOut))
	copy(out, dHOut)
	return out
}

func randHyperConnectionWeights(rng *rand.Rand, n, d int) *HyperConnectionWeights {
	flat := n * d
	rs := func(count int, scale float64) []float32 {
		v := make([]float32, count)
		for i := range v {
			v[i] = float32((rng.Float64()*2 - 1) * scale)
		}
		return v
	}
	nw := make([]float32, flat)
	for i := range nw {
		nw[i] = float32(0.8 + 0.4*rng.Float64())
	}
	return &HyperConnectionWeights{
		NHC: n, DModel: d, SinkhornIters: 20,
		WPre:  rs(n*flat, 1.0/math.Sqrt(float64(flat))),
		WRes:  rs(n*n*flat, 1.0/math.Sqrt(float64(flat))),
		WPost: rs(n*flat, 1.0/math.Sqrt(float64(flat))),
		SPre:  rs(n, 0.3), SRes: rs(n*n, 0.3), SPost: rs(n, 0.3),
		AlphaPre: 0.4, AlphaRes: 0.5, AlphaPost: 0.3,
		NormWeight: nw, NormEps: 1e-6,
	}
}

// The loss calls the SHIPPED forward, so there is no second spelling to keep in
// agreement.
func hyperConnectionLoss(x, dRes []float32, rows int, w *HyperConnectionWeights) (float64, error) {
	out, err := HyperConnectionForward(x, rows, w, identitySubLayer)
	if err != nil {
		return 0, err
	}
	s := 0.0
	for i := range out {
		s += float64(out[i]) * float64(dRes[i])
	}
	return s, nil
}

func TestHyperConnectionBackwardMatchesFiniteDifference(t *testing.T) {
	rng := rand.New(rand.NewSource(20260804))
	const rows, n, d = 3, 2, 4
	flat := n * d
	x := make([]float32, rows*flat)
	for i := range x {
		x[i] = float32((rng.Float64()*2 - 1) * 1.0)
	}
	w := randHyperConnectionWeights(rng, n, d)
	dRes := make([]float32, rows*flat)
	for i := range dRes {
		dRes[i] = float32(rng.Float64()*2 - 1)
	}

	var hOut []float32
	capture := func(hIn []float32, rows, d int) []float32 {
		out := identitySubLayer(hIn, rows, d)
		hOut = append([]float32(nil), out...)
		return out
	}
	if _, err := HyperConnectionForward(x, rows, w, capture); err != nil {
		t.Fatalf("forward: %v", err)
	}
	g, err := HyperConnectionBackward(x, hOut, dRes, rows, w, identitySubBackward)
	if err != nil {
		t.Fatalf("backward: %v", err)
	}

	type probe struct {
		name string
		buf  []float32
		idx  int
		want float32
	}
	probes := []probe{
		{"x", x, 0, g.DX[0]},
		{"x", x, 5, g.DX[5]},
		{"WPre", w.WPre, 3, g.DWPre[3]},
		{"WRes", w.WRes, 7, g.DWRes[7]},
		{"WPost", w.WPost, 2, g.DWPost[2]},
		{"SPre", w.SPre, 1, g.DSPre[1]},
		{"SRes", w.SRes, 2, g.DSRes[2]},
		{"SPost", w.SPost, 0, g.DSPost[0]},
		{"NormWeight", w.NormWeight, 4, g.DNormWeight[4]},
	}

	steps := []float64{1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5}
	devs := make([]float64, len(steps))
	for si, h := range steps {
		worst := 0.0
		for _, p := range probes {
			orig := p.buf[p.idx]
			p.buf[p.idx] = float32(float64(orig) + h)
			up, err := hyperConnectionLoss(x, dRes, rows, w)
			if err != nil {
				t.Fatalf("forward: %v", err)
			}
			p.buf[p.idx] = float32(float64(orig) - h)
			dn, err := hyperConnectionLoss(x, dRes, rows, w)
			if err != nil {
				t.Fatalf("forward: %v", err)
			}
			p.buf[p.idx] = orig

			num := (up - dn) / (2 * h)
			den := max(1e-2, math.Abs(float64(p.want)))
			if rel := math.Abs(num-float64(p.want)) / den; rel > worst {
				worst = rel
			}
		}
		devs[si] = worst
		t.Logf("h=%-8.0e max relative deviation %.3e", h, worst)
	}

	best, bestAt := devs[0], 0
	for i, v := range devs {
		if v < best {
			best, bestAt = v, i
		}
	}
	if bestAt == 0 || bestAt == len(devs)-1 {
		t.Fatalf("deviation minimum at an ENDPOINT (h=%.0e, dev=%.3e): the sweep never bracketed "+
			"the optimum", steps[bestAt], best)
	}
	if best > 1e-3 {
		t.Fatalf("best relative deviation %.3e at h=%.0e exceeds 1e-3", best, steps[bestAt])
	}
	t.Logf("optimum bracketed at h=%.0e with relative deviation %.3e", steps[bestAt], best)
}

// The alpha scalars gate three different sub-paths; a sign error in one would be
// invisible in the aggregate probe above.
func TestHyperConnectionBackwardAlphaGradients(t *testing.T) {
	rng := rand.New(rand.NewSource(555))
	const rows, n, d = 3, 2, 4
	flat := n * d
	x := make([]float32, rows*flat)
	for i := range x {
		x[i] = float32((rng.Float64()*2 - 1) * 1.0)
	}
	w := randHyperConnectionWeights(rng, n, d)
	dRes := make([]float32, rows*flat)
	for i := range dRes {
		dRes[i] = float32(rng.Float64()*2 - 1)
	}
	var hOut []float32
	capture := func(hIn []float32, rows, d int) []float32 {
		out := identitySubLayer(hIn, rows, d)
		hOut = append([]float32(nil), out...)
		return out
	}
	if _, err := HyperConnectionForward(x, rows, w, capture); err != nil {
		t.Fatalf("forward: %v", err)
	}
	g, err := HyperConnectionBackward(x, hOut, dRes, rows, w, identitySubBackward)
	if err != nil {
		t.Fatalf("backward: %v", err)
	}

	const h = 1e-2
	for _, c := range []struct {
		name string
		ptr  *float32
		want float32
	}{
		{"AlphaPre", &w.AlphaPre, g.DAlphaPre},
		{"AlphaRes", &w.AlphaRes, g.DAlphaRes},
		{"AlphaPost", &w.AlphaPost, g.DAlphaPost},
	} {
		orig := *c.ptr
		*c.ptr = float32(float64(orig) + h)
		up, _ := hyperConnectionLoss(x, dRes, rows, w)
		*c.ptr = float32(float64(orig) - h)
		dn, _ := hyperConnectionLoss(x, dRes, rows, w)
		*c.ptr = orig
		num := (up - dn) / (2 * h)
		den := max(1e-2, math.Abs(float64(c.want)))
		if rel := math.Abs(num-float64(c.want)) / den; rel > 1e-3 {
			t.Fatalf("%s: analytic %g vs numerical %g, relative %.3e", c.name, c.want, num, rel)
		}
	}
}

// TestHyperConnectionResidualMixingIsDoublyStochastic checks the property the
// manifold constraint buys, independently of any reference: with the sub-layer
// contributing nothing, the update is X <- B X with B doubly stochastic, so no
// stream's magnitude can be amplified -- every output coordinate lies within the
// range of that coordinate across streams. Self-contained (synthetic fixture).
func TestHyperConnectionResidualMixingIsDoublyStochastic(t *testing.T) {
	rng := rand.New(rand.NewSource(31))
	const rows, n, d = 4, 2, 5
	flat := n * d
	x := make([]float32, rows*flat)
	for i := range x {
		x[i] = float32((rng.Float64()*2 - 1) * 2.0)
	}
	w := randHyperConnectionWeights(rng, n, d)
	zeroLayer := func(_ []float32, r, dd int) []float32 { return make([]float32, r*dd) }
	got, err := HyperConnectionForward(x, rows, w, zeroLayer)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	for r := range rows {
		for k := range d {
			lo, hi := math.Inf(1), math.Inf(-1)
			for s := range n {
				v := float64(x[r*flat+s*d+k])
				lo, hi = min(lo, v), max(hi, v)
			}
			for s := range n {
				v := float64(got[r*flat+s*d+k])
				const slack = 1e-5
				if v < lo-slack || v > hi+slack {
					t.Fatalf("row %d stream %d coord %d: %.9f outside input range [%.9f, %.9f]; B is not doubly stochastic",
						r, s, k, v, lo, hi)
				}
			}
		}
	}
}

// A truncated parameter bundle must be rejected, not read past.
func TestHyperConnectionRejectsShapeMismatch(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	const n, d = 2, 4
	w := randHyperConnectionWeights(rng, n, d)
	w.WRes = w.WRes[:len(w.WRes)-1]
	x := make([]float32, 3*n*d)
	if _, err := HyperConnectionForward(x, 3, w, func(in []float32, r, d int) []float32 {
		return make([]float32, r*d)
	}); err == nil {
		t.Fatal("expected a shape error for a truncated W_res")
	}
}

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
