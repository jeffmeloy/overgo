package thoughtbank

import (
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
