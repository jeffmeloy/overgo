package thoughtbank

import (
	"math"
	"math/rand"
	"testing"
)

// newHCAFixture builds a valid HCA (Sparse=false) attention layer. HCA attends
// every causally-valid compressed block with no indexer, so the loss is smooth
// in every parameter and a finite-difference bracket is valid end-to-end.
func newHCAFixture(rng *rand.Rand) (*CompressedHybridAttnWeights, []float32, int) {
	const seq, d, dh, nh, m, dLatentQ, nWin, nGroups = 9, 8, 4, 2, 3, 6, 2, 2
	rs := func(n int, scale float64) []float32 { return randSlice(rng, n, scale) }
	norm := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(1 + 0.2*(rng.Float64()*2-1))
		}
		return v
	}
	s := 1.0 / math.Sqrt(float64(d))
	w := &CompressedHybridAttnWeights{
		DModel: d, DHead: dh, NHeads: nh, M: m, DLatentQ: dLatentQ,
		NWin: nWin, NGroups: nGroups, Sparse: false, NormEps: 1e-6,
		WKV: rs(dh*d, s), WZ: rs(dh*d, s), Pos: rs(m*dh, s),
		WDq: rs(dLatentQ*d, s), WUq: rs(nh*dh*dLatentQ, s),
		WWk: rs(dh*d, s), WWv: rs(dh*d, s),
		QNorm: norm(dh), KVNorm: norm(dh),
		SinkLogits: rs(nh, 0.3),
		OutProj:    rs(d*d, s),
		OutGroup:   [][]float32{rs((d/nGroups)*(nh/nGroups)*dh, s), rs((d/nGroups)*(nh/nGroups)*dh, s)},
	}
	return w, rs(seq*d, 1.0), seq
}

func TestCompressedHybridAttentionBackwardHCAMatchesFiniteDifference(t *testing.T) {
	rng := rand.New(rand.NewSource(20260804))
	w, h, seq := newHCAFixture(rng)
	d := w.DModel
	dOut := make([]float32, seq*d)
	for i := range dOut {
		dOut[i] = float32(rng.Float64()*2 - 1)
	}

	loss := func() float64 {
		out, err := CompressedHybridAttentionForward(h, seq, w)
		if err != nil {
			t.Fatalf("forward: %v", err)
		}
		s := 0.0
		for i := range out {
			s += float64(out[i]) * float64(dOut[i])
		}
		return s
	}

	g, err := CompressedHybridAttentionBackward(h, dOut, seq, w)
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
		{"h", h, 5, g.DH[5]},
		{"h", h, 20, g.DH[20]},
		{"WKV", w.WKV, 7, g.DWKV[7]},
		{"WZ", w.WZ, 3, g.DWZ[3]},
		{"Pos", w.Pos, 2, g.DPos[2]},
		{"WDq", w.WDq, 10, g.DWDq[10]},
		{"WUq", w.WUq, 8, g.DWUq[8]},
		{"WWk", w.WWk, 4, g.DWWk[4]},
		{"WWv", w.WWv, 6, g.DWWv[6]},
		{"OutProj", w.OutProj, 12, g.DOutProj[12]},
		{"OutGroup0", w.OutGroup[0], 5, g.DOutGroup[0][5]},
		{"QNorm", w.QNorm, 1, g.DQNorm[1]},
		{"KVNorm", w.KVNorm, 2, g.DKVNorm[2]},
		{"SinkLogits", w.SinkLogits, 0, g.DSinkLogits[0]},
	}

	steps := []float64{1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5}
	devs := make([]float64, len(steps))
	var worstName string
	for si, hstep := range steps {
		worst := 0.0
		for _, p := range probes {
			orig := p.buf[p.idx]
			p.buf[p.idx] = float32(float64(orig) + hstep)
			up := loss()
			p.buf[p.idx] = float32(float64(orig) - hstep)
			dn := loss()
			p.buf[p.idx] = orig

			num := (up - dn) / (2 * hstep)
			den := math.Max(1e-2, math.Abs(float64(p.want)))
			if rel := math.Abs(num-float64(p.want)) / den; rel > worst {
				worst, worstName = rel, p.name
			}
		}
		devs[si] = worst
		t.Logf("h=%-8.0e max relative deviation %.3e (%s)", hstep, worst, worstName)
	}

	best, bestAt := devs[0], 0
	for i, v := range devs {
		if v < best {
			best, bestAt = v, i
		}
	}
	if bestAt == 0 || bestAt == len(devs)-1 {
		t.Fatalf("deviation minimum at an ENDPOINT (h=%.0e, dev=%.3e)", steps[bestAt], best)
	}
	if best > 1e-3 {
		t.Fatalf("best relative deviation %.3e at h=%.0e exceeds 1e-3", best, steps[bestAt])
	}
	t.Logf("optimum bracketed at h=%.0e with relative deviation %.3e", steps[bestAt], best)
}

// newCSABracketFixture builds a CSA (Sparse=true) layer. The finite-difference
// bracket is valid where the top-k selection does not flip; at small steps the
// finite selection margin holds it fixed and the interior minimum lands there.
func newCSABracketFixture(rng *rand.Rand) (*CompressedHybridAttnWeights, []float32, int) {
	const seq, d, dh, nh, m, dLatentQ, nIdx, topK, nWin, nGroups = 12, 8, 4, 2, 4, 6, 2, 2, 3, 2
	rs := func(n int, scale float64) []float32 { return randSlice(rng, n, scale) }
	norm := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(1 + 0.2*(rng.Float64()*2-1))
		}
		return v
	}
	s := 1.0 / math.Sqrt(float64(d))
	w := &CompressedHybridAttnWeights{
		DModel: d, DHead: dh, NHeads: nh, M: m, DLatentQ: dLatentQ,
		NIdxHeads: nIdx, TopK: topK, NWin: nWin, NGroups: nGroups, Sparse: true, NormEps: 1e-6,
		WKVa: rs(dh*d, s), WKVb: rs(dh*d, s), WZa: rs(dh*d, s), WZb: rs(dh*d, s),
		PosA: rs(m*dh, s), PosB: rs(m*dh, s),
		WDq: rs(dLatentQ*d, s), WUq: rs(nh*dh*dLatentQ, s),
		WWk: rs(dh*d, s), WWv: rs(dh*d, s),
		WW: rs(nIdx*d, s), WIq: rs(nIdx*dh*dLatentQ, s),
		QNorm: norm(dh), KVNorm: norm(dh),
		SinkLogits: rs(nh, 0.3),
		OutProj:    rs(d*d, s),
		OutGroup:   [][]float32{rs((d/nGroups)*(nh/nGroups)*dh, s), rs((d/nGroups)*(nh/nGroups)*dh, s)},
	}
	return w, rs(seq*d, 1.0), seq
}

func TestCompressedHybridAttentionBackwardCSAMatchesFiniteDifference(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	w, h, seq := newCSABracketFixture(rng)
	d := w.DModel
	dOut := make([]float32, seq*d)
	for i := range dOut {
		dOut[i] = float32(rng.Float64()*2 - 1)
	}

	loss := func() float64 {
		out, err := CompressedHybridAttentionForward(h, seq, w)
		if err != nil {
			t.Fatalf("forward: %v", err)
		}
		s := 0.0
		for i := range out {
			s += float64(out[i]) * float64(dOut[i])
		}
		return s
	}

	g, err := CompressedHybridAttentionBackward(h, dOut, seq, w)
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
		{"h", h, 5, g.DH[5]},
		{"WKVa", w.WKVa, 7, g.DWKVa[7]},
		{"WKVb", w.WKVb, 3, g.DWKVb[3]},
		{"WZa", w.WZa, 9, g.DWZa[9]},
		{"WZb", w.WZb, 2, g.DWZb[2]},
		{"PosA", w.PosA, 5, g.DPosA[5]},
		{"PosB", w.PosB, 1, g.DPosB[1]},
		{"WUq", w.WUq, 8, g.DWUq[8]},
		{"WWk", w.WWk, 4, g.DWWk[4]},
		{"WWv", w.WWv, 6, g.DWWv[6]},
		{"OutProj", w.OutProj, 12, g.DOutProj[12]},
		{"OutGroup0", w.OutGroup[0], 5, g.DOutGroup[0][5]},
		{"QNorm", w.QNorm, 1, g.DQNorm[1]},
		{"KVNorm", w.KVNorm, 2, g.DKVNorm[2]},
		{"SinkLogits", w.SinkLogits, 0, g.DSinkLogits[0]},
	}

	steps := []float64{3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5}
	devs := make([]float64, len(steps))
	var worstName string
	for si, hstep := range steps {
		worst := 0.0
		for _, p := range probes {
			orig := p.buf[p.idx]
			p.buf[p.idx] = float32(float64(orig) + hstep)
			up := loss()
			p.buf[p.idx] = float32(float64(orig) - hstep)
			dn := loss()
			p.buf[p.idx] = orig

			num := (up - dn) / (2 * hstep)
			den := math.Max(1e-2, math.Abs(float64(p.want)))
			if rel := math.Abs(num-float64(p.want)) / den; rel > worst {
				worst, worstName = rel, p.name
			}
		}
		devs[si] = worst
		t.Logf("h=%-8.0e max relative deviation %.3e (%s)", hstep, worst, worstName)
	}

	best, bestAt := devs[0], 0
	for i, v := range devs {
		if v < best {
			best, bestAt = v, i
		}
	}
	if bestAt == 0 || bestAt == len(devs)-1 {
		t.Fatalf("deviation minimum at an ENDPOINT (h=%.0e, dev=%.3e)", steps[bestAt], best)
	}
	if best > 2e-3 {
		t.Fatalf("best relative deviation %.3e at h=%.0e exceeds 2e-3 (a selection flip or wrong scatter)", best, steps[bestAt])
	}
	t.Logf("optimum bracketed at h=%.0e with relative deviation %.3e", steps[bestAt], best)
}

// The lightning indexer weights receive no gradient (detached): the gradients
// struct exposes none, AND perturbing them within a selection-stable
// neighbourhood leaves the loss exactly unchanged.
func TestCompressedHybridAttentionBackwardCSAIndexerDetached(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	w, h, seq := newCSABracketFixture(rng)
	dOut := make([]float32, seq*w.DModel)
	for i := range dOut {
		dOut[i] = float32(rng.Float64()*2 - 1)
	}
	loss := func() float64 {
		out, _ := CompressedHybridAttentionForward(h, seq, w)
		s := 0.0
		for i := range out {
			s += float64(out[i]) * float64(dOut[i])
		}
		return s
	}
	base := loss()
	for _, tc := range []struct {
		name string
		buf  []float32
		idx  int
	}{{"WW", w.WW, 3}, {"WIq", w.WIq, 5}} {
		orig := tc.buf[tc.idx]
		tc.buf[tc.idx] = float32(float64(orig) + 1e-4)
		up := loss()
		tc.buf[tc.idx] = orig
		if math.Abs(up-base) > 1e-9 {
			t.Fatalf("%s perturbation changed the loss by %.3e without flipping selection; the indexer must be flat (detached)", tc.name, up-base)
		}
	}
}
