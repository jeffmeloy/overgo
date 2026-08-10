package thoughtbank

import (
	"math"
	"math/rand"
	"testing"
)

func compressHeavyLoss(hPad, wKV, wZ, pos, dOut []float32, blocks, m, dModel, dHead int) (float64, error) {
	out, err := CompressKVHeavy(hPad, wKV, wZ, pos, blocks, m, dModel, dHead)
	if err != nil {
		return 0, err
	}
	s := 0.0
	for i := range out {
		s += float64(out[i]) * float64(dOut[i])
	}
	return s, nil
}

func TestCompressKVHeavyBackwardMatchesFiniteDifference(t *testing.T) {
	rng := rand.New(rand.NewSource(20260804))
	const blocks, m, dModel, dHead = 3, 4, 6, 5
	hPad := randSlice(rng, blocks*m*dModel, 1.0)
	s := 1.0 / math.Sqrt(float64(dModel))
	wKV := randSlice(rng, dHead*dModel, s)
	wZ := randSlice(rng, dHead*dModel, s)
	pos := randSlice(rng, m*dHead, 0.5)
	dOut := randSlice(rng, blocks*dHead, 1.0)

	g, err := CompressKVHeavyBackward(hPad, wKV, wZ, pos, dOut, blocks, m, dModel, dHead)
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
		{"hPad", hPad, 0, g.DHPad[0]},
		{"hPad", hPad, 31, g.DHPad[31]},
		{"wKV", wKV, 4, g.DWKV[4]},
		{"wKV", wKV, 17, g.DWKV[17]},
		{"wZ", wZ, 9, g.DWZ[9]},
		{"pos", pos, 3, g.DPos[3]},
		{"pos", pos, 11, g.DPos[11]},
	}

	steps := []float64{1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5}
	devs := make([]float64, len(steps))
	for si, h := range steps {
		worst := 0.0
		for _, p := range probes {
			orig := p.buf[p.idx]
			p.buf[p.idx] = float32(float64(orig) + h)
			up, err := compressHeavyLoss(hPad, wKV, wZ, pos, dOut, blocks, m, dModel, dHead)
			if err != nil {
				t.Fatalf("forward: %v", err)
			}
			p.buf[p.idx] = float32(float64(orig) - h)
			dn, err := compressHeavyLoss(hPad, wKV, wZ, pos, dOut, blocks, m, dModel, dHead)
			if err != nil {
				t.Fatalf("forward: %v", err)
			}
			p.buf[p.idx] = orig

			num := (up - dn) / (2 * h)
			den := math.Max(1e-2, math.Abs(float64(p.want)))
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
		t.Fatalf("deviation minimum at an ENDPOINT (h=%.0e, dev=%.3e)", steps[bestAt], best)
	}
	if best > 1e-4 {
		t.Fatalf("best relative deviation %.3e at h=%.0e exceeds 1e-4", best, steps[bestAt])
	}
	t.Logf("optimum bracketed at h=%.0e with relative deviation %.3e", steps[bestAt], best)
}

// The pooling weights are a softmax over the block axis, so shifting every z in
// a column by a constant must not change the output -- dPos summed over j is ~0
// per feature. (Unlike the sink softmax, this invariance holds exactly here.)
func TestCompressKVHeavyBackwardPosGradientIsShiftInvariant(t *testing.T) {
	rng := rand.New(rand.NewSource(4242))
	const blocks, m, dModel, dHead = 2, 4, 5, 3
	hPad := randSlice(rng, blocks*m*dModel, 1.0)
	s := 1.0 / math.Sqrt(float64(dModel))
	wKV := randSlice(rng, dHead*dModel, s)
	wZ := randSlice(rng, dHead*dModel, s)
	pos := randSlice(rng, m*dHead, 0.5)
	dOut := randSlice(rng, blocks*dHead, 1.0)

	g, err := CompressKVHeavyBackward(hPad, wKV, wZ, pos, dOut, blocks, m, dModel, dHead)
	if err != nil {
		t.Fatalf("backward: %v", err)
	}
	for e := 0; e < dHead; e++ {
		total := 0.0
		for j := 0; j < m; j++ {
			total += float64(g.DPos[j*dHead+e])
		}
		if math.Abs(total) > 1e-4 {
			t.Fatalf("dPos column %d sums to %g, want ~0 (softmax shift invariance)", e, total)
		}
	}
}

func compressSparseLoss(hPad, wKVa, wKVb, wZa, wZb, posA, posB, dOut []float32,
	blocks, m, dModel, dHead int) (float64, error) {
	out, err := CompressKVSparse(hPad, wKVa, wKVb, wZa, wZb, posA, posB, blocks, m, dModel, dHead)
	if err != nil {
		return 0, err
	}
	s := 0.0
	for i := range out {
		s += float64(out[i]) * float64(dOut[i])
	}
	return s, nil
}

func TestCompressKVSparseBackwardMatchesFiniteDifference(t *testing.T) {
	rng := rand.New(rand.NewSource(20260804))
	const blocks, m, dModel, dHead = 3, 4, 6, 5
	hPad := randSlice(rng, blocks*m*dModel, 1.0)
	s := 1.0 / math.Sqrt(float64(dModel))
	wKVa := randSlice(rng, dHead*dModel, s)
	wKVb := randSlice(rng, dHead*dModel, s)
	wZa := randSlice(rng, dHead*dModel, s)
	wZb := randSlice(rng, dHead*dModel, s)
	posA := randSlice(rng, m*dHead, 0.5)
	posB := randSlice(rng, m*dHead, 0.5)
	dOut := randSlice(rng, blocks*dHead, 1.0)

	g, err := CompressKVSparseBackward(hPad, wKVa, wKVb, wZa, wZb, posA, posB, dOut, blocks, m, dModel, dHead)
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
		{"hPad", hPad, 0, g.DHPad[0]},
		{"hPad", hPad, 31, g.DHPad[31]},
		{"wKVa", wKVa, 4, g.DWKVa[4]},
		{"wKVb", wKVb, 17, g.DWKVb[17]},
		{"wZa", wZa, 9, g.DWZa[9]},
		{"wZb", wZb, 2, g.DWZb[2]},
		{"posA", posA, 3, g.DPosA[3]},
		{"posB", posB, 11, g.DPosB[11]},
	}

	steps := []float64{1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5}
	devs := make([]float64, len(steps))
	for si, h := range steps {
		worst := 0.0
		for _, p := range probes {
			orig := p.buf[p.idx]
			p.buf[p.idx] = float32(float64(orig) + h)
			up, err := compressSparseLoss(hPad, wKVa, wKVb, wZa, wZb, posA, posB, dOut, blocks, m, dModel, dHead)
			if err != nil {
				t.Fatalf("forward: %v", err)
			}
			p.buf[p.idx] = float32(float64(orig) - h)
			dn, err := compressSparseLoss(hPad, wKVa, wKVb, wZa, wZb, posA, posB, dOut, blocks, m, dModel, dHead)
			if err != nil {
				t.Fatalf("forward: %v", err)
			}
			p.buf[p.idx] = orig

			num := (up - dn) / (2 * h)
			den := math.Max(1e-2, math.Abs(float64(p.want)))
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
		t.Fatalf("deviation minimum at an ENDPOINT (h=%.0e, dev=%.3e)", steps[bestAt], best)
	}
	if best > 1e-4 {
		t.Fatalf("best relative deviation %.3e at h=%.0e exceeds 1e-4", best, steps[bestAt])
	}
	t.Logf("optimum bracketed at h=%.0e with relative deviation %.3e", steps[bestAt], best)
}

// The block-0 phantom path: no predecessor, so the b-series must receive ZERO
// gradient and nothing may be NaN despite the -inf gate.
func TestCompressKVSparseBackwardPhantomBlockZero(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	const blocks, m, dModel, dHead = 1, 4, 5, 3
	hPad := randSlice(rng, blocks*m*dModel, 1.0)
	s := 1.0 / math.Sqrt(float64(dModel))
	wKVa := randSlice(rng, dHead*dModel, s)
	wKVb := randSlice(rng, dHead*dModel, s)
	wZa := randSlice(rng, dHead*dModel, s)
	wZb := randSlice(rng, dHead*dModel, s)
	posA := randSlice(rng, m*dHead, 0.5)
	posB := randSlice(rng, m*dHead, 0.5)
	dOut := randSlice(rng, blocks*dHead, 1.0)

	g, err := CompressKVSparseBackward(hPad, wKVa, wKVb, wZa, wZb, posA, posB, dOut, blocks, m, dModel, dHead)
	if err != nil {
		t.Fatalf("backward: %v", err)
	}

	for i, v := range g.DWKVb {
		if v != 0 {
			t.Fatalf("DWKVb[%d]=%g, want 0: the phantom predecessor carries no gradient", i, v)
		}
	}
	for i, v := range g.DWZb {
		if v != 0 {
			t.Fatalf("DWZb[%d]=%g, want 0", i, v)
		}
	}
	for i, v := range g.DPosB {
		if v != 0 {
			t.Fatalf("DPosB[%d]=%g, want 0", i, v)
		}
	}
	for _, pair := range []struct {
		name string
		buf  []float32
	}{{"DHPad", g.DHPad}, {"DWKVa", g.DWKVa}, {"DWZa", g.DWZa}, {"DPosA", g.DPosA}} {
		for i, v := range pair.buf {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("%s[%d]=%g is not finite: the phantom -inf gate leaked into a real gradient", pair.name, i, v)
			}
		}
	}
}
