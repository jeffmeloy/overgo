package thoughtbank

import (
	"math"
	"math/rand"
	"testing"

	"overgo/internal/hostmath"
	"overgo/internal/testutil"
)

func newFastWeightBankFixture(rng *rand.Rand, swiGLU bool) (*FastWeightBankWeights, []float32, []float32, int, int) {
	const d, mem, r, rows, slots = 6, 5, 3, 4, 3
	rs := func(n int, scale float64) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32((rng.Float64()*2 - 1) * scale)
		}
		return v
	}
	na := 1
	if swiGLU {
		na = 2
	}
	s := 1.0 / math.Sqrt(float64(mem))
	w := &FastWeightBankWeights{
		DModel: d, MemDim: mem, Rank: r, SwiGLU: swiGLU, NormEps: 1e-6,
		FWA: rs(na*r*d*mem, s),
		FWB: rs(d*r*mem, s),
		FWO: rs(d*d, 1.0/math.Sqrt(float64(d))),
		NormWeight: func() []float32 {
			v := rs(d, 0.2)
			for i := range v {
				v[i] += 1
			}
			return v
		}(),
	}
	h := rs(rows*d, 1.0)
	bank := rs(slots*mem, 1.0)
	return w, h, bank, rows, slots
}

func bankBracket(t *testing.T, swiGLU bool) {
	rng := rand.New(rand.NewSource(20260804))
	w, h, bank, rows, slots := newFastWeightBankFixture(rng, swiGLU)
	d := w.DModel
	dOut := make([]float32, rows*d)
	for i := range dOut {
		dOut[i] = float32(rng.Float64()*2 - 1)
	}
	loss := func() float64 {
		out, err := FastWeightBankRead(h, bank, rows, slots, w)
		if err != nil {
			t.Fatalf("forward: %v", err)
		}
		s := 0.0
		for i := range out {
			s += float64(out[i]) * float64(dOut[i])
		}
		return s
	}
	g, err := FastWeightBankReadBackward(h, bank, dOut, rows, slots, w)
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
		{"h", h, 13, g.DH[13]},
		{"bank", bank, 2, g.DBank[2]},
		{"bank", bank, 9, g.DBank[9]},
		{"FWA", w.FWA, 7, g.DFWA[7]},
		{"FWB", w.FWB, 4, g.DFWB[4]},
		{"FWO", w.FWO, 10, g.DFWO[10]},
		{"NormWeight", w.NormWeight, 2, g.DNormWeight[2]},
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

// gelu path first, isolating the sequential-slot reverse scan from the clamp.
func TestFastWeightBankReadBackwardGELU(t *testing.T) { bankBracket(t, false) }

// SwiGLU path exercises the clamped silu(zg)*zv gate and its branch.
func TestFastWeightBankReadBackwardSwiGLU(t *testing.T) { bankBracket(t, true) }

// TestFastWeightBankReadBackwardClampActive forces the SwiGLU gate into the clamp
// region for a real fraction of entries, so the derivative-zero branch is
// exercised rather than skipped. If the clamp derivative were not zeroed, the
// analytic gradient would carry a contribution the loss (flat on the plateau)
// does not, and the bracket would break.
func TestFastWeightBankReadBackwardClampActive(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	w, h, bank, rows, slots := newFastWeightBankFixture(rng, true)
	for i := range w.FWA {
		w.FWA[i] *= 20
	}

	// Confirm clamping actually occurs, else the test is a weaker duplicate.
	clampSeen := false
	d, r := w.DModel, w.Rank
	ds := 1.0 / math.Sqrt(float64(d))
	y0 := rmsNormNew(h, w.NormWeight, rows, d, w.NormEps)
	for s := 0; s < slots && !clampSeen; s++ {
		slot := bank[s*w.MemDim : (s+1)*w.MemDim]
		a := make([]float64, 2*r*d)
		for i := range a {
			a[i] = dot(w.FWA[i*w.MemDim:(i+1)*w.MemDim], slot)
		}
		for tt := 0; tt < rows && !clampSeen; tt++ {
			row := y0[tt*d : (tt+1)*d]
			for k := 0; k < r; k++ {
				var gg, v float64
				for j := 0; j < d; j++ {
					gg += a[k*d+j] * float64(row[j])
					v += a[r*d+k*d+j] * float64(row[j])
				}
				gg *= ds
				v *= ds
				if math.Abs(gg/(1+math.Exp(-gg))*v) > FastWeightClamp {
					clampSeen = true
					break
				}
			}
		}
	}
	if !clampSeen {
		t.Fatal("fixture did not induce clamping at slot 0; the clamp branch would go untested")
	}

	dOut := make([]float32, rows*d)
	for i := range dOut {
		dOut[i] = float32(rng.Float64()*2 - 1)
	}
	loss := func() float64 {
		out, err := FastWeightBankRead(h, bank, rows, slots, w)
		if err != nil {
			t.Fatalf("forward: %v", err)
		}
		s := 0.0
		for i := range out {
			s += float64(out[i]) * float64(dOut[i])
		}
		return s
	}
	g, err := FastWeightBankReadBackward(h, bank, dOut, rows, slots, w)
	if err != nil {
		t.Fatalf("backward: %v", err)
	}
	type probe struct {
		buf  []float32
		idx  int
		want float32
	}
	probes := []probe{{h, 5, g.DH[5]}, {w.FWA, 7, g.DFWA[7]}, {w.FWB, 4, g.DFWB[4]}, {bank, 2, g.DBank[2]}}
	steps := []float64{1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5}
	devs := make([]float64, len(steps))
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
				worst = rel
			}
		}
		devs[si] = worst
		t.Logf("h=%-8.0e max relative deviation %.3e", hstep, worst)
	}
	best, bestAt := devs[0], 0
	for i, v := range devs {
		if v < best {
			best, bestAt = v, i
		}
	}
	if bestAt == 0 || bestAt == len(devs)-1 {
		t.Fatalf("deviation minimum at an ENDPOINT (h=%.0e, dev=%.3e) with clamping active", steps[bestAt], best)
	}
	if best > 2e-3 {
		t.Fatalf("best relative deviation %.3e exceeds 2e-3 with clamping active: the clamp "+
			"derivative may not be zeroed", best)
	}
	t.Logf("clamp-active optimum bracketed at h=%.0e with relative deviation %.3e", steps[bestAt], best)
}

// TestFastWeightEmptyBankIsIdentity is the property the delta form buys: with no
// slots there is nothing to apply, so the read must return its input unchanged
// rather than the normalised input. Self-contained (synthetic fixture).
func TestFastWeightEmptyBankIsIdentity(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	w, h, _, rows, _ := newFastWeightBankFixture(rng, true)
	got, err := FastWeightBankRead(h, nil, rows, 0, w)
	if err != nil {
		t.Fatalf("FastWeightBankRead: %v", err)
	}
	for i := range got {
		if math.Abs(float64(got[i]-h[i])) > 1e-6 {
			t.Fatalf("coord %d changed with an empty bank: %v -> %v", i, h[i], got[i])
		}
	}
}

// TestFastWeightSlotsComposeInOrder: the slots form a STACK, not a sum. Feeding
// the same two slots in the opposite order must change the answer, or the
// sequential composition -- the reason the read has depth at all -- is not
// happening.
func TestFastWeightSlotsComposeInOrder(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	w, h, bank, rows, slots := newFastWeightBankFixture(rng, true)
	if slots < 2 {
		t.Skip("need at least two slots")
	}
	fwd, err := FastWeightBankRead(h, bank, rows, slots, w)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	rev := make([]float32, len(bank))
	for s := 0; s < slots; s++ {
		copy(rev[s*w.MemDim:(s+1)*w.MemDim], bank[(slots-1-s)*w.MemDim:(slots-s)*w.MemDim])
	}
	back, err := FastWeightBankRead(h, rev, rows, slots, w)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	if testutil.MaxAbsDiff(fwd, back) < 1e-6 {
		t.Fatal("reversing the bank left the read unchanged; slots are summing, not composing")
	}
}

// TestFastWeightClampIsDistinctFromSwiGLUClamp guards against a later
// "unification" of the two constants. They are 8 (this bank) and 10 (the MoE
// expert SwiGLU, owned by hostmath) in the reference and belong to different
// operators; collapsing them would change whichever site moved.
func TestFastWeightClampIsDistinctFromSwiGLUClamp(t *testing.T) {
	if FastWeightClamp == hostmath.SwiGLUClampLinear || FastWeightClamp == hostmath.SwiGLUClampGate {
		t.Fatalf("FastWeightClamp (%v) has been unified with the SwiGLU clamps (%v/%v); the reference keeps them separate",
			FastWeightClamp, float64(hostmath.SwiGLUClampLinear), float64(hostmath.SwiGLUClampGate))
	}
}
