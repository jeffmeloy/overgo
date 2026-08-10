package thoughtbank

import (
	"math"
	"math/rand"
	"testing"
)

// The loss calls the shipped forward, so there is no second spelling to keep in
// agreement.
func sinkSoftmaxLoss(logits, sink, dOut []float32, rows, heads, n int) float64 {
	out := make([]float32, rows*heads*n)
	AttentionSinkSoftmaxInto(out, logits, sink, rows, heads, n)
	s := 0.0
	for i := range out {
		s += float64(out[i]) * float64(dOut[i])
	}
	return s
}

func TestAttentionSinkSoftmaxBackwardMatchesFiniteDifference(t *testing.T) {
	rng := rand.New(rand.NewSource(20260804))
	const rows, heads, n = 3, 2, 5
	logits := randSlice(rng, rows*heads*n, 2.0)
	sink := randSlice(rng, heads, 1.0)
	dOut := randSlice(rng, rows*heads*n, 1.0)

	out := make([]float32, rows*heads*n)
	AttentionSinkSoftmaxInto(out, logits, sink, rows, heads, n)
	dLogits, dSink := AttentionSinkSoftmaxBackward(dOut, out, logits, sink, rows, heads, n)

	type probe struct {
		name string
		buf  []float32
		idx  int
		want float32
	}
	probes := []probe{
		{"logit", logits, 0, dLogits[0]},
		{"logit", logits, 7, dLogits[7]},
		{"logit", logits, 22, dLogits[22]},
		{"sink", sink, 0, dSink[0]},
		{"sink", sink, 1, dSink[1]},
	}

	steps := []float64{1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5}
	devs := make([]float64, len(steps))
	for si, h := range steps {
		worst := 0.0
		for _, p := range probes {
			orig := p.buf[p.idx]
			p.buf[p.idx] = float32(float64(orig) + h)
			up := sinkSoftmaxLoss(logits, sink, dOut, rows, heads, n)
			p.buf[p.idx] = float32(float64(orig) - h)
			dn := sinkSoftmaxLoss(logits, sink, dOut, rows, heads, n)
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

// The correction lands on the forward's argmax, and only the LOGITS define it.
// Given an (out, logits) pair that disagree about the argmax, which does the
// function use? The property the fix establishes, checkable exactly.
func TestAttentionSinkSoftmaxBackwardTakesArgmaxFromLogits(t *testing.T) {
	const rows, heads, n = 1, 1, 3
	logits := []float32{1.0, 2.0, -5.0} // argmax at index 1
	sink := []float32{2.0}
	dOut := []float32{1.0, -1.0, 0.5}

	out := []float32{0.50, 0.30, 0.05} // argmax of p is 0
	dLogits, _ := AttentionSinkSoftmaxBackward(dOut, out, logits, sink, rows, heads, n)

	c := 0.0
	total := 0.0
	for i := range out {
		c += float64(dOut[i]) * float64(out[i])
		total += float64(out[i])
	}
	s := 1 - total
	corr := -s * c
	want0 := float64(out[0]) * (float64(dOut[0]) - c)
	want1 := float64(out[1])*(float64(dOut[1])-c) + corr
	if math.Abs(float64(dLogits[0])-want0) > 1e-6 || math.Abs(float64(dLogits[1])-want1) > 1e-6 {
		t.Fatalf("correction is on the wrong entry: got [%g %g], want [%g %g] with the logit-argmax at index 1",
			dLogits[0], dLogits[1], want0, want1)
	}
}

// The max-shift correction exists ONLY because the sink is un-shifted. With a
// strong sink the term is large; dropping it would be wrong in proportion to the
// sink's share.
func TestAttentionSinkSoftmaxBackwardNeedsTheMaxShiftTerm(t *testing.T) {
	const rows, heads, n = 1, 1, 4
	logits := []float32{2.0, 0.5, -1.0, 0.0}
	sink := []float32{3.0}
	dOut := []float32{1.0, -0.5, 0.25, 0.75}

	out := make([]float32, rows*heads*n)
	AttentionSinkSoftmaxInto(out, logits, sink, rows, heads, n)
	dLogits, _ := AttentionSinkSoftmaxBackward(dOut, out, logits, sink, rows, heads, n)

	const h = 1e-3
	orig := logits[0]
	logits[0] = float32(float64(orig) + h)
	up := sinkSoftmaxLoss(logits, sink, dOut, rows, heads, n)
	logits[0] = float32(float64(orig) - h)
	dn := sinkSoftmaxLoss(logits, sink, dOut, rows, heads, n)
	logits[0] = orig
	num := (up - dn) / (2 * h)

	p := make([]float64, n)
	total := 0.0
	for i := range out {
		p[i] = float64(out[i])
		total += p[i]
	}
	s := 1 - total
	c := 0.0
	for i := range p {
		c += float64(dOut[i]) * p[i]
	}
	uncorrected := p[0] * (float64(dOut[0]) - c)

	if math.Abs(num-float64(dLogits[0])) > 1e-3*math.Max(1, math.Abs(num)) {
		t.Fatalf("corrected gradient %g does not match numerical %g", dLogits[0], num)
	}
	if math.Abs(num-uncorrected) < 1e-3*math.Max(1, math.Abs(num)) {
		t.Fatalf("the uncorrected form %g also matches numerical %g (s=%.3f): fixture does not exercise the term",
			uncorrected, num, s)
	}
	t.Logf("sink share s=%.3f: corrected %g matches numerical %g; uncorrected %g does not", s, dLogits[0], num, uncorrected)
}
