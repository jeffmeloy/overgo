package oscillatorimage

import (
	"encoding/json"
	"testing"
)

// TestConv2dBackwardMatchesTorch: conv VJP (dx, dW, dB) vs torch autograd
// (protocol: adaptive TestUn0Conv2dBackwardMatchesTorch, gate 1e-4).
func TestConv2dBackwardMatchesTorch(t *testing.T) {
	var m map[string]json.RawMessage
	loadReferenceGolden(t, "un0_conv_bwd_golden.json", &m)
	gi := func(k string) int { var v int; json.Unmarshal(m[k], &v); return v }
	gf := func(k string) []float64 { var v []float64; json.Unmarshal(m[k], &v); return v }
	b, cin, cout, h, w := gi("b"), gi("cin"), gi("cout"), gi("h"), gi("w")
	dx, dW, dB := conv2dSame3x3Backward(f32of(gf("x")), f32of(gf("weight")), f32of(gf("dout")), b, cin, cout, h, w)
	requireWithin(t, "dx", dx, gf("dx"), tolOperator)
	requireWithin(t, "dW", dW, gf("dW"), tolOperator)
	requireWithin(t, "dB", dB, gf("dB"), tolOperator)
}

// TestResizeConvBlockBackwardMatchesTorch: decoder-block VJP vs torch
// autograd (protocol: adaptive TestUn0ResizeConvBlockBackwardMatchesTorch).
func TestResizeConvBlockBackwardMatchesTorch(t *testing.T) {
	var m map[string]json.RawMessage
	loadReferenceGolden(t, "un0_block_bwd_golden.json", &m)
	gi := func(k string) int { var v int; json.Unmarshal(m[k], &v); return v }
	gf := func(k string) []float64 { var v []float64; json.Unmarshal(m[k], &v); return v }
	b, cin, cout, h, w := gi("b"), gi("cin"), gi("cout"), gi("h"), gi("w")
	dx, dW1, dB1, dW2, dB2 := resizeConvBlockBackward(
		f32of(gf("x")), f32of(gf("w1")), f32of(gf("b1")), f32of(gf("w2")), f32of(gf("b2")),
		f32of(gf("dout")), b, cin, cout, h, w, decoderNegativeSlope)
	requireWithin(t, "dx", dx, gf("dx"), tolOperator)
	requireWithin(t, "dW1", dW1, gf("dW1"), tolOperator)
	requireWithin(t, "dB1", dB1, gf("dB1"), tolOperator)
	requireWithin(t, "dW2", dW2, gf("dW2"), tolOperator)
	requireWithin(t, "dB2", dB2, gf("dB2"), tolOperator)
}

// TestDynamicsBackwardFiniteDiff: coupled state/parameter VJPs vs central
// differences (the reference gates these VJPs by finite difference, not
// goldens).
func TestDynamicsBackwardFiniteDiff(t *testing.T) {
	b, n, nc := 2, 5, 2
	tot := n + nc
	state := randF32(b*tot, 31)
	omega := randF32(n, 32)
	omegaC := randF32(nc, 33)
	K := randF32(n*n, 34)
	Kc := randF32(nc*nc, 35)
	drive := randF32(b*n*nc, 36)
	dOut := randF32(b*tot, 37)
	dState := make([]float32, len(state))
	dOmega, dK := make([]float32, len(omega)), make([]float32, len(K))
	dOmegaC, dKc, dDrive := make([]float32, len(omegaC)), make([]float32, len(Kc)), make([]float32, len(drive))
	conditionalKuramotoBackwardAccumulate(dState, dOmega, dK, dOmegaC, dKc, dDrive, state, K, Kc, drive, dOut, b, n, nc, 1, 1, 1)
	loss := func() float64 {
		v := make([]float32, len(state))
		trig := make([]float64, 2*tot)
		conditionalKuramotoForwardInto(v, state, omega, omegaC, K, Kc, drive, b, n, nc, 1, 1, 1, trig[:tot], trig[tot:])
		var s float64
		for i := range v {
			s += float64(dOut[i]) * float64(v[i])
		}
		return s
	}
	idx := func(a []float32) []int { return []int{0, len(a) / 2, len(a) - 1} }
	finiteDiffGradCheck(t, "dState", state, dState, idx(state), 1e-4, 1e-2, loss)
	finiteDiffGradCheck(t, "dOmega", omega, dOmega, idx(omega), 1e-4, 1e-2, loss)
	finiteDiffGradCheck(t, "dK", K, dK, idx(K), 1e-4, 1e-2, loss)
	finiteDiffGradCheck(t, "dOmegaC", omegaC, dOmegaC, idx(omegaC), 1e-4, 1e-2, loss)
	finiteDiffGradCheck(t, "dKc", Kc, dKc, idx(Kc), 1e-4, 1e-2, loss)
	finiteDiffGradCheck(t, "dDrive", drive, dDrive, idx(drive), 1e-4, 1e-2, loss)
}

// TestReadoutBackwardFiniteDiff: ref-oscillator sin/cos readout VJP vs
// central differences.
func TestReadoutBackwardFiniteDiff(t *testing.T) {
	b, n := 2, 6
	phases := randF32(b*n, 21)
	dFeat := randF32(b*2*n, 22)
	dPhases := make([]float32, len(phases))
	readoutTransformBackwardInto(dPhases, n, 0, dFeat, phases, b, n, n, 0, "ref_oscillator", "sin_cos")
	loss := func() float64 {
		f := readoutTransform(phases, b, n, n, 0, "ref_oscillator", "sin_cos")
		var s float64
		for i := range f {
			s += float64(dFeat[i]) * float64(f[i])
		}
		return s
	}
	finiteDiffGradCheck(t, "dPhases", phases, dPhases, []int{0, 1, n, b*n - 1}, 1e-4, 1e-2, loss)
}
