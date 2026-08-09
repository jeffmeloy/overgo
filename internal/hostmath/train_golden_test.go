package hostmath

import (
	"math"
	"testing"
)

// Component goldens for the llama-architecture training primitives
// (scripts/gen_llama_train_golden.py): GQA causal attention, interleaved
// rope, SiLU-gated MLP, mean softmax cross-entropy.

func TestCausalAttentionGQAMatchesGolden(t *testing.T) {
	var g struct {
		Seq     int       `json:"seq"`
		Heads   int       `json:"heads"`
		KVHeads int       `json:"kv_heads"`
		HeadDim int       `json:"head_dim"`
		Q       []float64 `json:"q"`
		K       []float64 `json:"k"`
		V       []float64 `json:"v"`
		Out     []float64 `json:"out"`
		Dout    []float64 `json:"dout"`
		GradQ   []float64 `json:"grad_q"`
		GradK   []float64 `json:"grad_k"`
		GradV   []float64 `json:"grad_v"`
	}
	readFixture(t, "llama_attn_gqa_grad_golden.json", &g)
	// The golden applies score scale 1/sqrt(hd); hostmath's contract folds
	// the scale into q, so scale here and unscale dq below.
	scale := 1.0 / math.Sqrt(float64(g.HeadDim))
	q := f64To32(g.Q)
	for i := range q {
		q[i] = float32(float64(q[i]) * scale)
	}
	k, v := f64To32(g.K), f64To32(g.V)
	out := make([]float32, len(q))
	CausalAttention(out, q, k, v, g.Seq, g.Heads, g.KVHeads, g.HeadDim)
	checkClose(t, "out", out, g.Out)
	dq := make([]float32, len(q))
	dk := make([]float32, len(k))
	dv := make([]float32, len(v))
	CausalAttentionBackward(dq, dk, dv, q, k, v, f64To32(g.Dout), g.Seq, g.Heads, g.KVHeads, g.HeadDim)
	for i := range dq {
		dq[i] = float32(float64(dq[i]) * scale)
	}
	checkClose(t, "grad_q", dq, g.GradQ)
	checkClose(t, "grad_k", dk, g.GradK)
	checkClose(t, "grad_v", dv, g.GradV)
}

func TestRotaryInterleavedMatchesGolden(t *testing.T) {
	var g struct {
		HeadDim int       `json:"head_dim"`
		Pos     int       `json:"pos"`
		Theta   float64   `json:"theta"`
		X       []float64 `json:"x"`
		Out     []float64 `json:"out"`
		Dout    []float64 `json:"dout"`
		GradX   []float64 `json:"grad_x"`
	}
	readFixture(t, "llama_rope_interleaved_grad_golden.json", &g)
	inv := RopeInvFreq(g.Theta, g.HeadDim)
	x := f64To32(g.X)
	ApplyRotaryInterleaved(x, inv, g.Pos)
	checkClose(t, "out", x, g.Out)
	dx := f64To32(g.Dout)
	RotaryInterleavedBackward(dx, inv, g.Pos)
	checkClose(t, "grad_x", dx, g.GradX)
}

func TestGatedMLPMatchesGolden(t *testing.T) {
	var g struct {
		Rows      int       `json:"rows"`
		Dim       int       `json:"dim"`
		Inter     int       `json:"intermediate"`
		X         []float64 `json:"x"`
		WGate     []float64 `json:"w_gate"`
		WUp       []float64 `json:"w_up"`
		WDown     []float64 `json:"w_down"`
		Out       []float64 `json:"out"`
		Dout      []float64 `json:"dout"`
		GradX     []float64 `json:"grad_x"`
		GradWGate []float64 `json:"grad_w_gate"`
		GradWUp   []float64 `json:"grad_w_up"`
		GradWDown []float64 `json:"grad_w_down"`
	}
	readFixture(t, "llama_gatedmlp_grad_golden.json", &g)
	x, wg, wu, wd := f64To32(g.X), f64To32(g.WGate), f64To32(g.WUp), f64To32(g.WDown)
	rows, d, inter := g.Rows, g.Dim, g.Inter
	gate := make([]float32, rows*inter)
	up := make([]float32, rows*inter)
	h := make([]float32, rows*inter)
	out := make([]float32, rows*d)
	Linear(gate, x, wg, rows, d, inter)
	Linear(up, x, wu, rows, d, inter)
	SiLUGate(h, gate, up)
	Linear(out, h, wd, rows, inter, d)
	checkClose(t, "out", out, g.Out)

	dout := f64To32(g.Dout)
	dh := make([]float32, rows*inter)
	dwd := make([]float32, len(wd))
	LinearBackward(dh, dwd, nil, h, wd, dout, rows, inter, d, false)
	dGate := make([]float32, rows*inter)
	dUp := make([]float32, rows*inter)
	SiLUGateBackward(dGate, dUp, gate, up, dh)
	dx := make([]float32, rows*d)
	dwg := make([]float32, len(wg))
	dwu := make([]float32, len(wu))
	LinearBackward(dx, dwg, nil, x, wg, dGate, rows, d, inter, false)
	LinearBackward(dx, dwu, nil, x, wu, dUp, rows, d, inter, true)
	checkClose(t, "grad_x", dx, g.GradX)
	checkClose(t, "grad_w_gate", dwg, g.GradWGate)
	checkClose(t, "grad_w_up", dwu, g.GradWUp)
	checkClose(t, "grad_w_down", dwd, g.GradWDown)
}

func TestSoftmaxCrossEntropyMatchesGolden(t *testing.T) {
	var g struct {
		Rows       int       `json:"rows"`
		Classes    int       `json:"classes"`
		Logits     []float64 `json:"logits"`
		Targets    []int     `json:"targets"`
		Loss       float64   `json:"loss"`
		GradLogits []float64 `json:"grad_logits"`
	}
	readFixture(t, "llama_softmaxce_grad_golden.json", &g)
	logits := f64To32(g.Logits)
	dLogits := make([]float32, len(logits))
	loss := SoftmaxCrossEntropy(dLogits, logits, g.Targets, g.Rows, g.Classes)
	if d := math.Abs(loss - g.Loss); d > componentTol {
		t.Fatalf("loss |%g - %g| = %g > %g", loss, g.Loss, d, componentTol)
	}
	checkClose(t, "grad_logits", dLogits, g.GradLogits)
}
