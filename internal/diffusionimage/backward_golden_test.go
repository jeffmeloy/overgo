package diffusionimage

import (
	"math"
	"testing"
)

type gradSample struct {
	Index int
	Value float64
}

type uditGradFixture struct {
	DOut        []float32
	DX          []float32
	GradSamples map[string][]gradSample
}

// TestTinyModelBackwardMatchesTorch: full-model VJP vs the torch autograd
// golden — every tensor class (res/attn blocks, transitions, projections,
// scalars incl. both uses of the shared middle residual scale) is sampled.
func TestTinyModelBackwardMatchesTorch(t *testing.T) {
	fx := loadGolden[uditTinyFixture](t, "udit_tiny")
	oracle := loadGolden[uditGradFixture](t, "udit_tiny_grad")
	m := compileTiny(t, fx)
	_, trace, err := m.forward(fx.X, fx.B, fx.H, fx.W, true)
	if err != nil {
		t.Fatal(err)
	}
	grads := Grads{}
	dx, err := m.backwardFromTrace(trace, oracle.DOut, grads)
	if err != nil {
		t.Fatal(err)
	}
	requireWithin(t, "tiny dX", dx, oracle.DX, tolGrad)
	worst := 0.0
	for key, samples := range oracle.GradSamples {
		got, ok := grads[key]
		if !ok {
			t.Fatalf("missing grad for %s", key)
		}
		for _, sample := range samples {
			if sample.Index < 0 || sample.Index >= len(got) {
				t.Fatalf("%s grad sample index %d outside len %d", key, sample.Index, len(got))
			}
			diff := math.Abs(float64(got[sample.Index]) - sample.Value)
			if diff > worst {
				worst = diff
			}
			if diff > tolGrad {
				t.Fatalf("%s[%d] grad got %.9g want %.9g diff %.3g > %.3g", key, sample.Index, got[sample.Index], sample.Value, diff, tolGrad)
			}
		}
	}
	t.Logf("tiny grad samples: %d tensors, worst abs diff %.6e (gate %.0e)", len(oracle.GradSamples), worst, tolGrad)
}

// dotLoss: <dOut, f(...)> — the linear functional every FD check probes.
func dotLoss(dOut, y []float32) float64 {
	var total float64
	for i, v := range y {
		total += float64(dOut[i]) * float64(v)
	}
	return total
}

func constGrad(n int, seed int) []float32 {
	out := make([]float32, n)
	state := uint32(seed)*2654435761 + 1
	for i := range out {
		state = state*1664525 + 1013904223
		out[i] = float32(int32(state>>16)%256-128) / 256
	}
	return out
}

func TestXATGLUBackwardFiniteDiff(t *testing.T) {
	fx := loadGolden[struct {
		Rows      int       `json:"rows"`
		OutDim    int       `json:"out_dim"`
		Alpha     float64   `json:"alpha"`
		Projected []float32 `json:"projected"`
		Out       []float32 `json:"out"`
	}](t, "xatglu")
	dOut := constGrad(fx.Rows*fx.OutDim, 3)
	dProjected := make([]float32, len(fx.Projected))
	dAlpha := xatgluBackward(dProjected, fx.Projected, dOut, fx.Alpha, fx.Rows, fx.OutDim)
	loss := func() float64 {
		y := make([]float32, fx.Rows*fx.OutDim)
		xatgluGateInto(y, fx.Projected, fx.Alpha, fx.Rows, fx.OutDim)
		return dotLoss(dOut, y)
	}
	finiteDiffCheck(t, "xATGLU dProjected", fx.Projected, dProjected, spotIndices(len(fx.Projected)), loss)
	// Alpha FD: perturb the scalar through a wrapper loss.
	const eps = 3e-3
	alphaLoss := func(a float64) float64 {
		y := make([]float32, fx.Rows*fx.OutDim)
		xatgluGateInto(y, fx.Projected, a, fx.Rows, fx.OutDim)
		return dotLoss(dOut, y)
	}
	num := (alphaLoss(fx.Alpha+eps) - alphaLoss(fx.Alpha-eps)) / (2 * eps)
	if d := math.Abs(num - float64(dAlpha)); d > 2e-2*math.Max(math.Abs(num), 1) {
		t.Errorf("xATGLU dAlpha analytic=%.6g finite-diff=%.6g", dAlpha, num)
	}
}

func TestCPFactorBackwardFiniteDiff(t *testing.T) {
	fx := loadGolden[struct {
		Tokens int       `json:"bt"`
		NHead  int       `json:"n_head"`
		HeadD  int       `json:"head_dim"`
		QRank  int       `json:"q_rank"`
		AQ     []float32 `json:"A_q"`
		BQ     []float32 `json:"B_q"`
	}](t, "cplinear")
	dOut := constGrad(fx.Tokens*fx.NHead*fx.HeadD, 5)
	dA := make([]float32, len(fx.AQ))
	dB := make([]float32, len(fx.BQ))
	cpFactorContractBackward(dA, dB, fx.AQ, fx.BQ, dOut, fx.Tokens, fx.NHead, fx.QRank, fx.HeadD)
	loss := func() float64 {
		y := make([]float32, fx.Tokens*fx.NHead*fx.HeadD)
		cpFactorContractInto(y, fx.AQ, fx.BQ, fx.Tokens, fx.NHead, fx.QRank, fx.HeadD)
		return dotLoss(dOut, y)
	}
	finiteDiffCheck(t, "CP dA", fx.AQ, dA, spotIndices(len(fx.AQ)), loss)
	finiteDiffCheck(t, "CP dB", fx.BQ, dB, spotIndices(len(fx.BQ)), loss)
}

func TestGroupNormBackwardFiniteDiff(t *testing.T) {
	fx := loadGolden[struct {
		N      int       `json:"N"`
		C      int       `json:"C"`
		H      int       `json:"H"`
		W      int       `json:"W"`
		Groups int       `json:"groups"`
		Eps    float64   `json:"eps"`
		Weight []float32 `json:"weight"`
		Bias   []float32 `json:"bias"`
		X      []float32 `json:"x"`
	}](t, "groupnorm")
	dOut := constGrad(len(fx.X), 7)
	dx := make([]float32, len(fx.X))
	dW := make([]float32, fx.C)
	dB := make([]float32, fx.C)
	groupNormBackward(dx, dW, dB, fx.X, fx.Weight, dOut, fx.N, fx.C, fx.H, fx.W, fx.Groups, fx.Eps)
	loss := func() float64 {
		y := make([]float32, len(fx.X))
		groupNormInto(y, fx.X, fx.Weight, fx.Bias, fx.N, fx.C, fx.H, fx.W, fx.Groups, fx.Eps)
		return dotLoss(dOut, y)
	}
	finiteDiffCheck(t, "GroupNorm dx", fx.X, dx, spotIndices(len(fx.X)), loss)
	finiteDiffCheck(t, "GroupNorm dW", fx.Weight, dW, spotIndices(fx.C), loss)
	finiteDiffCheck(t, "GroupNorm dB", fx.Bias, dB, spotIndices(fx.C), loss)
}

func TestProjectionLeafOpsBackwardFiniteDiff(t *testing.T) {
	fx := loadGolden[projectionOpsFixture](t, "projection_ops")
	for _, tc := range []struct {
		name string
		fx   convProjectionFixture
	}{{"patch Conv2d", fx.Patch}, {"1x1 Conv2d", fx.OneByOne}} {
		dOut := constGrad(len(tc.fx.Y), 11)
		grads := Grads{}
		dx := conv2dValidStrideBackward(grads, "op", tc.fx.X, tc.fx.Weight, dOut, tc.fx.B, tc.fx.CIn, tc.fx.COut, tc.fx.H, tc.fx.W, tc.fx.Kernel, tc.fx.Stride)
		loss := func() float64 {
			y, _, _ := conv2dValidStride(tc.fx.X, tc.fx.Weight, tc.fx.Bias, tc.fx.B, tc.fx.CIn, tc.fx.COut, tc.fx.H, tc.fx.W, tc.fx.Kernel, tc.fx.Stride)
			return dotLoss(dOut, y)
		}
		finiteDiffCheck(t, tc.name+" dx", tc.fx.X, dx, spotIndices(len(tc.fx.X)), loss)
		finiteDiffCheck(t, tc.name+" dW", tc.fx.Weight, grads["op.weight"], spotIndices(len(tc.fx.Weight)), loss)
		finiteDiffCheck(t, tc.name+" dB", tc.fx.Bias, grads["op.bias"], spotIndices(tc.fx.COut), loss)
	}
	tr := fx.Transpose
	dOut := constGrad(len(tr.Y), 13)
	grads := Grads{}
	dx := convTranspose2dStrideBackward(grads, "op", tr.X, tr.Weight, dOut, tr.B, tr.CIn, tr.COut, tr.H, tr.W, tr.Kernel, tr.Stride)
	loss := func() float64 {
		y, _, _ := convTranspose2dStride(tr.X, tr.Weight, tr.Bias, tr.B, tr.CIn, tr.COut, tr.H, tr.W, tr.Kernel, tr.Stride)
		return dotLoss(dOut, y)
	}
	finiteDiffCheck(t, "ConvTranspose dx", tr.X, dx, spotIndices(len(tr.X)), loss)
	finiteDiffCheck(t, "ConvTranspose dW", tr.Weight, grads["op.weight"], spotIndices(len(tr.Weight)), loss)
	finiteDiffCheck(t, "ConvTranspose dB", tr.Bias, grads["op.bias"], spotIndices(tr.COut), loss)
}

func TestResBlockBackwardFiniteDiff(t *testing.T) {
	fx := loadGolden[resBlockFixture](t, "resblock")
	m := fx.model()
	blk := fx.block()
	dOut := constGrad(len(fx.X), 17)
	grads := Grads{}
	dx := blk.backward(m, fx.X, dOut, fx.N, fx.C, fx.H, fx.W, grads)
	loss := func() float64 {
		return dotLoss(dOut, blk.forward(m, fx.X, fx.N, fx.C, fx.H, fx.W))
	}
	finiteDiffCheck(t, "ResBlock dx", fx.X, dx, spotIndices(len(fx.X)), loss)
	finiteDiffCheck(t, "ResBlock dC1W", fx.C1W, grads["block.conv1.weight"], spotIndices(len(fx.C1W)), loss)
	finiteDiffCheck(t, "ResBlock dC2B", fx.C2B, grads["block.conv2.bias"], spotIndices(len(fx.C2B)), loss)
	finiteDiffCheck(t, "ResBlock dN1W", fx.N1W, grads["block.norm1.weight"], spotIndices(len(fx.N1W)), loss)
	// Residual-scale scalar via a wrapped loss.
	const eps = 3e-3
	scaleLoss := func(s float32) float64 {
		save := blk.residualScale
		blk.residualScale = s
		defer func() { blk.residualScale = save }()
		return dotLoss(dOut, blk.forward(m, fx.X, fx.N, fx.C, fx.H, fx.W))
	}
	num := (scaleLoss(blk.residualScale+eps) - scaleLoss(blk.residualScale-eps)) / (2 * eps)
	analytic := float64(grads["block.learned_residual_scale"][0])
	if d := math.Abs(num - analytic); d > 2e-2*math.Max(math.Abs(num), 1) {
		t.Errorf("ResBlock dScale analytic=%.6g finite-diff=%.6g", analytic, num)
	}
}

func TestTransformerBlockBackwardFiniteDiff(t *testing.T) {
	fx := loadGolden[transformerBlockFixture](t, "transformerblock")
	m := fx.model()
	blk := fx.block()
	dOut := constGrad(len(fx.X), 19)
	grads := Grads{}
	dx := blk.backward(m, fx.X, dOut, fx.N, fx.C, fx.H, fx.W, grads)
	loss := func() float64 {
		return dotLoss(dOut, blk.forward(m, fx.X, fx.N, fx.C, fx.H, fx.W))
	}
	finiteDiffCheck(t, "TransformerBlock dX", fx.X, dx, spotIndices(len(fx.X)), loss)
	finiteDiffCheck(t, "TransformerBlock dWAq", fx.WAq, grads["block.attn.c_qkv.W_A_q.weight"], spotIndices(len(fx.WAq)), loss)
	finiteDiffCheck(t, "TransformerBlock dWBv", fx.WBv, grads["block.attn.c_qkv.W_B_v.weight"], spotIndices(len(fx.WBv)), loss)
	finiteDiffCheck(t, "TransformerBlock dWoproj", fx.Woproj, grads["block.attn.o_proj.proj.weight"], spotIndices(len(fx.Woproj)), loss)
	finiteDiffCheck(t, "TransformerBlock dBoproj", fx.Boproj, grads["block.attn.o_proj.proj.bias"], spotIndices(len(fx.Boproj)), loss)
	finiteDiffCheck(t, "TransformerBlock dMLPGateW", fx.MLPGateW, grads["block.mlp.0.proj.weight"], spotIndices(len(fx.MLPGateW)), loss)
	finiteDiffCheck(t, "TransformerBlock dMLPDownW", fx.MLPDownW, grads["block.mlp.1.weight"], spotIndices(len(fx.MLPDownW)), loss)
	finiteDiffCheck(t, "TransformerBlock dNorm1W", fx.Norm1W, grads["block.norm1.weight"], spotIndices(len(fx.Norm1W)), loss)
	finiteDiffCheck(t, "TransformerBlock dNorm2B", fx.Norm2B, grads["block.norm2.bias"], spotIndices(len(fx.Norm2B)), loss)
	// Scalar grads (attention/MLP residual scales, both alphas) via wrapped losses.
	const eps = 3e-3
	scalarChecks := []struct {
		name     string
		analytic float64
		loss     func(delta float32) float64
	}{
		{"dAttnScale", float64(grads["block.learned_residual_scale_attn"][0]), func(delta float32) float64 {
			save := blk.attentionScale
			blk.attentionScale += delta
			defer func() { blk.attentionScale = save }()
			return loss()
		}},
		{"dMLPScale", float64(grads["block.learned_residual_scale_mlp"][0]), func(delta float32) float64 {
			save := blk.mlpScale
			blk.mlpScale += delta
			defer func() { blk.mlpScale = save }()
			return loss()
		}},
		{"dAlphaO", float64(grads["block.attn.o_proj.alpha"][0]), func(delta float32) float64 {
			save := blk.attention.AlphaO
			blk.attention.AlphaO += float64(delta)
			defer func() { blk.attention.AlphaO = save }()
			return loss()
		}},
		{"dMLPAlpha", float64(grads["block.mlp.0.alpha"][0]), func(delta float32) float64 {
			save := blk.mlpAlpha
			blk.mlpAlpha += delta
			defer func() { blk.mlpAlpha = save }()
			return loss()
		}},
	}
	for _, check := range scalarChecks {
		num := (check.loss(eps) - check.loss(-eps)) / (2 * eps)
		if d := math.Abs(num - check.analytic); d > 2e-2*math.Max(math.Abs(num), 1) {
			t.Errorf("TransformerBlock %s analytic=%.6g finite-diff=%.6g", check.name, check.analytic, num)
		}
	}
}

func TestPoolAndUpsampleBackwardFiniteDiff(t *testing.T) {
	const b, c, h, w = 1, 2, 4, 4
	x := constGrad(b*c*h*w, 23)
	dOutPool := constGrad(b*c*h*w/4, 29)
	dxPool := avgPool2xBackward(dOutPool, b, c, h, w)
	poolLoss := func() float64 { return dotLoss(dOutPool, avgPool2x(x, b, c, h, w)) }
	finiteDiffCheck(t, "avgPool2x dx", x, dxPool, spotIndices(len(x)), poolLoss)

	dOutUp := constGrad(b*c*h*w*4, 31)
	dxUp := upsampleNearest2xBackward(dOutUp, b, c, h, w)
	upLoss := func() float64 { return dotLoss(dOutUp, upsampleNearest2x(x, b, c, h, w)) }
	finiteDiffCheck(t, "upsample2x dx", x, dxUp, spotIndices(len(x)), upLoss)
}
