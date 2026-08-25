package densecausal

import "testing"

// qwen2 differs from llama only by q/k/v projection biases; same code path
// with bias slices non-nil. Golden: fixtures/qwen2_train_golden.json, a
// committed torch-oracle capture (tiny seeded dims, f64 values); the
// fixture is the authority and is not regenerated in-repo.

func readQwen2Golden(t *testing.T) *tinyGolden {
	t.Helper()
	return readGolden(t, "qwen2_train_golden.json", "qwen2_train_golden/v1")
}

func TestTinyQwen2LossAndGradsMatchGolden(t *testing.T) {
	g := readQwen2Golden(t)
	m := modelFromGolden(t, g)
	if !m.Dims.AttnBias {
		t.Fatal("qwen2 golden must derive AttnBias=true")
	}
	assertTrainGoldenParity(t, g, m)
}

func TestTinyQwen2ForwardOnlyLossAgrees(t *testing.T) {
	g := readQwen2Golden(t)
	m := modelFromGolden(t, g)
	lossForward, _, err := m.Loss(g.Tokens)
	if err != nil {
		t.Fatal(err)
	}
	lossGrad, _, _, err := m.LossAndGrads(g.Tokens)
	if err != nil {
		t.Fatal(err)
	}
	if lossForward != lossGrad {
		t.Fatalf("forward-only loss %g != gradient-path loss %g", lossForward, lossGrad)
	}
}

// TestQwen2PartialBiasTripleRejected: q/k/v biases are all-or-none per
// layer; a missing member must fail loudly at geometry derivation.
func TestQwen2PartialBiasTripleRejected(t *testing.T) {
	g := readQwen2Golden(t)
	weights := make(map[string][]float32, len(g.Params))
	shapes := make(map[string][]int, len(g.Params))
	for name, p := range g.Params {
		if name == "model.layers.1.self_attn.k_proj.bias" {
			continue
		}
		values := make([]float32, len(p.Values))
		for i, v := range p.Values {
			values[i] = float32(v)
		}
		weights[name] = values
		shapes[name] = p.Shape
	}
	if _, err := NewModel(weights, shapes, g.Config.NumAttentionHeads, g.Config.HeadDim, g.Config.RopeTheta, g.Config.RMSNormEps); err == nil {
		t.Fatal("partial bias triple accepted; want loud failure")
	} else {
		t.Logf("rejected as expected: %v", err)
	}
}

func TestRealQwen2ArtifactTrainingStepDecreasesLoss(t *testing.T) {
	requireLongTest(t)
	runRealArtifactStep(t, "Qwen2.5-0.5B")
}
