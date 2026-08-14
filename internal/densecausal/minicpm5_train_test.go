package densecausal

import (
	"math"
	"testing"
)

// Untied-head coverage strategy: NO new tiny torch golden. The llama math
// (rope/attention/mlp/norm/CE forward+backward) is already pinned by
// llama_train_golden.json; the untied delta only reroutes the head grad to
// its own slot. That routing is pinned here against the EXISTING tied golden:
// duplicating the embedding as lm_head.weight must reproduce the tied loss
// and logits exactly, with grads splitting as head-contribution vs
// input-scatter whose sum equals the tied embedding grad.
func TestUntiedHeadGradsSplitTiedGolden(t *testing.T) {
	g := readTinyGolden(t)
	tied := modelFromGolden(t, g)
	lossTied, _, gradsTied, err := tied.LossAndGrads(g.Tokens)
	if err != nil {
		t.Fatal(err)
	}

	weights := make(map[string][]float32, len(g.Params)+1)
	shapes := make(map[string][]int, len(g.Params)+1)
	for name, p := range g.Params {
		values := make([]float32, len(p.Values))
		for i, v := range p.Values {
			values[i] = float32(v)
		}
		weights[name] = values
		shapes[name] = p.Shape
	}
	head := append([]float32(nil), weights["model.embed_tokens.weight"]...)
	weights["lm_head.weight"] = head
	shapes["lm_head.weight"] = append([]int(nil), shapes["model.embed_tokens.weight"]...)
	untied, err := NewModel(weights, shapes, g.Config.NumAttentionHeads, g.Config.HeadDim, g.Config.RopeTheta, g.Config.RMSNormEps)
	if err != nil {
		t.Fatal(err)
	}
	if untied.HeadName != "lm_head.weight" {
		t.Fatalf("head name %q, want lm_head.weight", untied.HeadName)
	}

	lossUntied, _, gradsUntied, err := untied.LossAndGrads(g.Tokens)
	if err != nil {
		t.Fatal(err)
	}
	if lossUntied != lossTied {
		t.Fatalf("untied loss %.9f != tied %.9f", lossUntied, lossTied)
	}
	gradHead := gradsUntied["lm_head.weight"]
	gradScatter := gradsUntied["model.embed_tokens.weight"]
	gradTiedEmbed := gradsTied["model.embed_tokens.weight"]
	if gradHead == nil || gradScatter == nil {
		t.Fatal("untied grads missing lm_head.weight or model.embed_tokens.weight slot")
	}
	var headNorm, scatterNorm float64
	for i := range gradTiedEmbed {
		if got, want := gradHead[i]+gradScatter[i], gradTiedEmbed[i]; math.Abs(float64(got-want)) > 1e-7 {
			t.Fatalf("grad element %d: head %g + scatter %g = %g, tied %g", i, gradHead[i], gradScatter[i], got, want)
		}
		headNorm += float64(gradHead[i]) * float64(gradHead[i])
		scatterNorm += float64(gradScatter[i]) * float64(gradScatter[i])
	}
	// Both contributions must be alive — a zero side means the split is inert.
	if headNorm == 0 || scatterNorm == 0 {
		t.Fatalf("inert split: |head|^2=%g |scatter|^2=%g", headNorm, scatterNorm)
	}
	// Every other parameter grad is identical.
	for name, tiedGrad := range gradsTied {
		if name == "model.embed_tokens.weight" {
			continue
		}
		untiedGrad := gradsUntied[name]
		if len(untiedGrad) != len(tiedGrad) {
			t.Fatalf("grad %q length %d, tied %d", name, len(untiedGrad), len(tiedGrad))
		}
		for i := range tiedGrad {
			if untiedGrad[i] != tiedGrad[i] {
				t.Fatalf("grad %q element %d: untied %g, tied %g", name, i, untiedGrad[i], tiedGrad[i])
			}
		}
	}
}

func TestUntiedHeadWithoutTensorRejected(t *testing.T) {
	// NewModel derives tied-vs-untied from tensor presence; a malformed
	// lm_head shape must fail loudly.
	g := readTinyGolden(t)
	weights := make(map[string][]float32, len(g.Params)+1)
	shapes := make(map[string][]int, len(g.Params)+1)
	for name, p := range g.Params {
		values := make([]float32, len(p.Values))
		for i, v := range p.Values {
			values[i] = float32(v)
		}
		weights[name] = values
		shapes[name] = p.Shape
	}
	weights["lm_head.weight"] = make([]float32, g.Config.HiddenSize)
	shapes["lm_head.weight"] = []int{1, g.Config.HiddenSize}
	if _, err := NewModel(weights, shapes, g.Config.NumAttentionHeads, g.Config.HeadDim, g.Config.RopeTheta, g.Config.RMSNormEps); err == nil {
		t.Fatal("mis-shaped lm_head.weight accepted; want loud failure")
	} else {
		t.Logf("rejected as expected: %v", err)
	}
}

func TestRealMiniCPM5ArtifactTrainingStepDecreasesLoss(t *testing.T) {
	requireLongTest(t)
	// First untied artifact on the ladder (tie_word_embeddings=false,
	// head_dim 128 != hidden/heads 96, sharded-index checkpoint).
	runRealArtifactStep(t, "MiniCPM5-1B")
}
