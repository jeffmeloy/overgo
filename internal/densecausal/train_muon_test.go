package densecausal

import (
	"math"
	"math/rand"
	"strconv"
	"testing"
)

// tinyMuonModel builds a seeded, in-memory dense-causal model (tied embeddings,
// no attention bias) so the Muon training loop is exercised with no fixture or
// real artifact. Geometry: Vocab 32, Hidden 16, 2 heads x head-dim 8, 2 layers,
// intermediate 32.
func tinyMuonModel(t *testing.T) *Model {
	t.Helper()
	const (
		vocab   = 32
		hidden  = 16
		heads   = 2
		headDim = 8
		kvHeads = 2
		inter   = 32
		layers  = 2
	)
	rng := rand.New(rand.NewSource(1))
	weights := map[string][]float32{}
	shapes := map[string][]int{}
	matrix := func(name string, rows, cols int) {
		values := make([]float32, rows*cols)
		for i := range values {
			values[i] = float32(rng.NormFloat64() * 0.02)
		}
		weights[name] = values
		shapes[name] = []int{rows, cols}
	}
	norm := func(name string, n int) {
		values := make([]float32, n)
		for i := range values {
			values[i] = float32(1 + rng.NormFloat64()*0.01)
		}
		weights[name] = values
		shapes[name] = []int{n}
	}
	matrix("model.embed_tokens.weight", vocab, hidden)
	for layer := 0; layer < layers; layer++ {
		prefix := "model.layers." + strconv.Itoa(layer) + "."
		norm(prefix+"input_layernorm.weight", hidden)
		matrix(prefix+"self_attn.q_proj.weight", heads*headDim, hidden)
		matrix(prefix+"self_attn.k_proj.weight", kvHeads*headDim, hidden)
		matrix(prefix+"self_attn.v_proj.weight", kvHeads*headDim, hidden)
		matrix(prefix+"self_attn.o_proj.weight", hidden, heads*headDim)
		norm(prefix+"post_attention_layernorm.weight", hidden)
		matrix(prefix+"mlp.gate_proj.weight", inter, hidden)
		matrix(prefix+"mlp.up_proj.weight", inter, hidden)
		matrix(prefix+"mlp.down_proj.weight", hidden, inter)
	}
	norm("model.norm.weight", hidden)

	m, err := NewModel(weights, shapes, heads, headDim, 10000, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// TestMuonTrainingDecreasesLossTiny: multi-step Muon training over a fixed batch
// reduces the loss on the seeded in-memory model, with the base learning rate
// DERIVED from the parameter count (baseLR=0), not hand-tuned. Loss finite
// throughout.
func TestMuonTrainingDecreasesLossTiny(t *testing.T) {
	m := tinyMuonModel(t)
	tokens := []int{1, 5, 9, 3, 7, 2, 11, 4}
	const steps = 20
	trajectory, err := m.Train(tokens, steps, 0, 0.9)
	if err != nil {
		t.Fatal(err)
	}
	if len(trajectory) != steps {
		t.Fatalf("trajectory length %d, want %d", len(trajectory), steps)
	}
	for i, loss := range trajectory {
		if math.IsNaN(loss) || math.IsInf(loss, 0) {
			t.Fatalf("trajectory[%d] = %g is not finite", i, loss)
		}
	}
	after, _, err := m.Loss(tokens)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("loss %.6f -> %.6f over %d Muon steps", trajectory[0], after, steps)
	if !(after < trajectory[0]) {
		t.Fatalf("Muon training did not reduce loss: before %.6f after %.6f", trajectory[0], after)
	}
}

// TestTrainResumeMatchesUninterrupted: an exact step-boundary checkpoint/resume
// split (5 steps, snapshot optimizer state, resume 5 more) produces bitwise-
// identical weights to an uninterrupted 10-step run. Both use the derived rate.
func TestTrainResumeMatchesUninterrupted(t *testing.T) {
	tokens := []int{1, 5, 9, 3, 7, 2, 11, 4}
	a := tinyMuonModel(t)
	if _, err := a.Train(tokens, 10, 0, 0.9); err != nil {
		t.Fatal(err)
	}
	b := tinyMuonModel(t)
	_, state, err := b.TrainResume(tokens, 5, 0, 0.9, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := b.TrainResume(tokens, 5, 0, 0.9, &state); err != nil {
		t.Fatal(err)
	}
	for name, wa := range a.Weights {
		wb := b.Weights[name]
		for i := range wa {
			if wa[i] != wb[i] {
				t.Fatalf("weight %q[%d]: uninterrupted %g != resumed %g", name, i, wa[i], wb[i])
			}
		}
	}
}

// Real-artifact Muon training is deliberately NOT gated here: one host
// Newton-Schulz step over a 500M model's large matrices exceeds go test's 600s
// timeout. Real-model Muon verification belongs to the device-NS rung (rung 2)
// or a research lane with its own timeout; the existing single-step SGD
// TestRealArtifactTrainingStepDecreasesLoss already exercises the real backward.
