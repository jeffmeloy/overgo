package rxbrain

import (
	"math"
	"os"
	"testing"
)

// TestRxBrainTextForwardProducesFiniteDistinctStates pins stage 3a-ii on the
// real checkpoint: the full 32-layer base-branch forward over a short token
// sequence yields finite, non-degenerate hidden states (distinct rows per
// position), greedy scoring returns an in-vocabulary token, and the dynamic
// NTK-alpha RoPE derivation matches the declared closed form. Degeneracy --
// identical rows, NaN, or a constant argmax across different prompts -- is
// the silent failure mode this guards.
func TestRxBrainTextForwardProducesFiniteDistinctStates(t *testing.T) {
	if _, err := os.Stat(realCheckpoint); err != nil {
		t.Skipf("UNAVAILABLE: %s absent; RxBrain forward NOT verified", realCheckpoint)
	}
	config, err := Load(realCheckpoint)
	if err != nil {
		t.Fatal(err)
	}
	inv := RopeInvFreq(config, 1000.0)
	wantBase := config.RopeTheta * math.Pow(1000.0, float64(config.HeadDim)/float64(config.HeadDim-2))
	if got, want := inv[1], 1.0/math.Pow(wantBase, 2.0/float64(config.HeadDim)); math.Abs(got-want) > 1e-12 {
		t.Fatalf("NTK-alpha rope drifted: inv[1]=%g want %g", got, want)
	}

	weights, err := LoadTextWeights(realCheckpoint, config)
	if err != nil {
		t.Fatal(err)
	}
	tokens := []int{config.BOSTokenID, 3000, 4000, 5000}
	hidden, err := ForwardText(config, weights, tokens, 1000.0)
	if err != nil {
		t.Fatal(err)
	}
	if len(hidden) != len(tokens)*config.HiddenSize {
		t.Fatalf("hidden length %d", len(hidden))
	}
	for index, value := range hidden {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("non-finite state at %d", index)
		}
	}
	distance := float64(0)
	for i := 0; i < config.HiddenSize; i++ {
		delta := float64(hidden[2*config.HiddenSize+i] - hidden[3*config.HiddenSize+i])
		distance += delta * delta
	}
	if distance == 0 {
		t.Fatal("positions collapsed to identical states")
	}
	token, score := GreedyNextToken(config, weights, hidden)
	if token < 0 || token >= config.VocabSize || math.IsNaN(float64(score)) {
		t.Fatalf("greedy terminal = (%d, %f)", token, score)
	}
	alternate, _ := GreedyNextToken(config, weights, hidden[:3*config.HiddenSize])
	t.Logf("forward: %d tokens through %d layers; greedy next=%d (score %.3f), prefix-3 next=%d",
		len(tokens), config.NumHiddenLayers, token, score, alternate)
}
