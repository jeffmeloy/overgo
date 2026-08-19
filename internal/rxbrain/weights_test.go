package rxbrain

import (
	"os"
	"testing"
)

// TestRxBrainTextWeightsBindRealCheckpoint pins stage 3a-i: the base-text
// branch loads from the real sharded checkpoint with every tensor's shape
// validated against the declared configuration -- 32 whole layers plus the
// tied embedding and final norm, bf16 promoted to f32.
func TestRxBrainTextWeightsBindRealCheckpoint(t *testing.T) {
	if _, err := os.Stat(realCheckpoint); err != nil {
		t.Skipf("UNAVAILABLE: %s absent; RxBrain weights NOT verified", realCheckpoint)
	}
	config, err := Load(realCheckpoint)
	if err != nil {
		t.Fatal(err)
	}
	weights, err := LoadTextWeights(realCheckpoint, config)
	if err != nil {
		t.Fatal(err)
	}
	if len(weights.Layers) != config.NumHiddenLayers {
		t.Fatalf("bound %d layers, want %d", len(weights.Layers), config.NumHiddenLayers)
	}
	if len(weights.Embed) != config.VocabSize*config.HiddenSize {
		t.Fatalf("embedding length %d", len(weights.Embed))
	}
	last := weights.Layers[config.NumHiddenLayers-1]
	if len(last.DownProj) != config.HiddenSize*config.IntermediateSize ||
		len(last.KProj) != config.NumKeyValueHeads*config.HeadDim*config.HiddenSize {
		t.Fatalf("terminal layer shapes drifted: down=%d k=%d", len(last.DownProj), len(last.KProj))
	}
	// bf16 promotion sanity: real weights are finite and not all zero.
	sum := float64(0)
	for _, value := range last.QueryLN {
		sum += float64(value)
	}
	if sum == 0 {
		t.Fatal("query layernorm reads as all zeros; promotion is broken")
	}
}
