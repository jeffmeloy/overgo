package rxbrain

import (
	"os"
	"testing"
)

// TestRxBrainVisionWeightsBindRealCheckpoint pins stage 3b-i: the whole
// SigLIP-style visual tower binds from the real sharded checkpoint with
// every tensor's shape validated against the declared vision configuration
// -- 27 LayerNorm blocks with packed QKV and biases, the interpolatable
// position table, and the merger projecting into the decoder's hidden
// size. Merger output dimension equals text hidden (2048), proven against
// the checkpoint's own shapes, not the declared out_hidden_size (the fused
// pooler input).
func TestRxBrainVisionWeightsBindRealCheckpoint(t *testing.T) {
	if _, err := os.Stat(realCheckpoint); err != nil {
		t.Skipf("UNAVAILABLE: %s absent; RxBrain vision weights NOT verified", realCheckpoint)
	}
	config, err := Load(realCheckpoint)
	if err != nil {
		t.Fatal(err)
	}
	vision := config.Vision
	if vision.PositionSide() != vision.MaxImageSize/vision.PatchSize {
		t.Fatalf("position side %d", vision.PositionSide())
	}
	weights, err := LoadVisionWeights(realCheckpoint, config)
	if err != nil {
		t.Fatal(err)
	}
	if len(weights.Blocks) != vision.NumHiddenLayers {
		t.Fatalf("bound %d blocks, want %d", len(weights.Blocks), vision.NumHiddenLayers)
	}
	if len(weights.PosEmbed) != vision.PositionSide()*vision.PositionSide()*vision.HiddenSize {
		t.Fatalf("position table length %d", len(weights.PosEmbed))
	}
	if len(weights.MergerProj1W) != config.HiddenSize*vision.HiddenSize ||
		len(weights.MergerPool0W) != config.HiddenSize*2*config.HiddenSize {
		t.Fatalf("merger shapes drifted: proj1=%d pool0=%d", len(weights.MergerProj1W), len(weights.MergerPool0W))
	}
	last := weights.Blocks[vision.NumHiddenLayers-1]
	if len(last.QKVW) != 3*vision.HiddenSize*vision.HiddenSize || len(last.FC1W) != vision.IntermediateSize*vision.HiddenSize {
		t.Fatalf("terminal block shapes drifted: qkv=%d fc1=%d", len(last.QKVW), len(last.FC1W))
	}
	// bf16 promotion sanity: LayerNorm weights are real values, not zeros.
	sum := float64(0)
	for _, value := range last.Norm2W {
		sum += float64(value)
	}
	if sum == 0 {
		t.Fatal("norm2 weight reads as all zeros; promotion is broken")
	}
	t.Logf("vision tower bound: %d blocks, hidden=%d inter=%d patch=%d side=%d",
		len(weights.Blocks), vision.HiddenSize, vision.IntermediateSize, vision.PatchSize, vision.PositionSide())
}
