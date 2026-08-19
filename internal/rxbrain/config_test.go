package rxbrain

import (
	"os"
	"testing"
)

const realCheckpoint = "C:/Users/jeffm/adaptive_new/models/Hy-Embodied-RxBrain-1.0"

// TestRxBrainConfigLoadsRealCheckpoint pins stage 1 of the port against the
// real checkpoint: the declared configuration loads, validates, and carries
// the dimensions the decoder math will build on -- 32 layers of hidden 2048
// with 16 heads over 4 KV heads at CLA share factor 2, vocabulary 120818.
func TestRxBrainConfigLoadsRealCheckpoint(t *testing.T) {
	if _, err := os.Stat(realCheckpoint); err != nil {
		t.Skipf("UNAVAILABLE: %s absent; RxBrain config NOT verified", realCheckpoint)
	}
	config, err := Load(realCheckpoint)
	if err != nil {
		t.Fatal(err)
	}
	if config.HiddenSize != 2048 || config.NumHiddenLayers != 32 ||
		config.NumAttentionHeads != 16 || config.NumKeyValueHeads != 4 ||
		config.ClaShareFactor != 2 || config.VocabSize != 120818 {
		t.Fatalf("declared dimensions drifted: %+v", config)
	}
	if config.EOSTokenID != 120020 || config.ImageStartTokenID != 120119 {
		t.Fatalf("token contract drifted: eos=%d image_start=%d", config.EOSTokenID, config.ImageStartTokenID)
	}
}

// TestRxBrainConfigRefusals pins the validation contract without the model.
func TestRxBrainConfigRefusals(t *testing.T) {
	valid := Config{
		Architectures: []string{"UnifiedMoTForConditionalGeneration"},
		HiddenSize:    2048, IntermediateSize: 6144, NumHiddenLayers: 32,
		NumAttentionHeads: 16, NumKeyValueHeads: 4, HeadDim: 128,
		VocabSize: 120818, ClaShareFactor: 2, EOSTokenID: 120020,
		Vision: VisionConfig{
			HiddenSize: 1152, IntermediateSize: 4304, NumHiddenLayers: 27,
			NumAttentionHeads: 16, NumChannels: 3, PatchSize: 16,
			MaxImageSize: 2048, SpatialMergeSize: 2, TemporalPatchSize: 1,
			TextHiddenSize: 2048,
		},
	}
	if err := valid.Validate(); err != nil {
		t.Fatal(err)
	}
	for name, mutate := range map[string]func(*Config){
		"wrong architecture":       func(c *Config) { c.Architectures = []string{"LlamaForCausalLM"} },
		"kv heads indivisible":     func(c *Config) { c.NumKeyValueHeads = 3 },
		"head dim mismatch":        func(c *Config) { c.HeadDim = 96 },
		"cla indivisible":          func(c *Config) { c.ClaShareFactor = 5 },
		"eos outside vocab":        func(c *Config) { c.EOSTokenID = 200000 },
		"zero layers":              func(c *Config) { c.NumHiddenLayers = 0 },
		"vision heads indivisible": func(c *Config) { c.Vision.NumAttentionHeads = 5 },
		"vision patch indivisible": func(c *Config) { c.Vision.MaxImageSize = 2050 },
		"vision text hidden drift": func(c *Config) { c.Vision.TextHiddenSize = 4096 },
		"vision zero merge":        func(c *Config) { c.Vision.SpatialMergeSize = 0 },
	} {
		broken := valid
		mutate(&broken)
		if err := broken.Validate(); err == nil {
			t.Fatalf("%s accepted", name)
		}
	}
}
