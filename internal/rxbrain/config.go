// Package rxbrain ports Hy-Embodied-RxBrain inference: a HunYuanVL-family
// unified mixture-of-transformers checkpoint whose vision-QA path runs a
// SigLIP-style tower into a modality-routed decoder with cross-layer
// -attention KV sharing. The port derives from the model directory's own
// Python reference (model/config_unified_mot.py and siblings); the reference
// runtime in the comparison tree is concepts-only and is not copied.
//
// Stage 1 (this file): the declared configuration and the checkpoint tensor
// inventory, validated against each other so later stages build on checked
// facts rather than assumptions.
package rxbrain

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// Config carries the fields the vision-QA text path needs. The checkpoint
// declares many more; unknown fields are ignored on load and stage 2 pulls
// additional fields as its math requires them.
type Config struct {
	Architectures        []string `json:"architectures"`
	HiddenSize           int      `json:"hidden_size"`
	IntermediateSize     int      `json:"intermediate_size"`
	NumHiddenLayers      int      `json:"num_hidden_layers"`
	NumAttentionHeads    int      `json:"num_attention_heads"`
	NumKeyValueHeads     int      `json:"num_key_value_heads"`
	HeadDim              int      `json:"head_dim"`
	VocabSize            int      `json:"vocab_size"`
	OrgVocabSize         int      `json:"org_vocab_size"`
	ClaShareFactor       int      `json:"cla_share_factor"`
	RopeTheta            float64  `json:"rope_theta"`
	BOSTokenID           int      `json:"bos_token_id"`
	EOSTokenID           int      `json:"eos_token_id"`
	ImageStartTokenID    int      `json:"image_start_token_id"`
	FlowLatentPlaceholde int      `json:"flow_latent_placeholder_id"`
	HiddenAct            string   `json:"hidden_act"`

	Vision VisionConfig `json:"vision_config"`
}

// VisionConfig is the declared SigLIP-style tower contract: LayerNorm blocks
// with packed QKV and GELU MLPs, absolute position table interpolated over a
// fixed grid, and a 2x2 spatial merger projecting into the text stream.
type VisionConfig struct {
	HiddenSize        int    `json:"hidden_size"`
	IntermediateSize  int    `json:"intermediate_size"`
	NumHiddenLayers   int    `json:"num_hidden_layers"`
	NumAttentionHeads int    `json:"num_attention_heads"`
	NumChannels       int    `json:"num_channels"`
	PatchSize         int    `json:"patch_size"`
	MaxImageSize      int    `json:"max_image_size"`
	SpatialMergeSize  int    `json:"spatial_merge_size"`
	TemporalPatchSize int    `json:"temporal_patch_size"`
	TextHiddenSize    int    `json:"text_hidden_size"`
	HiddenAct         string `json:"hidden_act"`
}

// PositionSide is the side of the square position-embedding grid the tower
// interpolates over: declared max image size in patches.
func (v VisionConfig) PositionSide() int {
	return v.MaxImageSize / v.PatchSize
}

// PatchDim is the flattened pixel dimension of one temporal patch.
func (v VisionConfig) PatchDim() int {
	return v.NumChannels * v.TemporalPatchSize * v.PatchSize * v.PatchSize
}

// Load reads and validates the checkpoint's declared configuration.
func Load(directory string) (Config, error) {
	data, err := os.ReadFile(filepath.Join(directory, "config.json"))
	if err != nil {
		return Config{}, fmt.Errorf("rxbrain: read config: %w", err)
	}
	var config Config
	if err := json.Unmarshal(data, &config); err != nil {
		return Config{}, fmt.Errorf("rxbrain: parse config: %w", err)
	}
	if err := config.Validate(); err != nil {
		return Config{}, err
	}
	return config, nil
}

// Validate holds the declaration to the invariants the decoder math needs.
func (c Config) Validate() error {
	unified := false
	for _, architecture := range c.Architectures {
		if architecture == "UnifiedMoTForConditionalGeneration" {
			unified = true
		}
	}
	if !unified {
		return fmt.Errorf("rxbrain: unsupported architectures %v", c.Architectures)
	}
	if c.HiddenSize <= 0 || c.NumHiddenLayers <= 0 || c.NumAttentionHeads <= 0 {
		return fmt.Errorf("rxbrain: degenerate dimensions hidden=%d layers=%d heads=%d",
			c.HiddenSize, c.NumHiddenLayers, c.NumAttentionHeads)
	}
	if c.NumKeyValueHeads <= 0 || c.NumAttentionHeads%c.NumKeyValueHeads != 0 {
		return fmt.Errorf("rxbrain: KV heads %d do not divide %d attention heads",
			c.NumKeyValueHeads, c.NumAttentionHeads)
	}
	if c.HeadDim > 0 && c.HeadDim*c.NumAttentionHeads != c.HiddenSize {
		return fmt.Errorf("rxbrain: head_dim %d x heads %d != hidden %d",
			c.HeadDim, c.NumAttentionHeads, c.HiddenSize)
	}
	if c.ClaShareFactor <= 0 || c.NumHiddenLayers%c.ClaShareFactor != 0 {
		return fmt.Errorf("rxbrain: CLA share factor %d does not divide %d layers",
			c.ClaShareFactor, c.NumHiddenLayers)
	}
	if c.VocabSize <= 0 || c.EOSTokenID <= 0 || c.EOSTokenID >= c.VocabSize {
		return fmt.Errorf("rxbrain: vocabulary %d does not contain EOS %d", c.VocabSize, c.EOSTokenID)
	}
	return c.Vision.Validate(c.HiddenSize)
}

// Validate holds the vision declaration to the tower math's invariants.
func (v VisionConfig) Validate(textHidden int) error {
	if v.HiddenSize <= 0 || v.NumHiddenLayers <= 0 || v.NumAttentionHeads <= 0 || v.IntermediateSize <= 0 {
		return fmt.Errorf("rxbrain: degenerate vision dimensions hidden=%d layers=%d heads=%d inter=%d",
			v.HiddenSize, v.NumHiddenLayers, v.NumAttentionHeads, v.IntermediateSize)
	}
	if v.HiddenSize%v.NumAttentionHeads != 0 {
		return fmt.Errorf("rxbrain: vision heads %d do not divide hidden %d", v.NumAttentionHeads, v.HiddenSize)
	}
	if v.NumChannels <= 0 || v.PatchSize <= 0 || v.TemporalPatchSize <= 0 || v.SpatialMergeSize <= 0 {
		return fmt.Errorf("rxbrain: degenerate vision patching channels=%d patch=%d temporal=%d merge=%d",
			v.NumChannels, v.PatchSize, v.TemporalPatchSize, v.SpatialMergeSize)
	}
	if v.MaxImageSize <= 0 || v.MaxImageSize%v.PatchSize != 0 {
		return fmt.Errorf("rxbrain: vision max image size %d not divisible by patch %d", v.MaxImageSize, v.PatchSize)
	}
	if v.TextHiddenSize != textHidden {
		return fmt.Errorf("rxbrain: vision text_hidden_size %d disagrees with decoder hidden %d", v.TextHiddenSize, textHidden)
	}
	return nil
}
