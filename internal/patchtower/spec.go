// Package patchtower is the host (CPU, bf16-rounded float32) reference path
// for a merged-patch vision tower with an interpolated 2D position table:
// conv patch embed + bilinear pos-embed sampling, pre-norm attention blocks
// (LayerNorm affine, fused QKV with bias, full bidirectional attention,
// erf-GELU MLP), and a 4-matrix score-pooled merger. Every dimension comes
// from config.json / preprocessor_config.json / tensor shapes.
//
// Ported from adaptive_new go/extmodel merged_patch_vision.go (host arithmetic)
// and merged_patch_vision_cuda_windows.go (block interior op order).
package patchtower

import (
	"fmt"
	"path/filepath"

	"overgo/internal/jsonfile"
	"overgo/internal/media"
)

// LayerNormEps: torch nn.LayerNorm default eps; the checkpoint's
// vision_config carries no layer_norm_eps key, so the reference model's
// module default governs (adaptive MergedPatchVisionLayerNormEps).
const LayerNormEps = 1e-6

// Spec: tower geometry, all derived from the checkpoint configs.
type Spec struct {
	Hidden        int // vision_config.hidden_size
	Depth         int // vision_config.num_hidden_layers
	Heads         int // vision_config.num_attention_heads
	PatchSize     int // vision_config.patch_size
	PatchDim      int // channels * temporal_patch * patch * patch
	MergeSize     int // preprocessor merge_size
	PositionSide  int // vision_config.max_image_size / patch_size
	OutHidden     int // language hidden_size (merger output width)
	BlockMLPInter int // vision_config.intermediate_size
}

type visionConfigJSON struct {
	HiddenSize        int `json:"hidden_size"`
	IntermediateSize  int `json:"intermediate_size"`
	NumHiddenLayers   int `json:"num_hidden_layers"`
	NumAttentionHeads int `json:"num_attention_heads"`
	PatchSize         int `json:"patch_size"`
	MaxImageSize      int `json:"max_image_size"`
}

type modelConfigJSON struct {
	HiddenSize   int              `json:"hidden_size"`
	VisionConfig visionConfigJSON `json:"vision_config"`
}

// LoadSpec: reads config.json + preprocessor_config.json under modelDir.
func LoadSpec(modelDir string) (Spec, error) {
	var cfg modelConfigJSON
	if err := jsonfile.Decode(filepath.Join(modelDir, "config.json"), &cfg); err != nil {
		return Spec{}, fmt.Errorf("patch tower config: %w", err)
	}
	pre, err := LoadPreprocessConfig(modelDir)
	if err != nil {
		return Spec{}, err
	}
	v := cfg.VisionConfig
	if v.MaxImageSize <= 0 || v.PatchSize <= 0 || v.MaxImageSize%v.PatchSize != 0 {
		return Spec{}, fmt.Errorf("patch tower position grid: max image size %d not divisible by patch size %d", v.MaxImageSize, v.PatchSize)
	}
	spec := Spec{
		Hidden:        v.HiddenSize,
		Depth:         v.NumHiddenLayers,
		Heads:         v.NumAttentionHeads,
		PatchSize:     v.PatchSize,
		PatchDim:      media.RGBChannels * pre.TemporalPatchSize * pre.PatchSize * pre.PatchSize,
		MergeSize:     pre.MergeSize,
		PositionSide:  v.MaxImageSize / v.PatchSize,
		OutHidden:     cfg.HiddenSize,
		BlockMLPInter: v.IntermediateSize,
	}
	if spec.Hidden <= 0 || spec.Depth <= 0 || spec.Heads <= 0 || spec.PatchSize <= 0 ||
		spec.PatchDim <= 0 || spec.MergeSize <= 0 || spec.PositionSide <= 0 ||
		spec.OutHidden <= 0 || spec.BlockMLPInter <= 0 {
		return Spec{}, fmt.Errorf("patch tower spec incomplete: %+v", spec)
	}
	if spec.Hidden%spec.Heads != 0 {
		return Spec{}, fmt.Errorf("patch tower spec: hidden %d not divisible by heads %d", spec.Hidden, spec.Heads)
	}
	return spec, nil
}
