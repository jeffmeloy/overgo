package rxbrain

import (
	"fmt"

	"overgo/internal/safetensors"
)

// Stage 3b: the SigLIP-style visual tower. This file binds the tower's
// weights from the sharded checkpoint under full shape validation --
// LayerNorm blocks with packed QKV and biases, an interpolatable absolute
// position table, and the 2x2 spatial merger that projects patch states
// into the decoder's text stream. The encode math lands behind this
// binding in the next increment.

// VisionBlockWeights is one tower block: LayerNorm (weight+bias) pairs
// around packed-QKV attention and a GELU MLP, all projections biased.
type VisionBlockWeights struct {
	Norm1W, Norm1B []float32
	QKVW, QKVB     []float32
	ProjW, ProjB   []float32
	Norm2W, Norm2B []float32
	FC1W, FC1B     []float32
	FC2W, FC2B     []float32
}

// VisionWeights binds the whole tower: patch embedding, position table,
// depth blocks, and the merger whose output dimension is the decoder's
// hidden size (the declared out_hidden_size is the fused 2x pooler input,
// not the projection target -- the checkpoint's own merger shapes decide).
type VisionWeights struct {
	PatchEmbedW, PatchEmbedB   []float32
	PosEmbed                   []float32
	Blocks                     []VisionBlockWeights
	MergerProj1W, MergerProj1B []float32
	MergerProj2W, MergerProj2B []float32
	MergerPool0W, MergerPool0B []float32
	MergerPool2W, MergerPool2B []float32
}

// LoadVisionWeights binds the visual tower from the sharded checkpoint,
// validating every tensor's element count against the declared vision
// configuration. A missing tensor or a drifted shape refuses by name.
func LoadVisionWeights(directory string, config Config) (VisionWeights, error) {
	source, err := safetensors.OpenSource(directory)
	if err != nil {
		return VisionWeights{}, fmt.Errorf("rxbrain: open shards: %w", err)
	}
	defer source.Close()

	read := func(name string, wantRows, wantCols int) ([]float32, error) {
		tensor, ok := source.Tensors[name]
		if !ok {
			return nil, fmt.Errorf("rxbrain: tensor %s absent", name)
		}
		elements := uint64(1)
		for _, dimension := range tensor.Shape {
			elements *= dimension
		}
		want := uint64(wantRows)
		if wantCols > 0 {
			want *= uint64(wantCols)
		}
		if elements != want {
			return nil, fmt.Errorf("rxbrain: tensor %s shape %v does not hold %d x %d", name, tensor.Shape, wantRows, wantCols)
		}
		values, err := safetensors.ReadF32(tensor)
		if err != nil {
			return nil, fmt.Errorf("rxbrain: read %s: %w", name, err)
		}
		return values, nil
	}

	vision := config.Vision
	hidden, inter, out := vision.HiddenSize, vision.IntermediateSize, vision.TextHiddenSize
	side := vision.PositionSide()
	weights := VisionWeights{Blocks: make([]VisionBlockWeights, vision.NumHiddenLayers)}
	fixed := []struct {
		target *[]float32
		name   string
		rows   int
		cols   int
	}{
		{&weights.PatchEmbedW, "model.visual.vision_tower.patch_embed.proj.weight", hidden, vision.NumChannels * vision.PatchSize * vision.PatchSize},
		{&weights.PatchEmbedB, "model.visual.vision_tower.patch_embed.proj.bias", hidden, 0},
		{&weights.PosEmbed, "model.visual.vision_tower.pos_embed", side * side, hidden},
		{&weights.MergerProj1W, "model.visual.merger.proj1.weight", out, hidden},
		{&weights.MergerProj1B, "model.visual.merger.proj1.bias", out, 0},
		{&weights.MergerProj2W, "model.visual.merger.proj2.weight", out, out},
		{&weights.MergerProj2B, "model.visual.merger.proj2.bias", out, 0},
		{&weights.MergerPool0W, "model.visual.merger.pooler.predictor.0.weight", out, 2 * out},
		{&weights.MergerPool0B, "model.visual.merger.pooler.predictor.0.bias", out, 0},
		{&weights.MergerPool2W, "model.visual.merger.pooler.predictor.2.weight", out, out},
		{&weights.MergerPool2B, "model.visual.merger.pooler.predictor.2.bias", out, 0},
	}
	for _, spec := range fixed {
		if *spec.target, err = read(spec.name, spec.rows, spec.cols); err != nil {
			return VisionWeights{}, err
		}
	}
	for layer := range weights.Blocks {
		prefix := fmt.Sprintf("model.visual.vision_tower.blocks.%d.", layer)
		bind := &weights.Blocks[layer]
		specs := []struct {
			target *[]float32
			name   string
			rows   int
			cols   int
		}{
			{&bind.Norm1W, prefix + "norm1.weight", hidden, 0},
			{&bind.Norm1B, prefix + "norm1.bias", hidden, 0},
			{&bind.QKVW, prefix + "attn.qkv.weight", 3 * hidden, hidden},
			{&bind.QKVB, prefix + "attn.qkv.bias", 3 * hidden, 0},
			{&bind.ProjW, prefix + "attn.proj.weight", hidden, hidden},
			{&bind.ProjB, prefix + "attn.proj.bias", hidden, 0},
			{&bind.Norm2W, prefix + "norm2.weight", hidden, 0},
			{&bind.Norm2B, prefix + "norm2.bias", hidden, 0},
			{&bind.FC1W, prefix + "mlp.fc1.weight", inter, hidden},
			{&bind.FC1B, prefix + "mlp.fc1.bias", inter, 0},
			{&bind.FC2W, prefix + "mlp.fc2.weight", hidden, inter},
			{&bind.FC2B, prefix + "mlp.fc2.bias", hidden, 0},
		}
		for _, spec := range specs {
			if *spec.target, err = read(spec.name, spec.rows, spec.cols); err != nil {
				return VisionWeights{}, err
			}
		}
	}
	return weights, nil
}
