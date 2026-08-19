package rxbrain

import (
	"fmt"

	"overgo/internal/safetensors"
)

// LayerWeights holds one decoder layer's base-text branch plus the shared
// query/key layernorms. The vision branch loads in stage 3b; the generation
// branch never loads in the vision-QA scope.
type LayerWeights struct {
	InputLN, PostLN            []float32
	QProj, KProj, VProj, OProj []float32
	QueryLN, KeyLN             []float32
	GateProj, UpProj, DownProj []float32
}

// TextWeights is the text-only decode path: token embedding (tied output
// head -- the checkpoint carries no separate lm_head), final norm, and the
// per-layer base branch.
type TextWeights struct {
	Embed     []float32
	FinalNorm []float32
	Layers    []LayerWeights
}

// LoadTextWeights binds the base-text branch from the sharded checkpoint,
// validating every tensor's shape against the declared configuration before
// promotion. A missing tensor or a drifted shape refuses by name.
func LoadTextWeights(directory string, config Config) (TextWeights, error) {
	source, err := safetensors.OpenSource(directory)
	if err != nil {
		return TextWeights{}, fmt.Errorf("rxbrain: open shards: %w", err)
	}
	defer source.Close()

	kvDim := config.NumKeyValueHeads * config.HeadDim
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

	weights := TextWeights{Layers: make([]LayerWeights, config.NumHiddenLayers)}
	if weights.Embed, err = read("model.language_model.model.embed_tokens.weight", config.VocabSize, config.HiddenSize); err != nil {
		return TextWeights{}, err
	}
	if weights.FinalNorm, err = read("model.language_model.model.norm.weight", config.HiddenSize, 0); err != nil {
		return TextWeights{}, err
	}
	for layer := range weights.Layers {
		prefix := fmt.Sprintf("model.language_model.model.layers.%d.", layer)
		bind := &weights.Layers[layer]
		specs := []struct {
			target *[]float32
			name   string
			rows   int
			cols   int
		}{
			{&bind.InputLN, prefix + "input_layernorm.weight", config.HiddenSize, 0},
			{&bind.PostLN, prefix + "post_attention_layernorm.weight", config.HiddenSize, 0},
			{&bind.QProj, prefix + "self_attn.q_proj.weight", config.HiddenSize, config.HiddenSize},
			{&bind.KProj, prefix + "self_attn.k_proj.weight", kvDim, config.HiddenSize},
			{&bind.VProj, prefix + "self_attn.v_proj.weight", kvDim, config.HiddenSize},
			{&bind.OProj, prefix + "self_attn.o_proj.weight", config.HiddenSize, config.HiddenSize},
			{&bind.QueryLN, prefix + "self_attn.query_layernorm.weight", config.HeadDim, 0},
			{&bind.KeyLN, prefix + "self_attn.key_layernorm.weight", config.HeadDim, 0},
			{&bind.GateProj, prefix + "mlp.gate_proj.weight", config.IntermediateSize, config.HiddenSize},
			{&bind.UpProj, prefix + "mlp.up_proj.weight", config.IntermediateSize, config.HiddenSize},
			{&bind.DownProj, prefix + "mlp.down_proj.weight", config.HiddenSize, config.IntermediateSize},
		}
		for _, spec := range specs {
			if *spec.target, err = read(spec.name, spec.rows, spec.cols); err != nil {
				return TextWeights{}, err
			}
		}
	}
	return weights, nil
}
