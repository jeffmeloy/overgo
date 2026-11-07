package hybridtrain

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/gguf"
	"overgo/internal/hostmath"
	"overgo/internal/model"
	"overgo/internal/tensor"
)

// FrozenLanguageModel is a hybrid language model read from its GGUF file for a frozen walk: its layers in F32, every
// attention layer with the output gate its fused query projection carries, its final norm and head, and the open file
// its tokens' embedding rows are read from.
type FrozenLanguageModel struct {
	Geometry  StackConfig
	Layers    []hostmath.HybridLayerWeights
	FinalNorm []float32
	Head      []float32
	Vocab     int
	// rows reads tokens' embedding rows, and file is the model file it reads them from when the model has one.
	rows func(context.Context, []uint32) ([]float32, error)
	file *gguf.File
}

// LoadFrozenLanguageModel reads the hybrid language model at path for sequences of tokens tokens.
func LoadFrozenLanguageModel(ctx context.Context, path string, tokens int) (*FrozenLanguageModel, error) {
	file, err := gguf.Open(path)
	if err != nil {
		return nil, fmt.Errorf("hybrid stack: open artifact: %w", err)
	}
	fail := func(cause error) (*FrozenLanguageModel, error) { return nil, errors.Join(cause, file.Close()) }
	stack, weights, err := loadStack(ctx, file, tokens)
	if err != nil {
		return fail(err)
	}
	if weights.Output == nil {
		return fail(errors.New("hybrid stack: the artifact holds no head of its own"))
	}
	geometry := stack.Cfg
	// The served query projection carries each head's output gate after its query rows, [query | gate] a head.
	width := geometry.HeadDim * geometry.Hidden
	for layer, kind := range geometry.Types {
		if kind != FullAttention {
			continue
		}
		fused, err := model.LoadHostTensor(ctx, file, *weights.Layers[layer].AttentionQ)
		if err != nil {
			return fail(err)
		}
		if len(fused.Data) != tensor.PairedExtent*geometry.Heads*width {
			return fail(fmt.Errorf("hybrid stack: layer %d's query projection carries no output gate", layer))
		}
		gate := make([]float32, geometry.Heads*width)
		for head := range geometry.Heads {
			copy(gate[head*width:(head+1)*width], fused.Data[(tensor.PairedExtent*head+1)*width:(tensor.PairedExtent*head+tensor.PairedExtent)*width])
		}
		stack.Weights[layer].Attn.Gate = gate
	}
	finalNorm, err := model.LoadHostTensor(ctx, file, weights.OutputNorm)
	if err != nil {
		return fail(err)
	}
	head, err := model.LoadHostTensor(ctx, file, *weights.Output)
	if err != nil {
		return fail(err)
	}
	return &FrozenLanguageModel{
		Geometry: geometry, Layers: stack.Weights, FinalNorm: finalNorm.Data, Head: head.Data, Vocab: len(head.Data) / geometry.Hidden,
		file: file, rows: func(ctx context.Context, tokens []uint32) ([]float32, error) {
			rows, err := model.LoadHostRows(ctx, file, weights.TokenEmbedding, tokens)
			return rows.Data, err
		},
	}, nil
}

// TokenRows are the embedding rows of tokens, a row of the model's width each.
func (m *FrozenLanguageModel) TokenRows(ctx context.Context, tokens []uint32) ([]float32, error) {
	return m.rows(ctx, tokens)
}

// Close releases the model's file.
func (m *FrozenLanguageModel) Close() error {
	return m.file.Close()
}
