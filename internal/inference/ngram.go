package inference

import (
	"context"
	"errors"
	"fmt"
	"math"

	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// prepareNGramGraphInputs reads only the requested lexical rows from the authoritative GGUF.
func (r *Runner) prepareNGramGraphInputs(ctx context.Context, builder *tensor.Builder, tokens []uint32, past []float32, feeds map[*tensor.Tensor]reference.Value) (*tensor.Tensor, *tensor.Tensor, error) {
	if r.weights.NGramTokenEmbedding == nil {
		return nil, nil, errors.New("ngram embedding table is unavailable")
	}
	history := make([]uint32, len(past))
	for i, value := range past {
		if math.IsNaN(float64(value)) || value < 0 || float64(value) >= float64(r.spec.VocabularySize) || float32(uint32(value)) != value {
			return nil, nil, fmt.Errorf("ngram history value %g is not a vocabulary token", value)
		}
		history[i] = uint32(value)
	}
	rows, next, err := r.spec.NGramRows(tokens, history)
	if err != nil {
		return nil, nil, err
	}
	selected, err := model.LoadHostRows(ctx, r.file, *r.weights.NGramTokenEmbedding, rows)
	if err != nil {
		return nil, nil, err
	}
	selected.Shape = tensor.MustShape(uint64(r.spec.EmbeddingLength), uint64(len(tokens)))
	embedding := builder.Input("ngram.embedding", dtype.F32, selected.Shape)
	feeds[embedding] = selected
	values := make([]float32, len(next))
	for i, token := range next {
		values[i] = float32(token)
	}
	shape := tensor.MustShape(uint64(len(next)))
	nextHistory := builder.Input("ngram.history", dtype.F32, shape)
	feeds[nextHistory] = reference.Value{Shape: shape, Data: values}
	return embedding, nextHistory, builder.Err()
}
