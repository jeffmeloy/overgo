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

// prepareNGramGraphInputs loads lexical inputs for immediate host execution.
func (r *Runner) prepareNGramGraphInputs(ctx context.Context, builder *tensor.Builder, tokens []uint32, past []float32, feeds map[*tensor.Tensor]reference.Value) (*tensor.Tensor, *tensor.Tensor, error) {
	embedding, history, load, err := r.planNGramGraphInputs(ctx, builder, tokens, past, feeds)
	if err == nil {
		err = load()
	}
	return embedding, history, err
}

// planNGramGraphInputs prepares lexical inputs and a loader for their GGUF rows.
// The loader runs only after graph memory admission on the device path.
func (r *Runner) planNGramGraphInputs(ctx context.Context, builder *tensor.Builder, tokens []uint32, past []float32, feeds map[*tensor.Tensor]reference.Value) (*tensor.Tensor, *tensor.Tensor, func() error, error) {
	if r.weights.NGramTokenEmbedding == nil {
		return nil, nil, nil, errors.New("ngram embedding table is unavailable")
	}
	history := make([]uint32, len(past))
	for i, value := range past {
		if math.IsNaN(float64(value)) || value < 0 || float64(value) >= float64(r.spec.VocabularySize) || float32(uint32(value)) != value {
			return nil, nil, nil, fmt.Errorf("ngram history value %g is not a vocabulary token", value)
		}
		history[i] = uint32(value)
	}
	rows, next, err := r.spec.NGramRows(tokens, history)
	if err != nil {
		return nil, nil, nil, err
	}
	embedding := builder.Input("ngram.embedding", dtype.F32, tensor.MustShape(uint64(r.spec.EmbeddingLength), uint64(len(tokens))))
	load := func() error {
		selected, err := model.LoadHostRows(ctx, r.file, *r.weights.NGramTokenEmbedding, rows)
		if err != nil {
			return err
		}
		selected.Shape = embedding.Shape
		feeds[embedding] = selected
		return nil
	}
	values := make([]float32, len(next))
	for i, token := range next {
		values[i] = float32(token)
	}
	shape := tensor.MustShape(uint64(len(next)))
	nextHistory := builder.Input("ngram.history", dtype.F32, shape)
	feeds[nextHistory] = reference.Value{Shape: shape, Data: values}
	return embedding, nextHistory, load, builder.Err()
}
