package inference

import (
	"context"
	"errors"
	"fmt"
	"math"

	"overgo/internal/cuda/executor"
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
	if r.weights.NGramTokenEmbedding == nil || r.spec.NGram == nil || r.spec.NGram.Size == 0 {
		return nil, nil, nil, errors.New("ngram embedding table is unavailable")
	}
	embedding := builder.Input("ngram.embedding", dtype.F32, tensor.MustShape(uint64(r.spec.EmbeddingLength), uint64(len(tokens))))
	nextHistory := builder.Input("ngram.history", dtype.F32, tensor.MustShape(uint64(r.spec.NGram.Size-1)))
	if err := builder.Err(); err != nil {
		return nil, nil, nil, err
	}
	load, err := r.feedNGramInputs(ctx, embedding, nextHistory, tokens, past, feeds)
	if err != nil {
		return nil, nil, nil, err
	}
	return embedding, nextHistory, load, nil
}

// feedNGramInputs writes a step's next lexical history into feeds and returns the loader of its GGUF rows.
//   - the rows and the next history follow the step's tokens and the past's history, so a replayed step feeds them again
func (r *Runner) feedNGramInputs(ctx context.Context, embedding, nextHistory *tensor.Tensor, tokens []uint32, past []float32, feeds map[*tensor.Tensor]reference.Value) (func() error, error) {
	history := make([]uint32, len(past))
	for i, value := range past {
		if math.IsNaN(float64(value)) || value < 0 || float64(value) >= float64(r.spec.VocabularySize) || float32(uint32(value)) != value {
			return nil, fmt.Errorf("ngram history value %g is not a vocabulary token", value)
		}
		history[i] = uint32(value)
	}
	rows, next, err := r.spec.NGramRows(tokens, history)
	if err != nil {
		return nil, err
	}
	if uint64(len(next)) != nextHistory.Shape.Dims[0] || embedding.Shape.Dims[1] != uint64(len(tokens)) {
		return nil, fmt.Errorf("ngram step of %d tokens and %d history tokens differs from inputs %v and %v", len(tokens), len(next), embedding.Shape, nextHistory.Shape)
	}
	values := make([]float32, len(next))
	for i, token := range next {
		values[i] = float32(token)
	}
	feeds[nextHistory] = reference.Value{Shape: nextHistory.Shape, Data: values}
	return func() error {
		// A request holds the runner, so the reader is built once.
		if r.ngramRows == nil {
			reader, err := model.NewHostRows(r.file, *r.weights.NGramTokenEmbedding, model.NGramCacheRows)
			if err != nil {
				return err
			}
			r.ngramRows = reader
		}
		selected, err := r.ngramRows.Load(ctx, rows)
		if err != nil {
			return err
		}
		selected.Shape = embedding.Shape
		feeds[embedding] = selected
		return nil
	}, nil
}

// deviceNGramHistory reads the lexical history a device past carries, empty for a forward without a past.
//   - a past a device step made carries the history the step fed, so a decode step waits for no copy from the card
func (r *Runner) deviceNGramHistory(ctx context.Context, past *deviceKVCache) ([]float32, error) {
	if past == nil {
		return nil, nil
	}
	if past.ngramHistory != nil {
		return past.ngramHistory, nil
	}
	layer := int(r.spec.NGram.Layer)
	if layer >= len(past.States) {
		return nil, errors.New("ngram history layer is missing")
	}
	state, ok := past.States[layer][model.CacheStateNGramHistory]
	if !ok {
		return nil, errors.New("ngram token history is missing")
	}
	values, err := r.cuda.ReadDeviceValues(ctx, []executor.DeviceValue{state.Value})
	if err != nil {
		return nil, err
	}
	return values[0], nil
}
