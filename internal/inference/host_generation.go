package inference

import (
	"context"
	"errors"
	"strings"

	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

// generateCachedHost decodes on the host from a prompt state: each step forwards
// the pending token and samples the next.
func (r *Runner) generateCachedHost(
	ctx context.Context,
	ids []tokenizer.TokenID,
	hidden reference.Value,
	cache *KVCache,
	options GenerateOptions,
	advanceFirst bool,
	keep uint32,
	discard int,
) ([]tokenizer.TokenID, reference.Value, *KVCache, error) {
	outputTable := r.outputTensor()
	var generatedText strings.Builder
	// A failed step returns the ids so far with the last good state: the cache
	// then holds every id but the pending last one.
	for generatedIndex := range options.MaxNewTokens {
		if advanceFirst || generatedIndex > 0 {
			appendable, err := r.cacheForAppendKeeping(
				cache, 1, options.ContextShift, keep, discard,
			)
			if err != nil {
				return ids, hidden, cache, err
			}
			nextHidden, next, err := r.forwardCachedLocked(
				ctx, []tokenizer.TokenID{ids[len(ids)-1]}, appendable,
			)
			if err != nil {
				return ids, hidden, cache, err
			}
			hidden, cache = nextHidden, next
		}
		event, err := r.sampleHidden(ctx, outputTable, hidden, ids, options)
		if err != nil {
			return ids, hidden, cache, err
		}
		event.Index = generatedIndex
		ids = append(ids, event.ID)
		stop, err := r.deliverGenerationToken(&event, options, &generatedText)
		if err != nil {
			return ids, hidden, cache, err
		}
		if stop {
			break
		}
	}
	return ids, hidden, cache, nil
}

func (r *Runner) outputTensor() gguf.TensorInfo {
	if r.program.Model.Terminal().OutputHead == model.OutputHeadDedicated {
		return *r.weights.Output
	}
	return r.weights.TokenEmbedding
}

func (r *Runner) outputTensorFor(override *gguf.TensorInfo) gguf.TensorInfo {
	if override != nil {
		return *override
	}
	return r.outputTensor()
}

func (r *Runner) outputNormTensorFor(overrides ...*gguf.TensorInfo) gguf.TensorInfo {
	selected := r.weights.OutputNorm
	for _, override := range overrides {
		if override != nil {
			selected = *override
		}
	}
	return selected
}

func (r *Runner) sampleHidden(
	ctx context.Context,
	outputTable gguf.TensorInfo,
	hidden reference.Value,
	ids []tokenizer.TokenID,
	options GenerateOptions,
) (TokenEvent, error) {
	last := hidden.LastRowView()
	if len(last.Data) == 0 {
		return TokenEvent{}, errors.New("inference: generation hidden state is incompatible")
	}
	logits, err := r.logits(ctx, outputTable, last.Data)
	if err != nil {
		return TokenEvent{}, err
	}
	return sampleGenerationToken(logits, ids, options)
}
