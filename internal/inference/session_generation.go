package inference

import (
	"context"
	"errors"
	"slices"
	"strings"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
)

// StartSession: tokenizes prompt and generates at least one token, returning
// state that can be serialized or passed directly to ContinueSession
func (r *Runner) StartSession(
	ctx context.Context,
	prompt string,
	options GenerateOptions,
) (*Session, string, error) {
	if r == nil || r.vocab == nil {
		return nil, "", errors.New("inference: runner is nil")
	}
	if options.MaxNewTokens <= 0 {
		return nil, "", errors.New("inference: resumable generation needs at least one new token")
	}
	if err := normalizeGenerateOptions(&options); err != nil {
		return nil, "", err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil, "", errors.New("inference: runner is closed")
	}

	ids, err := r.promptTokenIDs(prompt, options)
	if err != nil {
		return nil, "", err
	}
	hidden, cache, err := r.forwardCachedLocked(ctx, ids, nil)
	if err != nil {
		return nil, "", err
	}
	outputTable := r.outputTensor()
	var generatedText strings.Builder
	for generatedIndex := range options.MaxNewTokens {
		if generatedIndex > 0 {
			cache, err = r.cacheForAppend(cache, 1, options.ContextShift)
			if err != nil {
				return nil, "", err
			}
			hidden, cache, err = r.forwardCachedLocked(
				ctx,
				[]tokenizer.TokenID{ids[len(ids)-1]},
				cache,
			)
			if err != nil {
				return nil, "", err
			}
		}
		event, sampleErr := r.sampleHidden(ctx, outputTable, hidden, ids, options)
		if sampleErr != nil {
			return nil, "", sampleErr
		}
		event.Index = generatedIndex
		ids = append(ids, event.ID)
		stop, deliverErr := r.deliverGenerationToken(&event, options, &generatedText)
		if deliverErr != nil {
			return nil, "", deliverErr
		}
		if stop {
			break
		}
	}
	session := &Session{TokenIDs: ids, Cache: cache}
	if err := r.validateSession(session); err != nil {
		return nil, "", err
	}
	text, err := r.vocab.Decode(ids, false)
	if err != nil {
		return nil, "", err
	}
	return session, text, nil
}

// ContinueSession appends tokens to valid resumable session; input state
// remains usable if generation fails; returned token history is fresh slice
func (r *Runner) ContinueSession(
	ctx context.Context,
	session *Session,
	options GenerateOptions,
) (*Session, string, error) {
	if r == nil || r.vocab == nil {
		return nil, "", errors.New("inference: runner is nil")
	}
	if options.MaxNewTokens < 0 {
		return nil, "", errors.New("inference: max new tokens is negative")
	}
	if err := normalizeGenerateOptions(&options); err != nil {
		return nil, "", err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil, "", errors.New("inference: runner is closed")
	}
	if err := r.validateSession(session); err != nil {
		return nil, "", err
	}
	ids := slices.Clone(session.TokenIDs)
	cache := session.Cache
	if options.MaxNewTokens > 0 && !r.isTerminal(ids[len(ids)-1]) {
		outputTable := r.outputTensor()
		var generatedText strings.Builder
		for generatedIndex := range options.MaxNewTokens {
			shiftedCache, shiftErr := r.cacheForAppend(cache, 1, options.ContextShift)
			if shiftErr != nil {
				return nil, "", shiftErr
			}
			cache = shiftedCache
			hidden, nextCache, err := r.forwardCachedLocked(
				ctx,
				[]tokenizer.TokenID{ids[len(ids)-1]},
				cache,
			)
			if err != nil {
				return nil, "", err
			}
			event, err := r.sampleHidden(ctx, outputTable, hidden, ids, options)
			if err != nil {
				return nil, "", err
			}
			cache = nextCache
			event.Index = generatedIndex
			ids = append(ids, event.ID)
			stop, deliverErr := r.deliverGenerationToken(&event, options, &generatedText)
			if deliverErr != nil {
				return nil, "", deliverErr
			}
			if stop {
				break
			}
		}
	}
	result := &Session{TokenIDs: ids, Cache: cache}
	if err := r.validateSession(result); err != nil {
		return nil, "", err
	}
	text, err := r.vocab.Decode(ids, false)
	if err != nil {
		return nil, "", err
	}
	return result, text, nil
}

func (r *Runner) outputTensor() gguf.TensorInfo {
	if r.weights.Output != nil {
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
	width := int(hidden.Shape.Dims[0])
	last := hidden.Data[len(hidden.Data)-width:]
	logits, err := r.logits(ctx, outputTable, last)
	if err != nil {
		return TokenEvent{}, err
	}
	return sampleGenerationToken(logits, ids, options)
}

func (r *Runner) isTerminal(id tokenizer.TokenID) bool {
	return r.vocab.IsEOG(id)
}
