package inference

import (
	"context"
	"errors"
	"slices"
	"strings"

	"overgo/internal/checked"
	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

// StartSession: tokenizes prompt and generates at least one token, returning
// state that can be serialized or passed directly to ContinueSession
func (r *Runner) StartSession(
	ctx context.Context,
	prompt string,
	options GenerateOptions,
) (*Session, string, error) {
	if r == nil || r.vocab == nil {
		return nil, "", errRunnerNil
	}
	if !checked.PositiveInts(options.MaxNewTokens) {
		return nil, "", errors.New("inference: resumable generation needs at least one new token")
	}
	if err := normalizeGenerateOptions(&options); err != nil {
		return nil, "", err
	}
	if err := r.lockOpen(); err != nil {
		return nil, "", err
	}
	defer r.mu.Unlock()

	ids, err := r.promptTokenIDs(prompt, options)
	if err != nil {
		return nil, "", err
	}
	hidden, cache, err := r.forwardCachedLocked(ctx, ids, nil)
	if err != nil {
		return nil, "", err
	}
	keepTokens := effectiveKeepTokens(options.KeepTokens, len(ids), r.spec.ContextLength)
	ids, cache, err = r.generateCachedHost(
		ctx, ids, hidden, cache, options, false, keepTokens, options.DiscardTokens,
	)
	if err != nil {
		return nil, "", err
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
		return nil, "", errRunnerNil
	}
	if !checked.NonNegativeInts(options.MaxNewTokens) {
		return nil, "", errors.New("inference: max new tokens is negative")
	}
	if err := normalizeGenerateOptions(&options); err != nil {
		return nil, "", err
	}
	if err := r.lockOpen(); err != nil {
		return nil, "", err
	}
	defer r.mu.Unlock()
	if err := r.validateSession(session); err != nil {
		return nil, "", err
	}
	ids := slices.Clone(session.TokenIDs)
	cache := session.Cache
	lastIDs, _ := checked.LastSlice(ids)
	lastID, _ := checked.First(lastIDs)
	if checked.PositiveInts(options.MaxNewTokens) && !r.vocab.IsEOG(lastID) {
		var err error
		keepTokens := effectiveKeepTokens(options.KeepTokens, len(ids), r.spec.ContextLength)
		ids, cache, err = r.generateCachedHost(
			ctx, ids, reference.Value{}, cache, options, true, keepTokens, options.DiscardTokens,
		)
		if err != nil {
			return nil, "", err
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

func (r *Runner) generateCachedHost(
	ctx context.Context,
	ids []tokenizer.TokenID,
	hidden reference.Value,
	cache *KVCache,
	options GenerateOptions,
	advanceFirst bool,
	keep uint32,
	discard int,
) ([]tokenizer.TokenID, *KVCache, error) {
	outputTable := r.outputTensor()
	var generatedText strings.Builder
	for generatedIndex := range options.MaxNewTokens {
		if advanceFirst || checked.Nonzero(generatedIndex) {
			var err error
			cache, err = r.cacheForAppendKeeping(
				cache, tensor.SingletonExtent, options.ContextShift, keep, discard,
			)
			if err != nil {
				return nil, nil, err
			}
			lastIDs, _ := checked.LastSlice(ids)
			hidden, cache, err = r.forwardCachedLocked(ctx, lastIDs, cache)
			if err != nil {
				return nil, nil, err
			}
		}
		event, err := r.sampleHidden(ctx, outputTable, hidden, ids, options)
		if err != nil {
			return nil, nil, err
		}
		event.Index = generatedIndex
		ids = append(ids, event.ID)
		stop, err := r.deliverGenerationToken(&event, options, &generatedText)
		if err != nil {
			return nil, nil, err
		}
		if stop {
			break
		}
	}
	return ids, cache, nil
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
	if !last.Defined() {
		return TokenEvent{}, errors.New("inference: generation hidden state is incompatible")
	}
	logits, err := r.logits(ctx, outputTable, last.Data)
	if err != nil {
		return TokenEvent{}, err
	}
	return sampleGenerationToken(logits, ids, options)
}
