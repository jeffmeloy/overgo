package inference

import (
	"context"
	"errors"
	"strings"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/sampling"
	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
)

// StartSession tokenizes a prompt and generates at least one token, returning
// state that can be serialized or passed directly to ContinueSession.
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
	if err := validateStopSequences(options.StopSequences); err != nil {
		return nil, "", err
	}
	if options.Sampler == nil {
		var err error
		options.Sampler, err = sampling.New(sampling.Config{})
		if err != nil {
			return nil, "", err
		}
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
		nextID, sampleErr := r.sampleHidden(ctx, outputTable, hidden, ids, options.Sampler)
		if sampleErr != nil {
			return nil, "", sampleErr
		}
		ids = append(ids, nextID)
		if err := r.reportToken(options.OnToken, nextID, generatedIndex); err != nil {
			return nil, "", err
		}
		if len(options.StopSequences) > 0 {
			piece, decodeErr := r.vocab.DecodePiece(nextID, false)
			if decodeErr != nil {
				return nil, "", decodeErr
			}
			generatedText.WriteString(piece)
		}
		if r.isTerminal(nextID) ||
			matchesStopSequence(generatedText.String(), options.StopSequences) {
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

// ContinueSession appends tokens to a valid resumable session. The input state
// remains usable if generation fails; returned token history is a fresh slice.
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
	if err := validateStopSequences(options.StopSequences); err != nil {
		return nil, "", err
	}
	if options.Sampler == nil {
		var err error
		options.Sampler, err = sampling.New(sampling.Config{})
		if err != nil {
			return nil, "", err
		}
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil, "", errors.New("inference: runner is closed")
	}
	if err := r.validateSession(session); err != nil {
		return nil, "", err
	}
	ids := append([]tokenizer.TokenID(nil), session.TokenIDs...)
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
			nextID, err := r.sampleHidden(ctx, outputTable, hidden, ids, options.Sampler)
			if err != nil {
				return nil, "", err
			}
			cache = nextCache
			ids = append(ids, nextID)
			if err := r.reportToken(options.OnToken, nextID, generatedIndex); err != nil {
				return nil, "", err
			}
			if len(options.StopSequences) > 0 {
				piece, decodeErr := r.vocab.DecodePiece(nextID, false)
				if decodeErr != nil {
					return nil, "", decodeErr
				}
				generatedText.WriteString(piece)
			}
			if r.isTerminal(nextID) ||
				matchesStopSequence(generatedText.String(), options.StopSequences) {
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

func (r *Runner) sampleHidden(
	ctx context.Context,
	outputTable gguf.TensorInfo,
	hidden reference.Value,
	ids []tokenizer.TokenID,
	sampler *sampling.Sampler,
) (tokenizer.TokenID, error) {
	width := int(hidden.Shape.Dims[0])
	last := hidden.Data[len(hidden.Data)-width:]
	logits, err := r.logits(ctx, outputTable, last)
	if err != nil {
		return 0, err
	}
	history := make([]int, len(ids))
	for index, id := range ids {
		history[index] = int(id)
	}
	next, err := sampler.SampleWithHistory(logits, history)
	return tokenizer.TokenID(next), err
}

func (r *Runner) reportToken(
	callback func(TokenEvent) error,
	id tokenizer.TokenID,
	index int,
) error {
	if callback == nil {
		return nil
	}
	piece, err := r.vocab.DecodePiece(id, false)
	if err != nil {
		return err
	}
	return callback(TokenEvent{ID: id, Piece: piece, Index: index})
}

func (r *Runner) isTerminal(id tokenizer.TokenID) bool {
	return r.vocab.IsEOG(id)
}
