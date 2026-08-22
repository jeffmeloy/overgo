package inference

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

	"overgo/internal/checked"
	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

// GenerateEncoderDecoder: source text to decoder tokens and resumable session.
func (r *Runner) GenerateEncoderDecoder(
	ctx context.Context,
	source string,
	options GenerateOptions,
) ([]tokenizer.TokenID, string, *EncoderDecoderSession, error) {
	if r == nil || r.vocab == nil {
		return nil, "", nil, errRunnerNil
	}
	if r.forwardProgram().Session != model.ForwardSessionEncoderDecoder {
		return nil, "", nil, errors.New("inference: generation requires a compiled encoder-decoder program")
	}
	if err := normalizeGenerateOptions(&options); err != nil {
		return nil, "", nil, err
	}
	if options.ProjectedInputs != nil {
		return nil, "", nil, errors.New("inference: encoder-decoder generation does not accept projected decoder inputs")
	}
	sourceIDs, err := r.promptTokenIDs(source, options)
	if err != nil {
		return nil, "", nil, err
	}
	startID := tokenizer.TokenID(r.spec.DecoderStartTokenID)
	if !checked.NonNegativeInts(int(startID)) || int(startID) >= r.vocab.Len() {
		return nil, "", nil, errors.New("inference: decoder start token is out of range")
	}

	if err := r.lockOpen(); err != nil {
		return nil, "", nil, err
	}
	defer r.mu.Unlock()
	restoreLoRA, err := r.applyGenerationLoRA(options)
	if err != nil {
		return nil, "", nil, err
	}
	defer restoreLoRA()

	promptStarted := time.Now()
	var encoder reference.Value
	var cached int
	if options.CachePrompt {
		if selected, reused := r.selectEncoderSourceCache(sourceIDs, options.MinCacheReuse); selected != nil {
			encoder = selected.Hidden
			cached = reused
		}
	}
	if !checked.Nonzero(cached) {
		encoder, err = r.forwardEncoderLocked(ctx, sourceIDs)
		if err != nil {
			return nil, "", nil, err
		}
		if options.CachePrompt {
			if err := r.storePromptCache(ctx, &cachedPrompt{
				Tokens:        slices.Clone(sourceIDs),
				Hidden:        encoder,
				LoRASignature: r.currentLoRASignature(),
			}); err != nil {
				return nil, "", nil, err
			}
		}
	}
	if options.OnPromptEvaluated != nil {
		options.OnPromptEvaluated(PromptEvaluation{
			Tokens: len(sourceIDs), Cached: cached, Duration: time.Since(promptStarted),
		})
	}
	session := &EncoderDecoderSession{Encoder: encoder}
	if !checked.Nonzero(options.MaxNewTokens) {
		return []tokenizer.TokenID{}, "", session, nil
	}

	history := []tokenizer.TokenID{startID}
	pending := startID
	var generated []tokenizer.TokenID
	var generatedText strings.Builder
	keepTokens := effectiveKeepTokens(options.KeepTokens, tensor.SingletonExtent, r.spec.ContextLength)
	for generatedIndex := range options.MaxNewTokens {
		if session.Cache != nil {
			shifted, shiftErr := r.cacheForAppendKeeping(
				session.Cache, tensor.SingletonExtent, options.ContextShift, keepTokens, options.DiscardTokens,
			)
			if shiftErr != nil {
				return nil, "", nil, shiftErr
			}
			session = &EncoderDecoderSession{Encoder: session.Encoder, Cache: shifted}
		}
		logitRows, nextSession, decodeErr := r.decodeEncoderDecoderLocked(
			ctx, session, []tokenizer.TokenID{pending},
		)
		if decodeErr != nil {
			return nil, "", nil, decodeErr
		}
		session = nextSession
		logits := logitRows.LastRowView().Data
		if len(logits) != r.vocab.Len() {
			return nil, "", nil, errors.New("inference: decoder logits shape is incompatible")
		}
		event, sampleErr := sampleGenerationToken(logits, history, options)
		if sampleErr != nil {
			return nil, "", nil, sampleErr
		}
		event.Index = generatedIndex
		generated = append(generated, event.ID)
		history = append(history, event.ID)
		pending = event.ID
		stop, deliverErr := r.deliverGenerationToken(&event, options, &generatedText)
		if deliverErr != nil {
			return nil, "", nil, deliverErr
		}
		if stop {
			break
		}
	}
	shifted, shiftErr := r.cacheForAppendKeeping(
		session.Cache, tensor.SingletonExtent, options.ContextShift, keepTokens, options.DiscardTokens,
	)
	if shiftErr != nil {
		return nil, "", nil, shiftErr
	}
	session = &EncoderDecoderSession{Encoder: session.Encoder, Cache: shifted}
	_, session, err = r.decodeEncoderDecoderLocked(ctx, session, []tokenizer.TokenID{pending})
	if err != nil {
		return nil, "", nil, err
	}
	text, err := r.vocab.Decode(generated, false)
	if err != nil {
		return nil, "", nil, fmt.Errorf("inference: decode output: %w", err)
	}
	return generated, text, session, nil
}
