package inference

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

	"overgo/internal/model"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

// GenerateT5: source text to generated decoder tokens and resumable session.
func (r *Runner) GenerateT5(
	ctx context.Context,
	source string,
	options GenerateOptions,
) ([]tokenizer.TokenID, string, *T5Session, error) {
	if r == nil || r.vocab == nil {
		return nil, "", nil, errors.New("inference: runner is nil")
	}
	if r.forwardPolicy() != model.ForwardT5 {
		return nil, "", nil, errors.New("inference: T5 generation requires T5 architecture")
	}
	if err := normalizeGenerateOptions(&options); err != nil {
		return nil, "", nil, err
	}
	if options.ProjectedInputs != nil {
		return nil, "", nil, errors.New("inference: T5 generation does not accept projected decoder inputs")
	}
	sourceIDs, err := r.promptTokenIDs(source, options)
	if err != nil {
		return nil, "", nil, err
	}
	startID := tokenizer.TokenID(r.spec.DecoderStartTokenID)
	if startID < 0 || int(startID) >= r.vocab.Len() {
		return nil, "", nil, errors.New("inference: T5 decoder start token is out of range")
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil, "", nil, errors.New("inference: runner is closed")
	}
	restoreLoRA, err := r.applyGenerationLoRA(options)
	if err != nil {
		return nil, "", nil, err
	}
	defer restoreLoRA()

	promptStarted := time.Now()
	var encoder reference.Value
	cached := 0
	if options.CachePrompt {
		if selected, reused := r.selectT5SourceCache(sourceIDs, options.MinCacheReuse); selected != nil {
			encoder = selected.Hidden
			cached = reused
		}
	}
	if cached == 0 {
		encoder, err = r.forwardT5EncoderLocked(ctx, sourceIDs)
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
	session := &T5Session{Encoder: encoder}
	if options.MaxNewTokens == 0 {
		return []tokenizer.TokenID{}, "", session, nil
	}

	history := []tokenizer.TokenID{startID}
	pending := startID
	generated := make([]tokenizer.TokenID, 0, options.MaxNewTokens)
	var generatedText strings.Builder
	keepTokens := effectiveKeepTokens(options.KeepTokens, 1, r.spec.ContextLength)
	for generatedIndex := range options.MaxNewTokens {
		if session.Cache != nil {
			shifted, shiftErr := r.cacheForAppendKeeping(
				session.Cache, 1, options.ContextShift, keepTokens, options.DiscardTokens,
			)
			if shiftErr != nil {
				return nil, "", nil, shiftErr
			}
			session = &T5Session{Encoder: session.Encoder, Cache: shifted}
		}
		logitRows, nextSession, decodeErr := r.decodeT5Locked(
			ctx, session, []tokenizer.TokenID{pending},
		)
		if decodeErr != nil {
			return nil, "", nil, decodeErr
		}
		session = nextSession
		width := int(logitRows.Shape.Dims[0])
		if width != r.vocab.Len() || len(logitRows.Data) < width {
			return nil, "", nil, errors.New("inference: T5 logits shape is incompatible")
		}
		logits := logitRows.Data[len(logitRows.Data)-width:]
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
		session.Cache, 1, options.ContextShift, keepTokens, options.DiscardTokens,
	)
	if shiftErr != nil {
		return nil, "", nil, shiftErr
	}
	session = &T5Session{Encoder: session.Encoder, Cache: shifted}
	_, session, err = r.decodeT5Locked(ctx, session, []tokenizer.TokenID{pending})
	if err != nil {
		return nil, "", nil, err
	}
	text, err := r.vocab.Decode(generated, false)
	if err != nil {
		return nil, "", nil, fmt.Errorf("inference: decode T5 output: %w", err)
	}
	return generated, text, session, nil
}
