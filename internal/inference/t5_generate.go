package inference

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"llamacpp2go/internal/sampling"
	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
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
	if r.spec.Architecture != "t5" {
		return nil, "", nil, errors.New("inference: T5 generation requires T5 architecture")
	}
	if options.MaxNewTokens < 0 {
		return nil, "", nil, errors.New("inference: max new tokens is negative")
	}
	if options.MinCacheReuse < 0 {
		return nil, "", nil, errors.New("inference: minimum cache reuse is negative")
	}
	if options.KeepTokens < -1 {
		return nil, "", nil, errors.New("inference: keep token count must be at least -1")
	}
	if options.DiscardTokens < 0 {
		return nil, "", nil, errors.New("inference: discard token count is negative")
	}
	if options.PostSamplingProbabilities < 0 {
		return nil, "", nil, errors.New("inference: post-sampling probability count is negative")
	}
	if options.ProjectedInputs != nil {
		return nil, "", nil, errors.New("inference: T5 generation does not accept projected decoder inputs")
	}
	if err := validateStopSequences(options.StopSequences); err != nil {
		return nil, "", nil, err
	}
	if options.Sampler == nil {
		var err error
		options.Sampler, err = sampling.New(sampling.Config{})
		if err != nil {
			return nil, "", nil, err
		}
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
	if options.LoRAConfigured {
		next, scaleErr := validatedLoRAScales(len(r.loraAdapters), options.LoRA)
		if scaleErr != nil {
			return nil, "", nil, scaleErr
		}
		previous := make([]float32, len(r.loraAdapters))
		for index := range r.loraAdapters {
			previous[index] = r.loraAdapters[index].scale
			r.loraAdapters[index].scale = next[index]
		}
		defer func() {
			for index := range r.loraAdapters {
				r.loraAdapters[index].scale = previous[index]
			}
		}()
	}

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
				Tokens:        append([]tokenizer.TokenID(nil), sourceIDs...),
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
		samplerHistory := make([]int, len(history))
		for index, id := range history {
			samplerHistory[index] = int(id)
		}
		next := 0
		selectedProbability := 0.0
		var topProbabilities []sampling.TokenProbability
		if options.PostSamplingProbabilities > 0 {
			probabilities, sampleErr := options.Sampler.SampleWithHistoryProbabilities(
				logits, samplerHistory, options.PostSamplingProbabilities,
			)
			if sampleErr != nil {
				return nil, "", nil, sampleErr
			}
			next = probabilities.Token
			selectedProbability = probabilities.SelectedProbability
			topProbabilities = probabilities.Top
		} else {
			var sampleErr error
			next, sampleErr = options.Sampler.SampleWithHistory(logits, samplerHistory)
			if sampleErr != nil {
				return nil, "", nil, sampleErr
			}
		}
		nextID := tokenizer.TokenID(next)
		generated = append(generated, nextID)
		history = append(history, nextID)
		pending = nextID
		event := TokenEvent{
			ID: nextID, Index: generatedIndex, Logits: logits,
			SelectedProbability: selectedProbability, TopProbabilities: topProbabilities,
		}
		if options.OnToken != nil || options.ShouldStop != nil || len(options.StopSequences) > 0 {
			piece, pieceErr := r.vocab.DecodePiece(nextID, false)
			if pieceErr != nil {
				return nil, "", nil, pieceErr
			}
			event.Piece = piece
			if options.OnToken != nil {
				if callbackErr := options.OnToken(event); callbackErr != nil {
					return nil, "", nil, callbackErr
				}
			}
		}
		generatedText.WriteString(event.Piece)
		if options.ShouldStop != nil && options.ShouldStop(event) {
			break
		}
		if r.vocab.IsEOG(nextID) || matchesStopSequence(generatedText.String(), options.StopSequences) {
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
