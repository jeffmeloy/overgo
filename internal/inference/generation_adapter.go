package inference

import (
	"errors"
	"strings"

	"llamacpp2go/internal/sampling"
	"llamacpp2go/internal/tokenizer"
)

func normalizeGenerateOptions(options *GenerateOptions) error {
	if options.MaxNewTokens < 0 {
		return errors.New("inference: max new tokens is negative")
	}
	if options.MinCacheReuse < 0 {
		return errors.New("inference: minimum cache reuse is negative")
	}
	if options.KeepTokens < -1 {
		return errors.New("inference: keep token count must be at least -1")
	}
	if options.DiscardTokens < 0 {
		return errors.New("inference: discard token count is negative")
	}
	if options.PostSamplingProbabilities < 0 {
		return errors.New("inference: post-sampling probability count is negative")
	}
	if err := validateStopSequences(options.StopSequences); err != nil {
		return err
	}
	if options.Sampler != nil {
		return nil
	}
	sampler, err := sampling.New(sampling.Config{})
	options.Sampler = sampler
	return err
}

func (r *Runner) applyGenerationLoRA(options GenerateOptions) (func(), error) {
	if !options.LoRAConfigured {
		return func() {}, nil
	}
	next, err := validatedLoRAScales(len(r.loraAdapters), options.LoRA)
	if err != nil {
		return nil, err
	}
	previous := make([]float32, len(r.loraAdapters))
	for index := range r.loraAdapters {
		previous[index] = r.loraAdapters[index].scale
		r.loraAdapters[index].scale = next[index]
	}
	return func() {
		for index := range r.loraAdapters {
			r.loraAdapters[index].scale = previous[index]
		}
	}, nil
}

func tokenHistory(ids []tokenizer.TokenID) []int {
	history := make([]int, len(ids))
	for index, id := range ids {
		history[index] = int(id)
	}
	return history
}

func sampleGenerationToken(
	logits []float32,
	ids []tokenizer.TokenID,
	options GenerateOptions,
) (TokenEvent, error) {
	history := tokenHistory(ids)
	if options.PostSamplingProbabilities > 0 {
		result, err := options.Sampler.SampleWithHistoryProbabilities(
			logits, history, options.PostSamplingProbabilities,
		)
		return TokenEvent{
			ID: tokenizer.TokenID(result.Token), Logits: logits,
			SelectedProbability: result.SelectedProbability,
			TopProbabilities:    result.Top,
		}, err
	}
	next, err := options.Sampler.SampleWithHistory(logits, history)
	return TokenEvent{ID: tokenizer.TokenID(next), Logits: logits}, err
}

func (r *Runner) deliverGenerationToken(
	event *TokenEvent,
	options GenerateOptions,
	generatedText *strings.Builder,
) (bool, error) {
	if options.OnToken != nil || options.ShouldStop != nil || len(options.StopSequences) > 0 {
		piece, err := r.vocab.DecodePiece(event.ID, false)
		if err != nil {
			return false, err
		}
		event.Piece = piece
		if options.OnToken != nil {
			if err := options.OnToken(*event); err != nil {
				return false, err
			}
		}
	}
	generatedText.WriteString(event.Piece)
	return options.ShouldStop != nil && options.ShouldStop(*event) ||
		r.vocab.IsEOG(event.ID) ||
		matchesStopSequence(generatedText.String(), options.StopSequences), nil
}
