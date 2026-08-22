package inference

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"

	"overgo/internal/checked"
	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/sequencescore"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

// TokenScore: negative log-likelihood assigned to one observed token
type TokenScore struct {
	Position       int               `json:"position"`
	TokenID        tokenizer.TokenID `json:"token_id"`
	NegativeLogLik float64           `json:"negative_log_likelihood"`
}

// PerplexityResult summarizes teacher-forced next-token scoring
type PerplexityResult struct {
	TokenCount            int          `json:"token_count"`
	EvaluatedTokens       int          `json:"evaluated_tokens"`
	NegativeLogLikelihood float64      `json:"negative_log_likelihood"`
	Perplexity            float64      `json:"perplexity"`
	Scores                []TokenScore `json:"scores,omitempty"`
}

// PerplexityOptions: selects scoring convention; ContextSize zero scores
// every next token in one continuous context; positive size uses llama.cpp's
// disjoint-window convention and scores latter half of each full window
type PerplexityOptions struct {
	ContextSize int
}

// Perplexity: tokenizes text with model-default BOS/EOS behavior and evaluates
// every token after first under teacher forcing
func (r *Runner) Perplexity(ctx context.Context, text string) (PerplexityResult, error) {
	return r.PerplexityWithOptions(ctx, text, PerplexityOptions{})
}

// PerplexityWithOptions: evaluates teacher-forced next-token likelihoods
func (r *Runner) PerplexityWithOptions(
	ctx context.Context,
	text string,
	options PerplexityOptions,
) (PerplexityResult, error) {
	if r == nil || r.vocab == nil {
		return PerplexityResult{}, errRunnerNil
	}
	if text == "" {
		return PerplexityResult{}, errors.New("inference: perplexity text is empty")
	}
	ids, err := r.vocab.Encode(text, tokenizer.EncodeOptions{AddSpecial: true})
	if err != nil {
		return PerplexityResult{}, err
	}
	if len(ids) < tensor.PairedExtent {
		return PerplexityResult{}, errors.New(
			"inference: perplexity requires at least two tokens including special tokens",
		)
	}
	if !checked.NonNegativeInts(options.ContextSize) {
		return PerplexityResult{}, errors.New("inference: perplexity context size is negative")
	}
	if options.ContextSize > int(r.spec.ContextLength) {
		return PerplexityResult{}, fmt.Errorf(
			"inference: perplexity context size %d exceeds model context length %d",
			options.ContextSize,
			r.spec.ContextLength,
		)
	}
	minimumWindow := tensor.PairedExtent * tensor.PairedExtent
	if checked.Nonzero(options.ContextSize) && options.ContextSize < minimumWindow {
		return PerplexityResult{}, errors.New(
			"inference: windowed perplexity context size must be at least 4",
		)
	}

	if err := r.lockOpen(); err != nil {
		return PerplexityResult{}, err
	}
	defer r.mu.Unlock()
	outputTable := r.outputTensor()
	result := PerplexityResult{
		TokenCount: len(ids),
	}
	if !checked.Nonzero(options.ContextSize) {
		scores, negativeLogLik, scoreErr := r.scoreTokenWindow(
			ctx,
			ids,
			tensor.SingletonExtent,
			tensor.FirstOffset,
			outputTable,
		)
		if scoreErr != nil {
			return PerplexityResult{}, scoreErr
		}
		result.Scores = scores
		result.NegativeLogLikelihood = -negativeLogLik.LogProbability
	} else {
		contextSize := options.ContextSize
		if len(ids) < tensor.PairedExtent*contextSize {
			return PerplexityResult{}, fmt.Errorf(
				"inference: windowed perplexity needs at least %d tokens, got %d",
				tensor.PairedExtent*contextSize,
				len(ids),
			)
		}
		windowCount := len(ids) / contextSize
		firstTarget := contextSize/tensor.PairedExtent + tensor.SingletonExtent
		result.Scores = make(
			[]TokenScore,
			tensor.FirstOffset,
			windowCount*(contextSize-firstTarget),
		)
		for window := range windowCount {
			offset := window * contextSize
			windowIDs := ids[offset : offset+contextSize]
			if r.vocab.AddBOS {
				windowIDs = slices.Clone(windowIDs)
				windowIDs[tensor.FirstOffset] = r.vocab.BOS
			}
			scores, continuation, scoreErr := r.scoreTokenWindow(
				ctx,
				windowIDs,
				firstTarget,
				offset,
				outputTable,
			)
			if scoreErr != nil {
				return PerplexityResult{}, fmt.Errorf(
					"inference: perplexity window %d: %w",
					window,
					scoreErr,
				)
			}
			result.Scores = append(result.Scores, scores...)
			result.NegativeLogLikelihood -= continuation.LogProbability
		}
	}
	result.EvaluatedTokens = len(result.Scores)
	result.Perplexity = math.Exp(
		result.NegativeLogLikelihood / float64(result.EvaluatedTokens),
	)
	if !checked.Finite64(result.Perplexity) {
		return PerplexityResult{}, errors.New("inference: perplexity is not finite")
	}
	return result, nil
}

func (r *Runner) scoreTokenWindow(
	ctx context.Context,
	ids []tokenizer.TokenID,
	firstTarget int,
	globalOffset int,
	outputInfo gguf.TensorInfo,
) ([]TokenScore, sequencescore.Score, error) {
	if firstTarget < tensor.SingletonExtent || firstTarget >= len(ids) {
		return nil, sequencescore.Score{}, errors.New("invalid first target position")
	}
	hidden, _, err := r.forwardCachedLocked(ctx, ids[:len(ids)-tensor.SingletonExtent], nil)
	if err != nil {
		return nil, sequencescore.Score{}, err
	}
	logits, err := r.logitsBatch(ctx, outputInfo, hidden)
	if err != nil {
		return nil, sequencescore.Score{}, err
	}
	vocabularySize := r.vocab.Len()
	if len(logits) != vocabularySize*(len(ids)-tensor.SingletonExtent) {
		return nil, sequencescore.Score{}, fmt.Errorf(
			"inference: logits contain %d values, need %d",
			len(logits),
			vocabularySize*(len(ids)-tensor.SingletonExtent),
		)
	}
	scores := make([]TokenScore, 0, len(ids)-firstTarget)
	var continuation sequencescore.Accumulator
	for targetPosition := firstTarget; targetPosition < len(ids); targetPosition++ {
		logitPosition := targetPosition - tensor.SingletonExtent
		value, scoreErr := negativeLogProbability(
			logits[logitPosition*vocabularySize:(logitPosition+tensor.SingletonExtent)*vocabularySize],
			int(ids[targetPosition]),
		)
		if scoreErr != nil {
			return nil, sequencescore.Score{}, fmt.Errorf(
				"score token at position %d: %w",
				globalOffset+targetPosition,
				scoreErr,
			)
		}
		if err := continuation.Observe(-value); err != nil {
			return nil, sequencescore.Score{}, err
		}
		scores = append(scores, TokenScore{
			Position:       globalOffset + targetPosition,
			TokenID:        ids[targetPosition],
			NegativeLogLik: value,
		})
	}
	result, err := continuation.Result()
	return scores, result, err
}

func (r *Runner) logitsBatch(
	ctx context.Context,
	outputInfo gguf.TensorInfo,
	hidden reference.Value,
) ([]float32, error) {
	rows, valid := tensor.MatrixRows(hidden.Shape, uint64(r.spec.EmbeddingLength))
	if !valid {
		return nil, errors.New("inference: batched logits require embedding-by-token hidden states")
	}
	tokenCount := int(rows)
	width := int(r.spec.EmbeddingLength)
	if !r.hasPreloadedWeights() {
		var initialLength int
		result := make([]float32, initialLength, tokenCount*r.vocab.Len())
		for token := range tokenCount {
			logits, err := model.DotRows(
				ctx,
				r.file,
				outputInfo,
				hidden.Data[token*width:(token+tensor.SingletonExtent)*width],
			)
			if err != nil {
				return nil, err
			}
			if err := addOutputBias(logits, r.outputBias); err != nil {
				return nil, err
			}
			scaleLogits(logits, r.spec.OutputLogitMultiplier())
			result = append(result, logits...)
		}
		return r.finalizeLogits(result), nil
	}
	runtime := r.newInferenceGraphRuntime(ctx)
	table, err := runtime.weight(outputInfo)
	if err != nil {
		return nil, err
	}
	input := runtime.input("perplexity.logits.input", hidden)
	output := runtime.builder.MulMat(table, input)
	if r.weights.OutputBias != nil {
		bias, biasErr := runtime.weight(*r.weights.OutputBias)
		if biasErr != nil {
			return nil, biasErr
		}
		output = runtime.builder.Add(output, bias)
	}
	if scale := r.spec.OutputLogitMultiplier(); !checked.Equal(scale, float32(tensor.SingletonExtent)) {
		output = runtime.builder.Scale(output, scale)
	}
	if err := runtime.builder.Err(); err != nil {
		return nil, err
	}
	results, err := runtime.execute(output)
	if err != nil {
		return nil, err
	}
	return r.finalizeLogits(results[output].Data), nil
}

func negativeLogProbability(logits []float32, target int) (float64, error) {
	if !checked.Nonzero(len(logits)) || !checked.ValidIndex(target, len(logits)) {
		var invalid float64
		return invalid, errors.New("invalid logits or target")
	}
	maximum := math.Inf(-tensor.SingletonExtent)
	for _, value := range logits {
		converted := float64(value)
		if math.IsNaN(converted) {
			var invalid float64
			return invalid, errors.New("logits contain NaN")
		}
		if converted > maximum {
			maximum = converted
		}
	}
	if math.IsInf(maximum, -tensor.SingletonExtent) {
		var invalid float64
		return invalid, errors.New("all logits are negative infinity")
	}
	var exponentialSum float64
	for _, value := range logits {
		exponentialSum += math.Exp(float64(value) - maximum)
	}
	result := maximum + math.Log(exponentialSum) - float64(logits[target])
	if !checked.Finite64(result) {
		var invalid float64
		return invalid, errors.New("negative log-likelihood is not finite")
	}
	return result, nil
}
