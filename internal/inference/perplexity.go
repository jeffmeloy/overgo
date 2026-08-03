package inference

import (
	"context"
	"errors"
	"fmt"
	"math"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/model"
	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
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
		return PerplexityResult{}, errors.New("inference: runner is nil")
	}
	if text == "" {
		return PerplexityResult{}, errors.New("inference: perplexity text is empty")
	}
	ids, err := r.vocab.Encode(text, tokenizer.EncodeOptions{AddSpecial: true})
	if err != nil {
		return PerplexityResult{}, err
	}
	if len(ids) < 2 {
		return PerplexityResult{}, errors.New(
			"inference: perplexity requires at least two tokens including special tokens",
		)
	}
	if options.ContextSize < 0 {
		return PerplexityResult{}, errors.New("inference: perplexity context size is negative")
	}
	if options.ContextSize > int(r.spec.ContextLength) {
		return PerplexityResult{}, fmt.Errorf(
			"inference: perplexity context size %d exceeds model context length %d",
			options.ContextSize,
			r.spec.ContextLength,
		)
	}
	if options.ContextSize > 0 && options.ContextSize < 4 {
		return PerplexityResult{}, errors.New(
			"inference: windowed perplexity context size must be at least 4",
		)
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return PerplexityResult{}, errors.New("inference: runner is closed")
	}
	outputTable := r.weights.TokenEmbedding
	if r.weights.Output != nil {
		outputTable = *r.weights.Output
	}
	result := PerplexityResult{
		TokenCount: len(ids),
	}
	if options.ContextSize == 0 {
		scores, negativeLogLik, scoreErr := r.scoreTokenWindow(
			ctx,
			ids,
			1,
			0,
			outputTable,
		)
		if scoreErr != nil {
			return PerplexityResult{}, scoreErr
		}
		result.Scores = scores
		result.NegativeLogLikelihood = negativeLogLik
	} else {
		contextSize := options.ContextSize
		if len(ids) < 2*contextSize {
			return PerplexityResult{}, fmt.Errorf(
				"inference: windowed perplexity needs at least %d tokens, got %d",
				2*contextSize,
				len(ids),
			)
		}
		windowCount := len(ids) / contextSize
		firstTarget := contextSize/2 + 1
		result.Scores = make(
			[]TokenScore,
			0,
			windowCount*(contextSize-firstTarget),
		)
		for window := range windowCount {
			offset := window * contextSize
			windowIDs := ids[offset : offset+contextSize]
			if r.vocab.AddBOS {
				windowIDs = append([]tokenizer.TokenID(nil), windowIDs...)
				windowIDs[0] = r.vocab.BOS
			}
			scores, negativeLogLik, scoreErr := r.scoreTokenWindow(
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
			result.NegativeLogLikelihood += negativeLogLik
		}
	}
	result.EvaluatedTokens = len(result.Scores)
	result.Perplexity = math.Exp(
		result.NegativeLogLikelihood / float64(result.EvaluatedTokens),
	)
	if math.IsNaN(result.Perplexity) || math.IsInf(result.Perplexity, 0) {
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
) ([]TokenScore, float64, error) {
	if firstTarget < 1 || firstTarget >= len(ids) {
		return nil, 0, errors.New("invalid first target position")
	}
	hidden, _, err := r.forwardCachedLocked(ctx, ids[:len(ids)-1], nil)
	if err != nil {
		return nil, 0, err
	}
	logits, err := r.logitsBatch(ctx, outputInfo, hidden)
	if err != nil {
		return nil, 0, err
	}
	vocabularySize := r.vocab.Len()
	if len(logits) != vocabularySize*(len(ids)-1) {
		return nil, 0, fmt.Errorf(
			"inference: logits contain %d values, need %d",
			len(logits),
			vocabularySize*(len(ids)-1),
		)
	}
	scores := make([]TokenScore, 0, len(ids)-firstTarget)
	var negativeLogLik float64
	for targetPosition := firstTarget; targetPosition < len(ids); targetPosition++ {
		logitPosition := targetPosition - 1
		value, scoreErr := negativeLogProbability(
			logits[logitPosition*vocabularySize:(logitPosition+1)*vocabularySize],
			int(ids[targetPosition]),
		)
		if scoreErr != nil {
			return nil, 0, fmt.Errorf(
				"score token at position %d: %w",
				globalOffset+targetPosition,
				scoreErr,
			)
		}
		negativeLogLik += value
		scores = append(scores, TokenScore{
			Position:       globalOffset + targetPosition,
			TokenID:        ids[targetPosition],
			NegativeLogLik: value,
		})
	}
	return scores, negativeLogLik, nil
}

func (r *Runner) logitsBatch(
	ctx context.Context,
	outputInfo gguf.TensorInfo,
	hidden reference.Value,
) ([]float32, error) {
	if hidden.Shape.Rank != 2 || hidden.Shape.Dims[0] != uint64(r.spec.EmbeddingLength) {
		return nil, errors.New("inference: batched logits require embedding-by-token hidden states")
	}
	tokenCount := int(hidden.Shape.Dims[1])
	width := int(hidden.Shape.Dims[0])
	if !r.hasPreloadedWeights() {
		result := make([]float32, 0, tokenCount*r.vocab.Len())
		for token := range tokenCount {
			logits, err := model.DotRows(
				ctx,
				r.file,
				outputInfo,
				hidden.Data[token*width:(token+1)*width],
				1024,
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
	if scale := r.spec.OutputLogitMultiplier(); scale != 1 {
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
	if len(logits) == 0 || target < 0 || target >= len(logits) {
		return 0, errors.New("invalid logits or target")
	}
	maximum := math.Inf(-1)
	for _, value := range logits {
		converted := float64(value)
		if math.IsNaN(converted) {
			return 0, errors.New("logits contain NaN")
		}
		if converted > maximum {
			maximum = converted
		}
	}
	if math.IsInf(maximum, -1) {
		return 0, errors.New("all logits are negative infinity")
	}
	var exponentialSum float64
	for _, value := range logits {
		exponentialSum += math.Exp(float64(value) - maximum)
	}
	result := maximum + math.Log(exponentialSum) - float64(logits[target])
	if math.IsNaN(result) || math.IsInf(result, 0) {
		return 0, errors.New("negative log-likelihood is not finite")
	}
	return result, nil
}
