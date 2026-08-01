package inference

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"

	"llamacpp2go/internal/cuda/driver"
	"llamacpp2go/internal/model"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/dtype"
	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
)

type RankResult struct {
	Scores []float32
	Labels []string
	Tokens int
}

func (r *Runner) RankPair(
	ctx context.Context,
	query string,
	document string,
) (RankResult, error) {
	if r == nil || r.vocab == nil {
		return RankResult{}, errors.New("inference: runner is nil")
	}
	prompt, err := r.rankPairPrompt(query, document)
	if err != nil {
		return RankResult{}, err
	}
	return r.Rank(ctx, prompt)
}

func (r *Runner) rankPairPrompt(query, document string) (string, error) {
	prompt := metadataString(r.file, "tokenizer.chat_template.rerank")
	if prompt != "" {
		prompt = strings.ReplaceAll(prompt, "{query}", query)
		prompt = strings.ReplaceAll(prompt, "{document}", document)
		return prompt, nil
	}
	var separator strings.Builder
	if r.vocab.AddEOS && r.vocab.EOS != tokenizer.NullToken {
		separator.WriteString(vocabularyTokenText(r.vocab, r.vocab.EOS))
	}
	if r.vocab.AddSEP && r.vocab.SEP != tokenizer.NullToken {
		separator.WriteString(vocabularyTokenText(r.vocab, r.vocab.SEP))
	}
	if separator.Len() == 0 {
		return "", errors.New("inference: rank pair has no rerank template or EOS/SEP separator")
	}
	return query + separator.String() + document, nil
}

func (r *Runner) Rank(ctx context.Context, text string) (RankResult, error) {
	if r == nil || r.vocab == nil {
		return RankResult{}, errors.New("inference: runner is nil")
	}
	ids, err := r.vocab.Encode(text, tokenizer.EncodeOptions{AddSpecial: true})
	if err != nil {
		return RankResult{}, err
	}
	return r.RankTokens(ctx, ids)
}

func (r *Runner) RankTokens(
	ctx context.Context,
	input []tokenizer.TokenID,
) (RankResult, error) {
	return r.RankTokensWithProjectedInputs(ctx, input, ProjectedInputs{})
}

// RankTokensWithProjectedInputs: exact-token rank with optional VL projections.
func (r *Runner) RankTokensWithProjectedInputs(
	ctx context.Context,
	input []tokenizer.TokenID,
	inputs ProjectedInputs,
) (RankResult, error) {
	if r == nil || r.vocab == nil {
		return RankResult{}, errors.New("inference: runner is nil")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return RankResult{}, errors.New("inference: runner is closed")
	}
	if r.spec.Architecture != "qwen3" && r.spec.Architecture != "qwen3vl" {
		return RankResult{}, fmt.Errorf(
			"inference: architecture %q has no pinned Qwen rank graph",
			r.spec.Architecture,
		)
	}
	if r.weights.ClassifierOutput == nil {
		return RankResult{}, errors.New("inference: classifier output tensor is missing")
	}
	if len(input) == 0 {
		return RankResult{}, errors.New("inference: rank token list is empty")
	}
	if inputs.MultiAxisPositions != nil && !supportsMultiAxisPositions(r.spec) {
		return RankResult{}, errors.New("inference: model does not support multi-axis positions")
	}
	if len(inputs.DeepstackEmbeddings) > 0 && !supportsDeepstackInputs(r.spec) {
		return RankResult{}, errors.New("inference: model does not support deepstack embeddings")
	}
	hidden, _, err := r.forwardCachedWithEmbeddingOverridesLocked(
		ctx, append([]tokenizer.TokenID(nil), input...), nil,
		inputs.EmbeddingOverrides, inputs.MultiAxisPositions, inputs.DeepstackEmbeddings,
	)
	if err != nil {
		return RankResult{}, err
	}
	width := int(hidden.Shape.Dims[0])
	tokens := int(hidden.Shape.Dims[1])
	if hidden.Shape.Rank != 2 || width != int(r.spec.EmbeddingLength) ||
		tokens != len(input) || len(hidden.Data) != width*tokens {
		return RankResult{}, errors.New("inference: rank hidden-state shape is incompatible")
	}
	last := hidden.Data[(tokens-1)*width:]
	scores, err := r.projectRankScores(ctx, last)
	if err != nil {
		return RankResult{}, err
	}
	softmaxScores(scores)
	labels := append([]string(nil), r.spec.ClassifierLabels...)
	if len(labels) == 0 {
		labels = make([]string, len(scores))
		for index := range labels {
			labels[index] = fmt.Sprint(index)
		}
	}
	return RankResult{Scores: scores, Labels: labels, Tokens: tokens}, nil
}

func (r *Runner) projectRankScores(ctx context.Context, hidden []float32) ([]float32, error) {
	info := *r.weights.ClassifierOutput
	if !r.hasPreloadedWeights() {
		return model.DotRows(ctx, r.file, info, hidden, 1024)
	}
	builder := r.newGraphBuilder()
	weight, pointer, err := r.deviceInput(builder, info)
	if err != nil {
		return nil, err
	}
	inputShape := tensor.MustShape(uint64(len(hidden)), 1)
	input := builder.Input("rank.input", dtype.F32, inputShape)
	output := builder.MulMat(weight, input)
	if err := builder.Err(); err != nil {
		return nil, err
	}
	inputValue, err := reference.NewValue(inputShape, hidden)
	if err != nil {
		return nil, err
	}
	results, err := r.cuda.ExecuteWithDeviceFeeds(
		ctx,
		[]*tensor.Tensor{output},
		map[*tensor.Tensor]reference.Value{input: inputValue},
		map[*tensor.Tensor]driver.DevicePtr{weight: pointer},
	)
	if err != nil {
		return nil, err
	}
	return append([]float32(nil), results[output].Data...), nil
}

func softmaxScores(scores []float32) {
	if len(scores) == 0 {
		return
	}
	maximum := scores[0]
	for _, score := range scores[1:] {
		maximum = max(maximum, score)
	}
	var sum float64
	for index, score := range scores {
		value := math.Exp(float64(score - maximum))
		scores[index] = float32(value)
		sum += value
	}
	if sum == 0 || math.IsNaN(sum) {
		return
	}
	inverse := float32(1 / sum)
	for index := range scores {
		scores[index] *= inverse
	}
}
