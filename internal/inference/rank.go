package inference

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"

	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

type RankResult struct {
	Scores []float32
	Labels []string
	Tokens int
}

func (r *Runner) SupportsRank() bool {
	return r != nil && r.weights.ClassifierOutput != nil &&
		r.program.Model.ProjectedInput().ClassifierHead
}

func (r *Runner) RankPair(
	ctx context.Context,
	query string,
	document string,
) (RankResult, error) {
	if r == nil || r.vocab == nil {
		return RankResult{}, errRunnerNil
	}
	prompt := metadataString(r.file, "tokenizer.chat_template.rerank")
	if prompt != "" {
		ids, err := r.vocab.Encode(rankTemplatePrompt(prompt, query, document), tokenizer.EncodeOptions{
			ParseSpecial: true,
		})
		if err != nil {
			return RankResult{}, err
		}
		return r.RankTokens(ctx, ids)
	}
	queryIDs, err := r.vocab.Encode(query, tokenizer.EncodeOptions{})
	if err != nil {
		return RankResult{}, err
	}
	documentIDs, err := r.vocab.Encode(document, tokenizer.EncodeOptions{})
	if err != nil {
		return RankResult{}, err
	}
	ids, err := assembleRankPairTokens(r.vocab, queryIDs, documentIDs)
	if err != nil {
		return RankResult{}, err
	}
	return r.RankTokens(ctx, ids)
}

func rankTemplatePrompt(source, query, document string) string {
	prompt := strings.ReplaceAll(source, "{query}", query)
	return strings.ReplaceAll(prompt, "{document}", document)
}

func assembleRankPairTokens(
	vocab *tokenizer.Vocab,
	query []tokenizer.TokenID,
	document []tokenizer.TokenID,
) ([]tokenizer.TokenID, error) {
	eos := vocab.EOS
	if eos == tokenizer.NullToken {
		eos = vocab.SEP
	}
	if vocab.AddEOS && eos == tokenizer.NullToken {
		return nil, errors.New("inference: rank pair EOS/SEP token is missing")
	}
	if vocab.AddSEP && vocab.SEP == tokenizer.NullToken {
		return nil, errors.New("inference: rank pair SEP token is missing")
	}
	if !vocab.AddEOS && !vocab.AddSEP {
		return nil, errors.New("inference: rank pair has no rerank template or EOS/SEP separator")
	}
	result := make([]tokenizer.TokenID, 0, len(query)+len(document)+4)
	if vocab.AddBOS {
		if vocab.BOS == tokenizer.NullToken {
			return nil, errors.New("inference: rank pair BOS token is missing")
		}
		result = append(result, vocab.BOS)
	}
	result = append(result, query...)
	if vocab.AddEOS {
		result = append(result, eos)
	}
	if vocab.AddSEP {
		result = append(result, vocab.SEP)
	}
	result = append(result, document...)
	if vocab.AddEOS {
		result = append(result, eos)
	}
	return result, nil
}

func (r *Runner) Rank(ctx context.Context, text string) (RankResult, error) {
	if r == nil || r.vocab == nil {
		return RankResult{}, errRunnerNil
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
		return RankResult{}, errRunnerNil
	}
	if err := r.lockOpen(); err != nil {
		return RankResult{}, err
	}
	defer r.mu.Unlock()
	program := r.program.Model.ProjectedInput()
	if !program.ClassifierHead {
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
	if inputs.MultiAxisPositions != nil && !program.MultiAxis {
		return RankResult{}, errors.New("inference: model does not support multi-axis positions")
	}
	if len(inputs.DeepstackEmbeddings) > 0 && program.DeepstackStreams == 0 {
		return RankResult{}, errors.New("inference: model does not support deepstack embeddings")
	}
	hidden, _, err := r.forwardCachedProjectedChunkLocked(ctx, slices.Clone(input), nil, inputs)
	if err != nil {
		return RankResult{}, err
	}
	rows, valid := r.spec.SequenceRows(hidden)
	tokens := int(rows)
	if !valid || tokens != len(input) {
		return RankResult{}, errors.New("inference: rank hidden-state shape is incompatible")
	}
	last := hidden.LastRowView()
	scores, err := r.projectRankScores(ctx, last.Data)
	if err != nil {
		return RankResult{}, err
	}
	softmaxScores(scores)
	labels := slices.Clone(r.spec.ClassifierLabels)
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
		return model.DotRows(ctx, r.file, info, hidden)
	}
	inputShape := tensor.MustShape(uint64(len(hidden)), 1)
	inputValue, err := reference.NewValue(inputShape, hidden)
	if err != nil {
		return nil, err
	}
	runtime := r.newInferenceGraphRuntime(ctx)
	input := runtime.input("rank.input", inputValue)
	weight, err := runtime.weight(info)
	if err != nil {
		return nil, err
	}
	output := runtime.builder.MulMat(weight, input)
	if err := runtime.builder.Err(); err != nil {
		return nil, err
	}
	results, err := runtime.execute(output)
	if err != nil {
		return nil, err
	}
	return slices.Clone(results[output].Data), nil
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
