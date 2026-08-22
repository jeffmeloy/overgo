package inference

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/checked"
	"overgo/internal/hostmath"
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
		ids, err := r.vocab.Encode(renderClassifierPrompt(prompt, query, document), tokenizer.EncodeOptions{
			ParseSpecial: true,
		})
		if err != nil {
			return RankResult{}, err
		}
		return r.classifyTokens(ctx, ids, ProjectedInputs{})
	}
	queryIDs, err := r.vocab.Encode(query, tokenizer.EncodeOptions{})
	if err != nil {
		return RankResult{}, err
	}
	documentIDs, err := r.vocab.Encode(document, tokenizer.EncodeOptions{})
	if err != nil {
		return RankResult{}, err
	}
	ids, err := assembleClassifierPairTokens(r.vocab, queryIDs, documentIDs)
	if err != nil {
		return RankResult{}, err
	}
	return r.classifyTokens(ctx, ids, ProjectedInputs{})
}

func renderClassifierPrompt(source, query, document string) string {
	prompt := strings.ReplaceAll(source, "{query}", query)
	return strings.ReplaceAll(prompt, "{document}", document)
}

func assembleClassifierPairTokens(
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
	return r.classifyTokens(ctx, ids, ProjectedInputs{})
}

func (r *Runner) RankTokens(
	ctx context.Context,
	input []tokenizer.TokenID,
) (RankResult, error) {
	return r.classifyTokens(ctx, input, ProjectedInputs{})
}

// RankTokensWithProjectedInputs: exact-token rank with optional VL projections.
func (r *Runner) RankTokensWithProjectedInputs(
	ctx context.Context,
	input []tokenizer.TokenID,
	inputs ProjectedInputs,
) (RankResult, error) {
	return r.classifyTokens(ctx, input, inputs)
}

func (r *Runner) classifyTokens(
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
	if !checked.Nonzero(len(input)) {
		return RankResult{}, errors.New("inference: rank token list is empty")
	}
	if inputs.MultiAxisPositions != nil && !program.MultiAxis {
		return RankResult{}, errors.New("inference: model does not support multi-axis positions")
	}
	if checked.Nonzero(len(inputs.DeepstackEmbeddings)) && !checked.Nonzero(program.DeepstackStreams) {
		return RankResult{}, errors.New("inference: model does not support deepstack embeddings")
	}
	hidden, _, err := r.forwardCachedProjectedChunkLocked(ctx, slices.Clone(input), nil, inputs)
	if err != nil {
		return RankResult{}, err
	}
	width, tokens, validHidden := hidden.MatrixExtents()
	if !validHidden || !checked.Equal(width, int(r.spec.EmbeddingLength)) ||
		!checked.Equal(tokens, len(input)) {
		return RankResult{}, errors.New("inference: rank hidden-state shape is incompatible")
	}
	last, err := reference.FinalRows(hidden, tensor.SingletonExtent)
	if err != nil {
		return RankResult{}, errors.New("inference: rank hidden-state shape is incompatible")
	}
	scores, err := r.projectClassifierScores(ctx, last.Data)
	if err != nil {
		return RankResult{}, err
	}
	hostmath.SoftmaxInPlace(scores)
	labels := slices.Clone(r.spec.ClassifierLabels)
	if !checked.Nonzero(len(labels)) {
		labels = make([]string, len(scores))
		for index := range labels {
			labels[index] = fmt.Sprint(index)
		}
	}
	return RankResult{Scores: scores, Labels: labels, Tokens: tokens}, nil
}

func (r *Runner) projectClassifierScores(ctx context.Context, hidden []float32) ([]float32, error) {
	info := *r.weights.ClassifierOutput
	if !r.hasPreloadedWeights() {
		return model.DotRows(ctx, r.file, info, hidden)
	}
	inputShape := tensor.MustShape(uint64(len(hidden)), tensor.SingletonExtent)
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
