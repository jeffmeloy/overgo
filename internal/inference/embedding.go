package inference

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"math"
	"slices"

	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

type EmbeddingPooling string

const (
	EmbeddingPoolingMean EmbeddingPooling = "mean"
	EmbeddingPoolingLast EmbeddingPooling = "last"
	EmbeddingPoolingNone EmbeddingPooling = "none"
)

type EmbeddingOptions struct {
	Pooling   EmbeddingPooling
	Normalize int
}

type EmbeddingResult struct {
	Vectors [][]float32
	Tokens  int
}

// Embed: returns historical mean-pooled, L2-normalized embedding
func (r *Runner) Embed(ctx context.Context, text string) ([]float32, int, error) {
	result, err := r.EmbedAdvanced(ctx, text, EmbeddingOptions{
		Pooling:   EmbeddingPoolingMean,
		Normalize: 2,
	})
	if err != nil {
		return nil, 0, err
	}
	return result.Vectors[0], result.Tokens, nil
}

func (r *Runner) EmbedAdvanced(
	ctx context.Context,
	text string,
	options EmbeddingOptions,
) (EmbeddingResult, error) {
	if r == nil || r.vocab == nil {
		return EmbeddingResult{}, errRunnerNil
	}
	ids, err := r.vocab.Encode(text, tokenizer.EncodeOptions{AddSpecial: true})
	if err != nil {
		return EmbeddingResult{}, err
	}
	if len(ids) == 0 {
		return EmbeddingResult{}, errors.New("inference: embedding text produced no tokens")
	}
	return r.EmbedTokensAdvanced(ctx, ids, options)
}

// EmbedTokens embeds exact caller-owned token sequence without implicit
// BOS/EOS insertion or lossy text round trip
func (r *Runner) EmbedTokens(
	ctx context.Context,
	input []tokenizer.TokenID,
) ([]float32, int, error) {
	result, err := r.EmbedTokensAdvanced(ctx, input, EmbeddingOptions{
		Pooling:   EmbeddingPoolingMean,
		Normalize: 2,
	})
	if err != nil {
		return nil, 0, err
	}
	return result.Vectors[0], result.Tokens, nil
}

func (r *Runner) EmbedTokensAdvanced(
	ctx context.Context,
	input []tokenizer.TokenID,
	options EmbeddingOptions,
) (EmbeddingResult, error) {
	if r == nil || r.vocab == nil {
		return EmbeddingResult{}, errRunnerNil
	}
	if len(input) == 0 {
		return EmbeddingResult{}, errors.New("inference: embedding token list is empty")
	}
	ids := slices.Clone(input)
	for index, id := range ids {
		if _, ok := r.vocab.Token(id); !ok {
			return EmbeddingResult{}, fmt.Errorf(
				"inference: embedding token %d has out-of-range ID %d",
				index,
				id,
			)
		}
	}
	if err := r.lockOpen(); err != nil {
		return EmbeddingResult{}, err
	}
	defer r.mu.Unlock()
	hidden, err := r.forwardLocked(ctx, ids)
	if err != nil {
		return EmbeddingResult{}, err
	}
	poolOptions := options
	poolOptions.Normalize = -1
	vectors, err := poolEmbeddings(hidden, poolOptions)
	if err != nil {
		return EmbeddingResult{}, err
	}
	vectors, err = r.projectEmbeddingVectors(ctx, vectors)
	if err != nil {
		return EmbeddingResult{}, err
	}
	pooling := options.Pooling
	pooling = cmp.Or(pooling, EmbeddingPoolingMean)
	if pooling != EmbeddingPoolingNone {
		for _, vector := range vectors {
			normalizeEmbedding(vector, options.Normalize)
		}
	}
	return EmbeddingResult{Vectors: vectors, Tokens: len(ids)}, nil
}

func (r *Runner) projectEmbeddingVectors(
	ctx context.Context,
	vectors [][]float32,
) ([][]float32, error) {
	var projections []gguf.TensorInfo
	if r.weights.Dense2Output != nil {
		projections = append(projections, *r.weights.Dense2Output)
	}
	if r.weights.Dense3Output != nil {
		projections = append(projections, *r.weights.Dense3Output)
	}
	if len(projections) == 0 {
		return vectors, nil
	}
	current := vectors
	for _, projection := range projections {
		var err error
		if r.hasPreloadedWeights() {
			current, err = r.projectEmbeddingVectorsDevice(ctx, current, projection)
		} else {
			current, err = r.projectEmbeddingVectorsHost(ctx, current, projection)
		}
		if err != nil {
			return nil, err
		}
	}
	return current, nil
}

func (r *Runner) projectEmbeddingVectorsHost(
	ctx context.Context,
	vectors [][]float32,
	projection gguf.TensorInfo,
) ([][]float32, error) {
	result := make([][]float32, len(vectors))
	for index, vector := range vectors {
		projected, err := model.DotRows(ctx, r.file, projection, vector)
		if err != nil {
			return nil, fmt.Errorf("inference: embedding projection %q: %w", projection.Name, err)
		}
		result[index] = projected
	}
	return result, nil
}

func (r *Runner) projectEmbeddingVectorsDevice(
	ctx context.Context,
	vectors [][]float32,
	projection gguf.TensorInfo,
) ([][]float32, error) {
	inputValue, err := reference.NewMatrixRows(vectors)
	if err != nil {
		return nil, err
	}
	runtime := r.newInferenceGraphRuntime(ctx)
	input := runtime.input("embedding_projection.input", inputValue)
	weight, err := runtime.weight(projection)
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
	projected := results[output]
	outWidth := int(projected.Shape.Dims[0])
	result := make([][]float32, len(vectors))
	for index := range result {
		start := index * outWidth
		result[index] = slices.Clone(projected.Data[start : start+outWidth])
	}
	return result, nil
}

func poolEmbeddings(
	hidden reference.Value,
	options EmbeddingOptions,
) ([][]float32, error) {
	width, tokens, valid := hidden.MatrixExtents()
	if !valid {
		return nil, errors.New("inference: hidden state has invalid embedding shape")
	}
	pooling := options.Pooling
	pooling = cmp.Or(pooling, EmbeddingPoolingMean)
	switch pooling {
	case EmbeddingPoolingNone:
		result := make([][]float32, tokens)
		for tokenIndex := range tokens {
			result[tokenIndex] = slices.Clone(hidden.Data[tokenIndex*width : (tokenIndex+1)*width])
		}
		return result, nil
	case EmbeddingPoolingLast:
		vector := slices.Clone(hidden.LastRowView().Data)
		normalizeEmbedding(vector, options.Normalize)
		return [][]float32{vector}, nil
	case EmbeddingPoolingMean:
		vector := make([]float32, width)
		for tokenIndex := range tokens {
			row := hidden.Data[tokenIndex*width : (tokenIndex+1)*width]
			for index, value := range row {
				vector[index] += value
			}
		}
		inverseTokens := float32(1) / float32(tokens)
		for index := range vector {
			vector[index] *= inverseTokens
		}
		normalizeEmbedding(vector, options.Normalize)
		return [][]float32{vector}, nil
	default:
		return nil, fmt.Errorf("inference: unsupported embedding pooling %q", pooling)
	}
}

func normalizeEmbedding(vector []float32, norm int) {
	sum := 0.0
	switch norm {
	case -1:
		sum = 1
	case 0:
		for _, value := range vector {
			sum = max(sum, math.Abs(float64(value)))
		}
		sum /= 32760
	case 2:
		for _, value := range vector {
			sum += float64(value) * float64(value)
		}
		sum = math.Sqrt(sum)
	default:
		for _, value := range vector {
			sum += math.Pow(math.Abs(float64(value)), float64(norm))
		}
		sum = math.Pow(sum, 1/float64(norm))
	}
	scale := float32(0)
	if sum > 0 {
		scale = float32(1 / sum)
	}
	for index := range vector {
		vector[index] *= scale
	}
}
