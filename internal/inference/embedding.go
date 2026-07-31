package inference

import (
	"context"
	"errors"
	"fmt"
	"math"

	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
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

// Embed returns the historical mean-pooled, L2-normalized embedding.
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
		return EmbeddingResult{}, errors.New("inference: runner is nil")
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

// EmbedTokens embeds an exact caller-owned token sequence without implicit
// BOS/EOS insertion or a lossy text round trip.
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
		return EmbeddingResult{}, errors.New("inference: runner is nil")
	}
	if len(input) == 0 {
		return EmbeddingResult{}, errors.New("inference: embedding token list is empty")
	}
	ids := append([]tokenizer.TokenID(nil), input...)
	for index, id := range ids {
		if _, ok := r.vocab.Token(id); !ok {
			return EmbeddingResult{}, fmt.Errorf(
				"inference: embedding token %d has out-of-range ID %d",
				index,
				id,
			)
		}
	}
	hidden, err := r.Forward(ctx, ids)
	if err != nil {
		return EmbeddingResult{}, err
	}
	vectors, err := poolEmbeddings(hidden, options)
	if err != nil {
		return EmbeddingResult{}, err
	}
	return EmbeddingResult{Vectors: vectors, Tokens: len(ids)}, nil
}

func poolEmbeddings(
	hidden reference.Value,
	options EmbeddingOptions,
) ([][]float32, error) {
	width := int(hidden.Shape.Dims[0])
	tokens := int(hidden.Shape.Dims[1])
	if hidden.Shape.Rank != 2 ||
		width <= 0 ||
		tokens <= 0 ||
		len(hidden.Data) != width*tokens {
		return nil, errors.New("inference: hidden state has invalid embedding shape")
	}
	pooling := options.Pooling
	if pooling == "" {
		pooling = EmbeddingPoolingMean
	}
	switch pooling {
	case EmbeddingPoolingNone:
		result := make([][]float32, tokens)
		for tokenIndex := range tokens {
			result[tokenIndex] = append(
				[]float32(nil),
				hidden.Data[tokenIndex*width:(tokenIndex+1)*width]...,
			)
		}
		return result, nil
	case EmbeddingPoolingLast:
		vector := append([]float32(nil), hidden.Data[(tokens-1)*width:]...)
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

func meanPoolNormalized(hidden reference.Value) ([]float32, error) {
	vectors, err := poolEmbeddings(hidden, EmbeddingOptions{
		Pooling:   EmbeddingPoolingMean,
		Normalize: 2,
	})
	if err != nil {
		return nil, err
	}
	return vectors[0], nil
}
