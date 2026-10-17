package model

import (
	"errors"

	"overgo/internal/hostmath"
	"overgo/internal/tensor"
)

// ngramGateFloor is the original Qwen4ExpTextPLELayer signed-root floor.
// It belongs to the model arithmetic contract and changes only with that contract.
const ngramGateFloor = float32(1e-6)

func buildNGramInjection(options blockDispatchOptions, input *tensor.Tensor) (DenseBlockResult, error) {
	c, s, w := options.Context, options.Spec, options.Weights
	n := s.NGram
	if n == nil || c.Sequences != tensor.SingletonExtent {
		return DenseBlockResult{}, errors.New("ngram injection requires one sequence and declared metadata")
	}
	past := c.PastStates[cacheStateNGramConvolution].Value
	if err := (graphWeights{input, c.PerLayerInput, c.NGramHistory, past, w.NGramKey, w.NGramValue, w.NGramKeyNorm, w.NGramQueryNorm, w.NGramConvNorm, w.NGramConv}).validate("ngram injection"); err != nil {
		return DenseBlockResult{}, err
	}
	b := c.Builder
	width, streams, tokens := uint64(s.EmbeddingLength), uint64(s.HyperConnectionCount), uint64(len(c.Positions))
	wide := width * streams
	history := uint64(n.ConvKernel-1) * uint64(n.Size)
	if input.Shape != tensor.MustShape(width, streams, tokens) || c.PerLayerInput.Shape != tensor.MustShape(width, tokens) ||
		past.Shape != tensor.MustShape(wide, history) || c.NGramHistory.Shape != tensor.MustShape(uint64(n.Size-1)) {
		return DenseBlockResult{}, errors.New("ngram injection input or history shape is invalid")
	}
	normalize := func(value, weight *tensor.Tensor) *tensor.Tensor {
		return b.Multiply(b.RMSNorm(b.Reshape(value, width, streams, tokens), s.RMSNormEpsilon), b.Reshape(weight, width, streams))
	}
	key := normalize(b.MulMat(w.NGramKey, c.PerLayerInput), w.NGramKeyNorm)
	query := normalize(input, w.NGramQueryNorm)
	gate := b.Scale(b.SumRows(b.Multiply(key, query)), hostmath.InvSqrt32(width))
	gate = b.Sigmoid(b.SignedSqrt(gate, ngramGateFloor))
	value := b.Reshape(b.MulMat(w.NGramValue, c.PerLayerInput), width, tensor.SingletonExtent, tokens)
	gated := b.Multiply(value, gate)
	normalized := b.Reshape(normalize(gated, w.NGramConvNorm), wide, tokens)
	full := b.Concat(past, normalized, tensor.SingletonExtent)
	var conv *tensor.Tensor
	for tap := uint64(0); tap < uint64(n.ConvKernel); tap++ {
		shifted := b.FlatSlice(full, tap*uint64(n.Size)*wide, wide, tokens)
		coefficient := b.Reshape(b.GroupSlice(w.NGramConv, tap, tensor.SingletonExtent, tensor.SingletonExtent, tensor.SingletonExtent), wide)
		term := b.Multiply(shifted, coefficient)
		if conv == nil {
			conv = term
		} else {
			conv = b.Add(conv, term)
		}
	}
	output := b.Add(input, b.Add(gated, b.Reshape(b.SiLU(conv), width, streams, tokens)))
	states := CacheStates[*tensor.Tensor]{
		CacheStateNGramHistory:     {Mode: CacheStateFixed, Value: c.NGramHistory},
		cacheStateNGramConvolution: {Mode: CacheStateFixed, Value: b.FlatSlice(full, tokens*wide, wide, history)},
	}
	return DenseBlockResult{Output: output, States: states}, b.Err()
}
