package model

import (
	"errors"

	"overgo/internal/tensor"
)

// CacheProjectionProgram: compiled K/V projection, normalization, and position math.
type CacheProjectionProgram struct {
	width, key, value, heads               uint64
	epsilon, frequencyBase, frequencyScale float32
	rotaryDimensions                       uint32
}

// Build executes the compiled cache-projection program.
func (p CacheProjectionProgram) Build(
	builder *tensor.Builder,
	fused *tensor.Tensor,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue *tensor.Tensor,
) (*tensor.Tensor, *tensor.Tensor, error) {
	if builder == nil || fused == nil || weights.AttentionK == nil || weights.AttentionV == nil || weights.AttentionKNorm == nil {
		return nil, nil, errors.New("compiled cache projection input is nil")
	}
	tokens, validInput := tensor.MatrixRows(fused.Shape, p.width)
	if !validInput || uint64(len(positions)) != tokens || (pastKey == nil) != (pastValue == nil) {
		return nil, nil, errors.New("compiled cache projection shape is incompatible")
	}
	key := builder.Reshape(
		builder.MulMat(weights.AttentionK, fused),
		p.key, p.heads, tokens,
	)
	value := builder.Reshape(
		builder.MulMat(weights.AttentionV, fused),
		p.value, p.heads, tokens,
	)
	key = builder.WeightedRMSNorm(key, weights.AttentionKNorm, p.epsilon)
	key = builder.RoPEWithOptions(key, tensor.RoPEOptions{
		Layout: tensor.RoPELayoutNeoX, Positions: positions,
		RotaryDimensions: p.rotaryDimensions, FrequencyBase: p.frequencyBase,
		FrequencyScale: p.frequencyScale,
	})
	if pastKey != nil {
		keyTokens, keyAxis, validKey := tensor.TrailingExtent(pastKey.Shape, p.key, p.heads)
		valueTokens, valueAxis, validValue := tensor.TrailingExtent(pastValue.Shape, p.value, p.heads)
		if !validKey || !validValue || keyTokens != valueTokens {
			return nil, nil, errors.New("compiled cache projection cache shape is incompatible")
		}
		key = builder.Concat(pastKey, key, keyAxis)
		value = builder.Concat(pastValue, value, valueAxis)
	}
	if err := builder.Err(); err != nil {
		return nil, nil, err
	}
	return key, value, nil
}

func compileCacheProjectionProgram(spec Spec, profile ArchitectureProfile) CacheProjectionProgram {
	if profile.Forward.Session != ForwardSessionPairedFeatures {
		return CacheProjectionProgram{}
	}
	return CacheProjectionProgram{
		width: uint64(spec.EmbeddingLength), key: uint64(spec.KeyLength),
		value: uint64(spec.ValueLength), heads: uint64(spec.HeadCountKV),
		epsilon: spec.RMSNormEpsilon, rotaryDimensions: spec.RopeDimensionCount,
		frequencyBase: spec.RopeFrequencyBase, frequencyScale: spec.ropeFrequencyScale(),
	}
}
