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
	if p.width == 0 || fused.Shape.Rank != 2 || fused.Shape.Dims[0] != p.width ||
		len(positions) != int(fused.Shape.Dims[1]) || (pastKey == nil) != (pastValue == nil) {
		return nil, nil, errors.New("compiled cache projection shape is incompatible")
	}
	tokens := fused.Shape.Dims[1]
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
		key = builder.Concat(pastKey, key, 2)
		value = builder.Concat(pastValue, value, 2)
	}
	if err := builder.Err(); err != nil {
		return nil, nil, err
	}
	return key, value, nil
}

func compileCacheProjectionProgram(spec Spec, profile ArchitectureProfile) CacheProjectionProgram {
	if profile.Forward != ForwardDFlash {
		return CacheProjectionProgram{}
	}
	frequencyScale := float32(1)
	if (spec.RopeScalingType == "linear" || spec.RopeScalingType == "yarn") && spec.RopeScalingFactor > 0 {
		frequencyScale = 1 / spec.RopeScalingFactor
	}
	return CacheProjectionProgram{
		width: uint64(spec.EmbeddingLength), key: uint64(spec.KeyLength),
		value: uint64(spec.ValueLength), heads: uint64(spec.HeadCountKV),
		epsilon: spec.RMSNormEpsilon, rotaryDimensions: spec.RopeDimensionCount,
		frequencyBase: spec.RopeFrequencyBase, frequencyScale: frequencyScale,
	}
}
