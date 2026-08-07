package model

import (
	"errors"
	"fmt"
	"math"

	"overgo/internal/tensor"
)

// WavPosNetGraphWeights: PosNet graph inputs
type WavPosNetGraphWeights struct {
	Norm1, Norm1Bias, Conv1, Conv1Bias *tensor.Tensor
	Norm2, Norm2Bias, Conv2, Conv2Bias *tensor.Tensor
	AttentionNorm, AttentionNormBias   *tensor.Tensor
	AttentionQ, AttentionQBias         *tensor.Tensor
	AttentionK, AttentionKBias         *tensor.Tensor
	AttentionV, AttentionVBias         *tensor.Tensor
	AttentionOutput, AttentionOutBias  *tensor.Tensor
}

// WavConvNextGraphWeights: ConvNeXt graph inputs
type WavConvNextGraphWeights struct {
	Depthwise, DepthwiseBias   *tensor.Tensor
	Norm, NormBias             *tensor.Tensor
	Pointwise1, Pointwise1Bias *tensor.Tensor
	Pointwise2, Pointwise2Bias *tensor.Tensor
	Gamma                      *tensor.Tensor
}

// WavTokenizerGraphWeights: complete decoder graph inputs
type WavTokenizerGraphWeights struct {
	InputConv, InputConvBias   *tensor.Tensor
	PosNet                     []WavPosNetGraphWeights
	TokenNorm, TokenNormBias   *tensor.Tensor
	ConvNext                   []WavConvNextGraphWeights
	OutputNorm, OutputNormBias *tensor.Tensor
	Output, OutputBias         *tensor.Tensor
}

// BuildWavTokenizerDecoder: token IDs to audio-feature frames
func BuildWavTokenizerDecoder(
	builder *tensor.Builder,
	embeddings *tensor.Tensor,
	spec Spec,
	weights WavTokenizerGraphWeights,
) (*tensor.Tensor, error) {
	if builder == nil || embeddings == nil {
		return nil, errors.New("WavTokenizer decoder input is nil")
	}
	if spec.Profile().Forward != ForwardWavTokenizer {
		return nil, errors.New("WavTokenizer decoder requires WavTokenizer forward policy")
	}
	if embeddings.Shape.Rank != 2 || embeddings.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return nil, errors.New("WavTokenizer decoder embedding shape is incompatible")
	}
	if len(weights.PosNet) != int(spec.PosNetBlockCount) || len(weights.ConvNext) != int(spec.ConvNextBlockCount) {
		return nil, errors.New("WavTokenizer decoder graph layer count is incompatible")
	}
	if err := (graphWeights{
		requireGraphWeight("input convolution", weights.InputConv),
		requireGraphWeight("input convolution bias", weights.InputConvBias),
		requireGraphWeight("token norm", weights.TokenNorm),
		requireGraphWeight("token norm bias", weights.TokenNormBias),
		requireGraphWeight("output norm", weights.OutputNorm),
		requireGraphWeight("output norm bias", weights.OutputNormBias),
		requireGraphWeight("output", weights.Output),
		requireGraphWeight("output bias", weights.OutputBias),
	}).validate("WavTokenizer decoder"); err != nil {
		return nil, err
	}

	current := builder.Conv1DSame(embeddings, weights.InputConv, weights.InputConvBias, false)
	for block, layer := range weights.PosNet {
		residual := current
		scope := fmt.Sprintf("PosNet block %d", block)
		switch block {
		case 0, 1, 3, 4:
			if err := (graphWeights{
				requireGraphWeight("norm-1", layer.Norm1),
				requireGraphWeight("norm-1 bias", layer.Norm1Bias),
				requireGraphWeight("convolution-1", layer.Conv1),
				requireGraphWeight("convolution-1 bias", layer.Conv1Bias),
				requireGraphWeight("norm-2", layer.Norm2),
				requireGraphWeight("norm-2 bias", layer.Norm2Bias),
				requireGraphWeight("convolution-2", layer.Conv2),
				requireGraphWeight("convolution-2 bias", layer.Conv2Bias),
			}).validate("WavTokenizer " + scope); err != nil {
				return nil, err
			}
			current = builder.GroupNorm(current, layer.Norm1, layer.Norm1Bias, spec.GroupNormGroups, spec.GroupNormEpsilon)
			current = builder.SiLU(current)
			current = builder.Conv1DSame(current, layer.Conv1, layer.Conv1Bias, false)
			current = builder.GroupNorm(current, layer.Norm2, layer.Norm2Bias, spec.GroupNormGroups, spec.GroupNormEpsilon)
			current = builder.SiLU(current)
			current = builder.Conv1DSame(current, layer.Conv2, layer.Conv2Bias, false)
			current = builder.Add(current, residual)
		case 2:
			if err := (graphWeights{
				requireGraphWeight("attention norm", layer.AttentionNorm),
				requireGraphWeight("attention norm bias", layer.AttentionNormBias),
				requireGraphWeight("attention Q", layer.AttentionQ),
				requireGraphWeight("attention Q bias", layer.AttentionQBias),
				requireGraphWeight("attention K", layer.AttentionK),
				requireGraphWeight("attention K bias", layer.AttentionKBias),
				requireGraphWeight("attention V", layer.AttentionV),
				requireGraphWeight("attention V bias", layer.AttentionVBias),
				requireGraphWeight("attention output", layer.AttentionOutput),
				requireGraphWeight("attention output bias", layer.AttentionOutBias),
			}).validate("WavTokenizer " + scope); err != nil {
				return nil, err
			}
			current = builder.GroupNorm(current, layer.AttentionNorm, layer.AttentionNormBias, spec.GroupNormGroups, spec.GroupNormEpsilon)
			tokens := current.Shape.Dims[1]
			width := uint64(spec.PosNetEmbeddingLength)
			query := builder.Reshape(builder.Conv1DSame(current, layer.AttentionQ, layer.AttentionQBias, false), width, 1, tokens)
			key := builder.Reshape(builder.Conv1DSame(current, layer.AttentionK, layer.AttentionKBias, false), width, 1, tokens)
			value := builder.Reshape(builder.Conv1DSame(current, layer.AttentionV, layer.AttentionVBias, false), width, 1, tokens)
			current = builder.AttentionWithOptions(query, key, value, tensor.AttentionOptions{
				Scale: float32(1 / math.Sqrt(float64(width))),
			})
			current = builder.Reshape(current, width, tokens)
			current = builder.Conv1DSame(current, layer.AttentionOutput, layer.AttentionOutBias, false)
			current = builder.Add(current, residual)
		case 5:
			if err := (graphWeights{
				requireGraphWeight("norm", layer.AttentionNorm),
				requireGraphWeight("norm bias", layer.AttentionNormBias),
			}).validate("WavTokenizer " + scope); err != nil {
				return nil, err
			}
			current = builder.GroupNorm(current, layer.AttentionNorm, layer.AttentionNormBias, spec.GroupNormGroups, spec.GroupNormEpsilon)
		default:
			return nil, fmt.Errorf("WavTokenizer PosNet block %d is unsupported", block)
		}
	}

	current = builder.AffineLayerNorm(current, weights.TokenNorm, weights.TokenNormBias, spec.LayerNormEpsilon)
	for block, layer := range weights.ConvNext {
		scope := fmt.Sprintf("ConvNeXt block %d", block)
		if err := (graphWeights{
			requireGraphWeight("depthwise convolution", layer.Depthwise),
			requireGraphWeight("depthwise convolution bias", layer.DepthwiseBias),
			requireGraphWeight("norm", layer.Norm),
			requireGraphWeight("norm bias", layer.NormBias),
			requireGraphWeight("pointwise-1", layer.Pointwise1),
			requireGraphWeight("pointwise-1 bias", layer.Pointwise1Bias),
			requireGraphWeight("pointwise-2", layer.Pointwise2),
			requireGraphWeight("pointwise-2 bias", layer.Pointwise2Bias),
			requireGraphWeight("gamma", layer.Gamma),
		}).validate("WavTokenizer " + scope); err != nil {
			return nil, err
		}
		residual := current
		current = builder.Conv1DSame(current, layer.Depthwise, layer.DepthwiseBias, true)
		current = builder.AffineLayerNorm(current, layer.Norm, layer.NormBias, spec.LayerNormEpsilon)
		current = builder.Add(builder.MulMat(layer.Pointwise1, current), layer.Pointwise1Bias)
		current = builder.GELU(current)
		current = builder.Add(builder.MulMat(layer.Pointwise2, current), layer.Pointwise2Bias)
		current = builder.Multiply(current, layer.Gamma)
		current = builder.Add(current, residual)
	}
	current = builder.AffineLayerNorm(current, weights.OutputNorm, weights.OutputNormBias, spec.LayerNormEpsilon)
	current = builder.Add(builder.MulMat(weights.Output, current), weights.OutputBias)
	if err := builder.Err(); err != nil {
		return nil, err
	}
	return current, nil
}
