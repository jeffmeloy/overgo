package model

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"overgo/internal/tensor"
)

// SequenceResidualGraphWeights: convolutional or attention residual inputs.
type SequenceResidualGraphWeights struct {
	Norm1, Norm1Bias, Conv1, Conv1Bias *tensor.Tensor
	Norm2, Norm2Bias, Conv2, Conv2Bias *tensor.Tensor
	AttentionNorm, AttentionNormBias   *tensor.Tensor
	AttentionQ, AttentionQBias         *tensor.Tensor
	AttentionK, AttentionKBias         *tensor.Tensor
	AttentionV, AttentionVBias         *tensor.Tensor
	AttentionOutput, AttentionOutBias  *tensor.Tensor
}

// SequenceConvGraphWeights: depthwise/pointwise residual inputs.
type SequenceConvGraphWeights struct {
	Depthwise, DepthwiseBias   *tensor.Tensor
	Norm, NormBias             *tensor.Tensor
	Pointwise1, Pointwise1Bias *tensor.Tensor
	Pointwise2, Pointwise2Bias *tensor.Tensor
	Gamma                      *tensor.Tensor
}

// SequenceOutputGraphWeights: complete compiled sequence-output inputs.
type SequenceOutputGraphWeights struct {
	InputConv, InputConvBias   *tensor.Tensor
	Residual                   []SequenceResidualGraphWeights
	TokenNorm, TokenNormBias   *tensor.Tensor
	Convolution                []SequenceConvGraphWeights
	OutputNorm, OutputNormBias *tensor.Tensor
	Output, OutputBias         *tensor.Tensor
}

type sequenceResidualOperator uint8

const (
	sequenceResidualConvolution sequenceResidualOperator = iota
	sequenceResidualAttention
	sequenceResidualNormalization
)

var wavTokenizerResidualProgram = [...]sequenceResidualOperator{
	sequenceResidualConvolution,
	sequenceResidualConvolution,
	sequenceResidualAttention,
	sequenceResidualConvolution,
	sequenceResidualConvolution,
	sequenceResidualNormalization,
}

// SequenceOutputProgram: ordered sequence terminal math.
type SequenceOutputProgram struct {
	residuals                  []sequenceResidualOperator
	convolutions, groupCount   uint32
	embedding, positionWidth   uint64
	groupEpsilon, layerEpsilon float32
}

// Build executes the compiled sequence-output program.
func (p SequenceOutputProgram) Build(
	builder *tensor.Builder,
	embeddings *tensor.Tensor,
	weights SequenceOutputGraphWeights,
) (*tensor.Tensor, error) {
	if builder == nil || embeddings == nil {
		return nil, errors.New("sequence-output input is nil")
	}
	if p.embedding == 0 || embeddings.Shape.Rank != 2 || embeddings.Shape.Dims[0] != p.embedding {
		return nil, errors.New("sequence-output embedding shape is incompatible")
	}
	if len(weights.Residual) != len(p.residuals) || len(weights.Convolution) != int(p.convolutions) {
		return nil, errors.New("sequence-output graph layer count is incompatible")
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
	for block, layer := range weights.Residual {
		residual := current
		scope := fmt.Sprintf("residual block %d", block)
		switch p.residuals[block] {
		case sequenceResidualConvolution:
			if err := (graphWeights{
				requireGraphWeight("norm-1", layer.Norm1),
				requireGraphWeight("norm-1 bias", layer.Norm1Bias),
				requireGraphWeight("convolution-1", layer.Conv1),
				requireGraphWeight("convolution-1 bias", layer.Conv1Bias),
				requireGraphWeight("norm-2", layer.Norm2),
				requireGraphWeight("norm-2 bias", layer.Norm2Bias),
				requireGraphWeight("convolution-2", layer.Conv2),
				requireGraphWeight("convolution-2 bias", layer.Conv2Bias),
			}).validate("sequence-output " + scope); err != nil {
				return nil, err
			}
			current = builder.GroupNorm(current, layer.Norm1, layer.Norm1Bias, p.groupCount, p.groupEpsilon)
			current = builder.SiLU(current)
			current = builder.Conv1DSame(current, layer.Conv1, layer.Conv1Bias, false)
			current = builder.GroupNorm(current, layer.Norm2, layer.Norm2Bias, p.groupCount, p.groupEpsilon)
			current = builder.SiLU(current)
			current = builder.Conv1DSame(current, layer.Conv2, layer.Conv2Bias, false)
			current = builder.Add(current, residual)
		case sequenceResidualAttention:
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
			}).validate("sequence-output " + scope); err != nil {
				return nil, err
			}
			current = builder.GroupNorm(current, layer.AttentionNorm, layer.AttentionNormBias, p.groupCount, p.groupEpsilon)
			tokens := current.Shape.Dims[1]
			width := p.positionWidth
			query := builder.Reshape(builder.Conv1DSame(current, layer.AttentionQ, layer.AttentionQBias, false), width, 1, tokens)
			key := builder.Reshape(builder.Conv1DSame(current, layer.AttentionK, layer.AttentionKBias, false), width, 1, tokens)
			value := builder.Reshape(builder.Conv1DSame(current, layer.AttentionV, layer.AttentionVBias, false), width, 1, tokens)
			current = builder.AttentionWithOptions(query, key, value, tensor.AttentionOptions{
				Scale: float32(1 / math.Sqrt(float64(width))),
			})
			current = builder.Reshape(current, width, tokens)
			current = builder.Conv1DSame(current, layer.AttentionOutput, layer.AttentionOutBias, false)
			current = builder.Add(current, residual)
		case sequenceResidualNormalization:
			if err := (graphWeights{
				requireGraphWeight("norm", layer.AttentionNorm),
				requireGraphWeight("norm bias", layer.AttentionNormBias),
			}).validate("sequence-output " + scope); err != nil {
				return nil, err
			}
			current = builder.GroupNorm(current, layer.AttentionNorm, layer.AttentionNormBias, p.groupCount, p.groupEpsilon)
		default:
			return nil, fmt.Errorf("sequence-output residual block %d is unsupported", block)
		}
	}

	current = builder.AffineLayerNorm(current, weights.TokenNorm, weights.TokenNormBias, p.layerEpsilon)
	for block, layer := range weights.Convolution {
		scope := fmt.Sprintf("convolution block %d", block)
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
		}).validate("sequence-output " + scope); err != nil {
			return nil, err
		}
		residual := current
		current = builder.Conv1DSame(current, layer.Depthwise, layer.DepthwiseBias, true)
		current = builder.AffineLayerNorm(current, layer.Norm, layer.NormBias, p.layerEpsilon)
		current = builder.Add(builder.MulMat(layer.Pointwise1, current), layer.Pointwise1Bias)
		current = builder.GELU(current)
		current = builder.Add(builder.MulMat(layer.Pointwise2, current), layer.Pointwise2Bias)
		current = builder.Multiply(current, layer.Gamma)
		current = builder.Add(current, residual)
	}
	current = builder.AffineLayerNorm(current, weights.OutputNorm, weights.OutputNormBias, p.layerEpsilon)
	current = builder.Add(builder.MulMat(weights.Output, current), weights.OutputBias)
	if err := builder.Err(); err != nil {
		return nil, err
	}
	return current, nil
}

func compileSequenceOutputProgram(spec Spec, profile ArchitectureProfile) SequenceOutputProgram {
	if profile.Forward != ForwardWavTokenizer {
		return SequenceOutputProgram{}
	}
	return SequenceOutputProgram{
		residuals: slices.Clone(wavTokenizerResidualProgram[:]), convolutions: spec.ConvNextBlockCount,
		embedding: uint64(spec.EmbeddingLength), positionWidth: uint64(spec.PosNetEmbeddingLength),
		groupCount: spec.GroupNormGroups, groupEpsilon: spec.GroupNormEpsilon,
		layerEpsilon: spec.LayerNormEpsilon,
	}
}
