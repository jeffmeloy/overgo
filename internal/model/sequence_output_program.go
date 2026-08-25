package model

import (
	"errors"
	"fmt"
	"slices"

	"overgo/internal/hostmath"
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
	sequenceResidualOperatorCount
)

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
	if _, validInput := tensor.MatrixRows(embeddings.Shape, p.embedding); !validInput {
		return nil, errors.New("sequence-output embedding shape is incompatible")
	}
	if len(weights.Residual) != len(p.residuals) || len(weights.Convolution) != int(p.convolutions) {
		return nil, errors.New("sequence-output graph layer count is incompatible")
	}
	if err := (graphWeights{
		weights.InputConv,
		weights.InputConvBias,
		weights.TokenNorm,
		weights.TokenNormBias,
		weights.OutputNorm,
		weights.OutputNormBias,
		weights.Output,
		weights.OutputBias,
	}).validate("sequence-output decoder"); err != nil {
		return nil, err
	}

	current := builder.Conv1DSame(embeddings, weights.InputConv, weights.InputConvBias, false)
	for block, layer := range weights.Residual {
		residual := current
		scope := fmt.Sprintf("residual block %d", block)
		switch p.residuals[block] {
		case sequenceResidualConvolution:
			if err := (graphWeights{
				layer.Norm1,
				layer.Norm1Bias,
				layer.Conv1,
				layer.Conv1Bias,
				layer.Norm2,
				layer.Norm2Bias,
				layer.Conv2,
				layer.Conv2Bias,
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
				layer.AttentionNorm,
				layer.AttentionNormBias,
				layer.AttentionQ,
				layer.AttentionQBias,
				layer.AttentionK,
				layer.AttentionKBias,
				layer.AttentionV,
				layer.AttentionVBias,
				layer.AttentionOutput,
				layer.AttentionOutBias,
			}).validate("sequence-output " + scope); err != nil {
				return nil, err
			}
			current = builder.GroupNorm(current, layer.AttentionNorm, layer.AttentionNormBias, p.groupCount, p.groupEpsilon)
			width := p.positionWidth
			tokens, validInput := tensor.MatrixRows(current.Shape, width)
			if !validInput {
				return nil, errors.New("sequence-output attention shape is incompatible")
			}
			query := builder.Reshape(builder.Conv1DSame(current, layer.AttentionQ, layer.AttentionQBias, false), width, tensor.SingletonExtent, tokens)
			key := builder.Reshape(builder.Conv1DSame(current, layer.AttentionK, layer.AttentionKBias, false), width, tensor.SingletonExtent, tokens)
			value := builder.Reshape(builder.Conv1DSame(current, layer.AttentionV, layer.AttentionVBias, false), width, tensor.SingletonExtent, tokens)
			current = builder.AttentionWithOptions(query, key, value, tensor.AttentionOptions{
				Scale: hostmath.InvSqrt32(width),
			})
			current = builder.Reshape(current, width, tokens)
			current = builder.Conv1DSame(current, layer.AttentionOutput, layer.AttentionOutBias, false)
			current = builder.Add(current, residual)
		case sequenceResidualNormalization:
			if err := (graphWeights{
				layer.AttentionNorm,
				layer.AttentionNormBias,
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
			layer.Depthwise,
			layer.DepthwiseBias,
			layer.Norm,
			layer.NormBias,
			layer.Pointwise1,
			layer.Pointwise1Bias,
			layer.Pointwise2,
			layer.Pointwise2Bias,
			layer.Gamma,
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
	if profile.Forward.Operation != ForwardOperationAudioTokens {
		return SequenceOutputProgram{}
	}
	return SequenceOutputProgram{
		residuals: slices.Clone(profile.Forward.SequenceResiduals), convolutions: spec.ConvNextBlockCount,
		embedding: uint64(spec.EmbeddingLength), positionWidth: uint64(spec.PosNetEmbeddingLength),
		groupCount: spec.GroupNormGroups, groupEpsilon: spec.GroupNormEpsilon,
		layerEpsilon: spec.LayerNormEpsilon,
	}
}
