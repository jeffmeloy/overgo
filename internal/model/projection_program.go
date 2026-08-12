package model

import (
	"errors"
	"math"

	"overgo/internal/tensor"
)

// ProjectionRole: indexed auxiliary projection program.
type ProjectionRole uint8

const (
	ProjectionFeature ProjectionRole = iota
	ProjectionPairedInput
	ProjectionPairedOutput
	projectionRoleCount
)

type projectionInput uint8

const (
	projectionDirect projectionInput = iota
	projectionScaledPair
)

type projectionNorm uint8

const (
	projectionNormNone projectionNorm = iota
	projectionNormBefore
	projectionNormAfter
)

// ProjectionProgram: compiled input, normalization, and output math.
type ProjectionProgram struct {
	input   projectionInput
	norm    projectionNorm
	width   uint64
	scale   float32
	epsilon float32
	outputs uint8
}

// ProjectionOperands: indexed graph inputs and weights.
type ProjectionOperands struct {
	Input, Paired                     *tensor.Tensor
	Primary, Secondary, Normalization *tensor.Tensor
}

// ProjectionResult: primary and optional secondary projections.
type ProjectionResult struct {
	Primary, Secondary *tensor.Tensor
}

// Build executes the compiled projection program.
func (p ProjectionProgram) Build(
	builder *tensor.Builder,
	operands ProjectionOperands,
) (ProjectionResult, error) {
	if builder == nil || operands.Input == nil || operands.Primary == nil ||
		p.width == 0 || p.outputs == 0 || p.outputs > 2 || operands.Input.Shape.Rank != 2 ||
		operands.Input.Shape.Dims[0] != p.width {
		return ProjectionResult{}, errors.New("compiled projection is incompatible")
	}
	current := operands.Input
	if p.input == projectionScaledPair {
		if operands.Paired == nil || !current.Shape.Equal(operands.Paired.Shape) || p.scale <= 0 {
			return ProjectionResult{}, errors.New("compiled paired projection is incompatible")
		}
		current = builder.Concat(builder.Scale(current, p.scale), operands.Paired, 0)
	} else if operands.Paired != nil {
		return ProjectionResult{}, errors.New("compiled projection has an unexpected paired input")
	}
	if p.norm == projectionNormBefore {
		if operands.Normalization == nil {
			return ProjectionResult{}, errors.New("compiled projection normalization is absent")
		}
		current = builder.WeightedRMSNorm(current, operands.Normalization, p.epsilon)
	}
	result := ProjectionResult{Primary: builder.MulMat(operands.Primary, current)}
	if p.norm == projectionNormAfter {
		if operands.Normalization == nil {
			return ProjectionResult{}, errors.New("compiled projection normalization is absent")
		}
		result.Primary = builder.WeightedRMSNorm(result.Primary, operands.Normalization, p.epsilon)
	} else if p.norm == projectionNormNone && operands.Normalization != nil {
		return ProjectionResult{}, errors.New("compiled projection has an unexpected normalization")
	}
	if p.outputs == 2 {
		if operands.Secondary == nil {
			return ProjectionResult{}, errors.New("compiled secondary projection is absent")
		}
		result.Secondary = builder.MulMat(operands.Secondary, current)
	} else if operands.Secondary != nil {
		return ProjectionResult{}, errors.New("compiled projection has an unexpected secondary output")
	}
	return result, builder.Err()
}

func compileProjectionPrograms(spec Spec, profile ArchitectureProfile) [projectionRoleCount]ProjectionProgram {
	var programs [projectionRoleCount]ProjectionProgram
	switch profile.Forward.Session {
	case ForwardSessionFeatureDraft:
		programs[ProjectionFeature] = ProjectionProgram{
			width: uint64(eagle3TargetLayerCount) * uint64(spec.TargetHiddenSize), outputs: 1,
		}
	case ForwardSessionPairedFeatures:
		programs[ProjectionFeature] = ProjectionProgram{
			width: uint64(len(spec.TargetLayers)) * uint64(spec.EmbeddingLength),
			norm:  projectionNormAfter, epsilon: spec.RMSNormEpsilon, outputs: 1,
		}
	case ForwardSessionPairedProjection:
		programs[ProjectionPairedInput] = ProjectionProgram{
			input: projectionScaledPair, width: uint64(spec.TargetHiddenSize),
			scale: float32(math.Sqrt(float64(spec.TargetHiddenSize))), outputs: 1,
		}
		programs[ProjectionPairedOutput] = ProjectionProgram{
			width: uint64(spec.EmbeddingLength), norm: projectionNormBefore,
			epsilon: spec.RMSNormEpsilon, outputs: 2,
		}
	}
	return programs
}
