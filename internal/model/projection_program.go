package model

import (
	"errors"

	"overgo/internal/hostmath"
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

type projectionOutputs uint8

const (
	projectionNoOutputs projectionOutputs = iota
	projectionPrimary
	projectionPrimarySecondary
	projectionOutputsCount
)

func (o projectionOutputs) valid() bool { return o > projectionNoOutputs && o < projectionOutputsCount }

// ProjectionProgram: compiled input, normalization, and output math.
type ProjectionProgram struct {
	input   projectionInput
	norm    projectionNorm
	width   uint64
	scale   float32
	epsilon float32
	outputs projectionOutputs
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
	validInput := false
	if operands.Input != nil {
		_, validInput = tensor.MatrixRows(operands.Input.Shape, p.width)
	}
	if builder == nil || operands.Input == nil || operands.Primary == nil ||
		!p.outputs.valid() || !validInput {
		return ProjectionResult{}, errors.New("compiled projection is incompatible")
	}
	current := operands.Input
	if p.input == projectionScaledPair {
		if operands.Paired == nil || !current.Shape.Equal(operands.Paired.Shape) || !positiveFinite(p.scale) {
			return ProjectionResult{}, errors.New("compiled paired projection is incompatible")
		}
		current = builder.Concat(builder.Scale(current, p.scale), operands.Paired, tensor.FirstOffset)
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
	if p.outputs == projectionPrimarySecondary {
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
			width: uint64(len(spec.TargetLayers)) * uint64(spec.TargetHiddenSize), outputs: projectionPrimary,
		}
	case ForwardSessionPairedFeatures:
		programs[ProjectionFeature] = ProjectionProgram{
			width: uint64(len(spec.TargetLayers)) * uint64(spec.EmbeddingLength),
			norm:  projectionNormAfter, epsilon: spec.RMSNormEpsilon, outputs: projectionPrimary,
		}
	case ForwardSessionPairedProjection:
		programs[ProjectionPairedInput] = ProjectionProgram{
			input: projectionScaledPair, width: uint64(spec.TargetHiddenSize),
			scale: hostmath.Sqrt32(uint64(spec.TargetHiddenSize)), outputs: projectionPrimary,
		}
		programs[ProjectionPairedOutput] = ProjectionProgram{
			width: uint64(spec.EmbeddingLength), norm: projectionNormBefore,
			epsilon: spec.RMSNormEpsilon, outputs: projectionPrimarySecondary,
		}
	}
	return programs
}
