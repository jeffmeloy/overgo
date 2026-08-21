// Package bridgegraph compiles sealed cross-model representation operators.
package bridgegraph

import (
	"errors"
	"fmt"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/representation"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

type Operator string

const (
	OperatorLinear          Operator = "linear"
	OperatorMLPGELU         Operator = "mlp-gelu"
	OperatorAlignVocabulary Operator = "align-vocabulary"
)

type EmbeddingMode string

const (
	EmbeddingTied   EmbeddingMode = "tied"
	EmbeddingUntied EmbeddingMode = "untied"
)

// Definition binds one shape-changing operator to exact interface artifacts.
type Definition struct {
	Source          artifact.ID
	Target          artifact.ID
	Operator        Operator
	Intermediate    uint64
	Bias            bool
	Vocabulary      artifact.ID
	VocabularyHead  artifact.ID
	TargetEmbedding artifact.ID
	VocabularySize  uint64
	VocabularyLimit uint64
	EmbeddingMode   EmbeddingMode
}

// Weights are graph nodes supplied by the bridge artifact loader.
type Weights struct {
	First      *tensor.Tensor
	FirstBias  *tensor.Tensor
	Second     *tensor.Tensor
	SecondBias *tensor.Tensor
	Embedding  *tensor.Tensor
}

// Compiler is stateless; its zero value admits canonical contract bytes.
type Compiler struct{}

// Program is an immutable MLP bridge execution contract.
type Program struct {
	definition Definition
	source     representation.Contract
	target     representation.Contract
	sourceMin  uint64
	sourceMax  uint64
	sourceSize uint64
	targetSize uint64
}

// Boundary is the immutable identity and tap metadata consumed by an
// inference integration.
type Boundary struct {
	Contract artifact.ID
	Model    artifact.ID
	Tap      representation.TapPoint
	Layer    *uint32
	Channels uint64
}

var (
	matrixExtents = tensor.MatrixRows
	declareTensor = tensor.NewShape
)

// Boundaries returns the exact source and target interfaces admitted during
// compilation.
func (p Program) Boundaries() (Boundary, Boundary) {
	return programBoundary(p.source, p.sourceSize), programBoundary(p.target, p.targetSize)
}

func programBoundary(contract representation.Contract, width uint64) Boundary {
	boundary := Boundary{
		Contract: contract.ID, Model: contract.Producer.Model,
		Tap: contract.Producer.Tap, Channels: width,
	}
	if contract.Producer.Layer != nil {
		layer := *contract.Producer.Layer
		boundary.Layer = &layer
	}
	return boundary
}

// Compile validates contract identity, sequence preservation, and weight geometry.
func (Compiler) Compile(definition Definition, sourceContent, targetContent []byte) (Program, error) {
	source, err := representation.ParseContract(sourceContent)
	if err != nil {
		return Program{}, fmt.Errorf("bridge graph: parse source contract: %w", err)
	}
	target, err := representation.ParseContract(targetContent)
	if err != nil {
		return Program{}, fmt.Errorf("bridge graph: parse target contract: %w", err)
	}
	if definition.Source != source.ID || definition.Target != target.ID {
		return Program{}, errors.New("bridge graph: definition contract identity differs")
	}
	sourceSize, sourceMin, sourceMax, err := matrixContract(source)
	if err != nil {
		return Program{}, fmt.Errorf("bridge graph: source: %w", err)
	}
	targetSize, targetMin, targetMax, err := matrixContract(target)
	if err != nil {
		return Program{}, fmt.Errorf("bridge graph: target: %w", err)
	}
	if targetMin > sourceMin || targetMax < sourceMax ||
		source.Sequence.Mask != target.Sequence.Mask ||
		source.Sequence.Padding != target.Sequence.Padding ||
		source.Sequence.Position != target.Sequence.Position ||
		len(source.Sequence.PositionAxes) != len(target.Sequence.PositionAxes) ||
		!slices.Equal(source.Sequence.SpecialTokens, target.Sequence.SpecialTokens) {
		return Program{}, errors.New("bridge graph: linear operator cannot change sequence semantics")
	}
	switch definition.Operator {
	case OperatorLinear:
		if checked.Nonzero(definition.Intermediate) || vocabularyDefinitionPresent(definition) {
			return Program{}, errors.New("bridge graph: linear operator has an intermediate width")
		}
	case OperatorMLPGELU:
		if !checked.Nonzero(definition.Intermediate) || vocabularyDefinitionPresent(definition) {
			return Program{}, errors.New("bridge graph: MLP operator requires an intermediate width")
		}
	case OperatorAlignVocabulary:
		if err := validateVocabularyDefinition(definition, target); err != nil {
			return Program{}, err
		}
	default:
		return Program{}, errors.New("bridge graph: unsupported operator")
	}
	return Program{
		definition: definition, source: source, target: target,
		sourceMin: sourceMin, sourceMax: sourceMax,
		sourceSize: sourceSize, targetSize: targetSize,
	}, nil
}

// Build emits the validated bridge into an existing tensor graph.
func (p Program) Build(builder *tensor.Builder, input *tensor.Tensor, weights Weights) (*tensor.Tensor, error) {
	if builder == nil || input == nil {
		return nil, errors.New("bridge graph: builder or input is absent")
	}
	if err := builder.Err(); err != nil {
		return nil, fmt.Errorf("bridge graph: builder: %w", err)
	}
	tokens, matrix := matrixExtents(input.Shape, p.sourceSize)
	if input.Type != dtype.F32 || !matrix {
		return nil, errors.New("bridge graph: input differs from source matrix contract")
	}
	if tokens < p.sourceMin || tokens > p.sourceMax {
		return nil, errors.New("bridge graph: input sequence is outside source bounds")
	}
	if p.definition.Operator == OperatorAlignVocabulary {
		return p.buildVocabularyAlignment(builder, input, weights, tokens)
	}
	firstOutput := p.targetSize
	if p.definition.Operator == OperatorMLPGELU {
		firstOutput = p.definition.Intermediate
	}
	if err := validateProjection(weights.First, weights.FirstBias, p.sourceSize, firstOutput, p.definition.Bias); err != nil {
		return nil, fmt.Errorf("bridge graph: first projection: %w", err)
	}
	if p.definition.Operator == OperatorLinear {
		if weights.Second != nil || weights.SecondBias != nil || weights.Embedding != nil {
			return nil, errors.New("bridge graph: linear operator has second projection tensors")
		}
	} else if weights.Embedding != nil {
		return nil, errors.New("bridge graph: MLP operator has a vocabulary embedding tensor")
	} else if err := validateProjection(weights.Second, weights.SecondBias, p.definition.Intermediate, p.targetSize, p.definition.Bias); err != nil {
		return nil, fmt.Errorf("bridge graph: second projection: %w", err)
	}

	output := applyNormalization(builder, input, p.source.Normalization)
	output = builder.MulMat(weights.First, output)
	if weights.FirstBias != nil {
		output = builder.Add(output, weights.FirstBias)
	}
	if p.definition.Operator == OperatorMLPGELU {
		output = builder.GELU(output)
		output = builder.MulMat(weights.Second, output)
		if weights.SecondBias != nil {
			output = builder.Add(output, weights.SecondBias)
		}
	}
	output = applyNormalization(builder, output, p.target.Normalization)
	if err := builder.Err(); err != nil {
		return nil, fmt.Errorf("bridge graph: build: %w", err)
	}
	if output == nil {
		return nil, errors.New("bridge graph: compiled output is absent")
	}
	outputTokens, matrix := matrixExtents(output.Shape, p.targetSize)
	if output.Type != dtype.F32 || !matrix || outputTokens != tokens {
		return nil, errors.New("bridge graph: compiled output differs from target matrix contract")
	}
	return output, nil
}

func vocabularyDefinitionPresent(definition Definition) bool {
	return definition.Vocabulary.Valid() || definition.VocabularyHead.Valid() ||
		definition.TargetEmbedding.Valid() || checked.Nonzero(definition.VocabularySize) ||
		checked.Nonzero(definition.VocabularyLimit) || definition.EmbeddingMode != ""
}

func validateVocabularyDefinition(definition Definition, target representation.Contract) error {
	if checked.Nonzero(definition.Intermediate) ||
		definition.Vocabulary.Kind() != artifact.KindTokenizer ||
		definition.VocabularyHead.Kind() != artifact.KindTensorSet ||
		definition.TargetEmbedding.Kind() != artifact.KindTensorSet ||
		!checked.Nonzero(definition.VocabularySize) ||
		!checked.Nonzero(definition.VocabularyLimit) ||
		definition.VocabularySize > definition.VocabularyLimit {
		return errors.New("bridge graph: vocabulary alignment identity or bound is invalid")
	}
	if !contractAuthority(target, representation.AuthorityTokenizer, definition.Vocabulary) {
		return errors.New("bridge graph: target vocabulary authority differs")
	}
	switch definition.EmbeddingMode {
	case EmbeddingTied:
		if definition.VocabularyHead != definition.TargetEmbedding {
			return errors.New("bridge graph: tied vocabulary tensors differ")
		}
	case EmbeddingUntied:
		if definition.VocabularyHead == definition.TargetEmbedding {
			return errors.New("bridge graph: untied vocabulary tensors are identical")
		}
	default:
		return errors.New("bridge graph: vocabulary embedding mode is invalid")
	}
	return nil
}

func contractAuthority(contract representation.Contract, role representation.AuthorityRole, id artifact.ID) bool {
	for _, authority := range contract.Authorities {
		if authority.Role == role && authority.Artifact == id {
			return true
		}
	}
	return false
}

func (p Program) buildVocabularyAlignment(
	builder *tensor.Builder,
	input *tensor.Tensor,
	weights Weights,
	tokens uint64,
) (*tensor.Tensor, error) {
	if weights.Second != nil || weights.SecondBias != nil {
		return nil, errors.New("bridge graph: vocabulary alignment has an extra projection")
	}
	if err := validateProjection(
		weights.First, weights.FirstBias, p.sourceSize,
		p.definition.VocabularySize, p.definition.Bias,
	); err != nil {
		return nil, fmt.Errorf("bridge graph: vocabulary head: %w", err)
	}
	if err := validateProjection(
		weights.Embedding, nil, p.definition.VocabularySize, p.targetSize, false,
	); err != nil {
		return nil, fmt.Errorf("bridge graph: target embedding: %w", err)
	}
	output := applyNormalization(builder, input, p.source.Normalization)
	output = builder.MulMat(weights.First, output)
	if weights.FirstBias != nil {
		output = builder.Add(output, weights.FirstBias)
	}
	output = builder.Softmax(output)
	output = builder.MulMat(weights.Embedding, output)
	output = applyNormalization(builder, output, p.target.Normalization)
	if err := builder.Err(); err != nil {
		return nil, fmt.Errorf("bridge graph: build vocabulary alignment: %w", err)
	}
	outputTokens, matrix := matrixExtents(output.Shape, p.targetSize)
	if output.Type != dtype.F32 || !matrix || outputTokens != tokens {
		return nil, errors.New("bridge graph: vocabulary output differs from target matrix contract")
	}
	return output, nil
}

func matrixContract(contract representation.Contract) (width, minimum, maximum uint64, err error) {
	axes := contract.Tensor.Axes
	if contract.Tensor.DataType != dtype.F32 || len(axes) != tensor.PairedExtent ||
		axes[tensor.FirstOffset].Kind != representation.AxisChannel ||
		axes[tensor.SingletonExtent].Kind != contract.Sequence.Axis {
		return tensor.FirstOffset, tensor.FirstOffset, tensor.FirstOffset,
			errors.New("contract must be an F32 channel-by-sequence matrix")
	}
	width = axes[tensor.FirstOffset].Bounds.Extent
	minimum, maximum = axisRange(axes[tensor.SingletonExtent].Bounds)
	if !checked.Nonzero(width) || !checked.Nonzero(minimum) || maximum < minimum {
		return tensor.FirstOffset, tensor.FirstOffset, tensor.FirstOffset,
			errors.New("contract matrix bounds are invalid")
	}
	return width, minimum, maximum, nil
}

func axisRange(bounds representation.AxisBounds) (uint64, uint64) {
	if checked.Nonzero(bounds.Extent) {
		return bounds.Extent, bounds.Extent
	}
	return bounds.Minimum, bounds.Maximum
}

func validateProjection(weight, bias *tensor.Tensor, input, output uint64, withBias bool) error {
	weightShape, err := declareTensor(input, output)
	if err != nil {
		return err
	}
	if weight == nil || weight.Type != dtype.F32 || !weight.Shape.Equal(weightShape) {
		return errors.New("weight geometry differs")
	}
	if !withBias {
		if bias != nil {
			return errors.New("bias is disabled but present")
		}
		return nil
	}
	biasShape, err := declareTensor(output)
	if err != nil {
		return err
	}
	if bias == nil || bias.Type != dtype.F32 || !bias.Shape.Equal(biasShape) {
		return errors.New("bias geometry differs")
	}
	return nil
}

func applyNormalization(builder *tensor.Builder, input *tensor.Tensor, contract representation.NormalizationContract) *tensor.Tensor {
	switch contract.Kind {
	case representation.NormalizationRMS:
		return builder.RMSNorm(input, contract.Epsilon)
	case representation.NormalizationLayer:
		return builder.LayerNorm(input, contract.Epsilon)
	case representation.NormalizationL2:
		return builder.L2Norm(input, contract.Epsilon)
	default:
		return input
	}
}
