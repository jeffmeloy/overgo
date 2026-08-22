// Package bridgegraph compiles sealed cross-model representation operators.
package bridgegraph

import (
	"errors"
	"fmt"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/hostmath"
	"overgo/internal/representation"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

type Operator string

const (
	OperatorLinear          Operator = "linear"
	OperatorMLPGELU         Operator = "mlp-gelu"
	OperatorAlignVocabulary Operator = "align-vocabulary"
	OperatorPerceiver       Operator = "perceiver-resampler"
	OperatorConditionalLoRA Operator = "conditional-lora"
)

type EmbeddingMode string

const (
	EmbeddingTied   EmbeddingMode = "tied"
	EmbeddingUntied EmbeddingMode = "untied"
)

// Definition binds one shape-changing operator to exact interface artifacts.
type Definition struct {
	Source           artifact.ID   `json:"source"`
	Target           artifact.ID   `json:"target"`
	Operator         Operator      `json:"operator"`
	Intermediate     uint64        `json:"intermediate,omitempty"`
	Bias             bool          `json:"bias,omitempty"`
	Vocabulary       artifact.ID   `json:"vocabulary,omitzero"`
	VocabularyHead   artifact.ID   `json:"vocabulary_head,omitzero"`
	TargetEmbedding  artifact.ID   `json:"target_embedding,omitzero"`
	VocabularySize   uint64        `json:"vocabulary_size,omitempty"`
	VocabularyLimit  uint64        `json:"vocabulary_limit,omitempty"`
	EmbeddingMode    EmbeddingMode `json:"embedding_mode,omitempty"`
	LatentCount      uint64        `json:"latent_count,omitempty"`
	HeadCount        uint64        `json:"head_count,omitempty"`
	SourceTokenLimit uint64        `json:"source_token_limit,omitempty"`
	LoRARank         uint64        `json:"lora_rank,omitempty"`
	ScaleLimit       float32       `json:"scale_limit,omitempty"`
}

// Valid reports whether the operator belongs to the sealed bridge vocabulary.
func (operator Operator) Valid() bool {
	switch operator {
	case OperatorLinear, OperatorMLPGELU, OperatorAlignVocabulary, OperatorPerceiver, OperatorConditionalLoRA:
		return true
	default:
		return false
	}
}

// Weights are graph nodes supplied by the bridge artifact loader.
type Weights struct {
	First      *tensor.Tensor
	FirstBias  *tensor.Tensor
	Second     *tensor.Tensor
	SecondBias *tensor.Tensor
	Embedding  *tensor.Tensor
	Latents    *tensor.Tensor
	Query      *tensor.Tensor
	Key        *tensor.Tensor
	Value      *tensor.Tensor
	Output     *tensor.Tensor
	Mask       *tensor.Tensor
}

// ConditionalLoRAWeights keep both low-rank bases fixed while a source
// projection produces one bounded scale per sequence position.
type ConditionalLoRAWeights struct {
	Conditioner *tensor.Tensor
	A           *tensor.Tensor
	B           *tensor.Tensor
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
	reframeTensor = (*tensor.Builder).Reshape
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
	switch definition.Operator {
	case OperatorLinear:
		if checked.Nonzero(definition.Intermediate) || vocabularyDefinitionPresent(definition) ||
			perceiverDefinitionPresent(definition) || conditionalLoRADefinitionPresent(definition) {
			return Program{}, errors.New("bridge graph: linear operator has an intermediate width")
		}
		if err := validatePreservedSequence(source, target, sourceMin, sourceMax, targetMin, targetMax); err != nil {
			return Program{}, err
		}
	case OperatorMLPGELU:
		if !checked.Nonzero(definition.Intermediate) || vocabularyDefinitionPresent(definition) ||
			perceiverDefinitionPresent(definition) || conditionalLoRADefinitionPresent(definition) {
			return Program{}, errors.New("bridge graph: MLP operator requires an intermediate width")
		}
		if err := validatePreservedSequence(source, target, sourceMin, sourceMax, targetMin, targetMax); err != nil {
			return Program{}, err
		}
	case OperatorAlignVocabulary:
		if perceiverDefinitionPresent(definition) || conditionalLoRADefinitionPresent(definition) {
			return Program{}, errors.New("bridge graph: vocabulary alignment has Perceiver settings")
		}
		if err := validatePreservedSequence(source, target, sourceMin, sourceMax, targetMin, targetMax); err != nil {
			return Program{}, err
		}
		if err := validateVocabularyDefinition(definition, target); err != nil {
			return Program{}, err
		}
	case OperatorPerceiver:
		if vocabularyDefinitionPresent(definition) || conditionalLoRADefinitionPresent(definition) {
			return Program{}, errors.New("bridge graph: Perceiver has vocabulary settings")
		}
		if err := validatePerceiverDefinition(definition, source, target, sourceMin, sourceMax, targetMin, targetMax, targetSize); err != nil {
			return Program{}, err
		}
	case OperatorConditionalLoRA:
		if vocabularyDefinitionPresent(definition) || perceiverDefinitionPresent(definition) {
			return Program{}, errors.New("bridge graph: conditional LoRA has unrelated operator settings")
		}
		if err := validatePreservedSequence(source, target, sourceMin, sourceMax, targetMin, targetMax); err != nil {
			return Program{}, err
		}
		if err := validateConditionalLoRADefinition(definition, targetSize); err != nil {
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
	if p.definition.Operator == OperatorPerceiver {
		return p.buildPerceiver(builder, input, weights, tokens)
	}
	if p.definition.Operator == OperatorConditionalLoRA {
		return nil, errors.New("bridge graph: conditional LoRA requires source and target inputs")
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

func perceiverDefinitionPresent(definition Definition) bool {
	return checked.Nonzero(definition.LatentCount) || checked.Nonzero(definition.HeadCount) ||
		checked.Nonzero(definition.SourceTokenLimit)
}

func conditionalLoRADefinitionPresent(definition Definition) bool {
	return checked.Nonzero(definition.LoRARank) || checked.Nonzero(definition.ScaleLimit)
}

func validateConditionalLoRADefinition(definition Definition, targetBasis uint64) error {
	if checked.Nonzero(definition.Intermediate) || !checked.Nonzero(definition.LoRARank) ||
		definition.LoRARank > targetBasis || !checked.PositiveFinite32(definition.ScaleLimit) {
		return errors.New("bridge graph: conditional LoRA rank or scale bound is invalid")
	}
	return nil
}

func validatePreservedSequence(
	source, target representation.Contract,
	sourceMin, sourceMax, targetMin, targetMax uint64,
) error {
	if targetMin > sourceMin || targetMax < sourceMax ||
		source.Sequence.Mask != target.Sequence.Mask ||
		source.Sequence.Padding != target.Sequence.Padding ||
		source.Sequence.Position != target.Sequence.Position ||
		len(source.Sequence.PositionAxes) != len(target.Sequence.PositionAxes) ||
		!slices.Equal(source.Sequence.SpecialTokens, target.Sequence.SpecialTokens) {
		return errors.New("bridge graph: sequence-preserving operator cannot change sequence semantics")
	}
	return nil
}

func validatePerceiverDefinition(
	definition Definition,
	source, target representation.Contract,
	sourceMin, sourceMax, targetMin, targetMax, targetChannels uint64,
) error {
	_, divisible := checked.DivExact64(targetChannels, definition.HeadCount)
	if checked.Nonzero(definition.Intermediate) || !checked.Nonzero(definition.LatentCount) ||
		!checked.Nonzero(definition.HeadCount) || !divisible ||
		!checked.Nonzero(definition.SourceTokenLimit) ||
		definition.SourceTokenLimit < sourceMin || definition.SourceTokenLimit > sourceMax ||
		targetMin != definition.LatentCount || targetMax != definition.LatentCount ||
		source.Sequence.Mask == representation.MaskNone ||
		target.Sequence.Mask != representation.MaskNone ||
		target.Sequence.Padding != representation.PaddingNone {
		return errors.New("bridge graph: Perceiver sequence or attention bounds are invalid")
	}
	return nil
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

func (p Program) buildPerceiver(
	builder *tensor.Builder,
	input *tensor.Tensor,
	weights Weights,
	tokens uint64,
) (*tensor.Tensor, error) {
	if weights.First != nil || weights.FirstBias != nil || weights.Second != nil ||
		weights.SecondBias != nil || weights.Embedding != nil || weights.Mask == nil {
		return nil, errors.New("bridge graph: Perceiver tensor set is incomplete or mixed")
	}
	if tokens > p.definition.SourceTokenLimit {
		return nil, errors.New("bridge graph: Perceiver source exceeds its token limit")
	}
	checks := []struct {
		name          string
		value         *tensor.Tensor
		input, output uint64
	}{
		{"latents", weights.Latents, p.targetSize, p.definition.LatentCount},
		{"query", weights.Query, p.targetSize, p.targetSize},
		{"key", weights.Key, p.sourceSize, p.targetSize},
		{"value", weights.Value, p.sourceSize, p.targetSize},
		{"output", weights.Output, p.targetSize, p.targetSize},
	}
	for _, check := range checks {
		if err := validateProjection(check.value, nil, check.input, check.output, false); err != nil {
			return nil, fmt.Errorf("bridge graph: Perceiver %s: %w", check.name, err)
		}
	}
	headChannels, _ := checked.DivExact64(p.targetSize, p.definition.HeadCount)
	source := applyNormalization(builder, input, p.source.Normalization)
	latents := weights.Latents
	query := reframeTensor(
		builder, builder.MulMat(weights.Query, latents),
		headChannels, p.definition.HeadCount, p.definition.LatentCount,
	)
	key := reframeTensor(
		builder, builder.MulMat(weights.Key, source),
		headChannels, p.definition.HeadCount, tokens,
	)
	value := reframeTensor(
		builder, builder.MulMat(weights.Value, source),
		headChannels, p.definition.HeadCount, tokens,
	)
	attention := builder.AttentionWithOptions(query, key, value, tensor.AttentionOptions{
		KeyBias: weights.Mask, Scale: hostmath.InvSqrt32(headChannels),
	})
	attention = reframeTensor(builder, attention, p.targetSize, p.definition.LatentCount)
	output := builder.Add(latents, builder.MulMat(weights.Output, attention))
	output = applyNormalization(builder, output, p.target.Normalization)
	if err := builder.Err(); err != nil {
		return nil, fmt.Errorf("bridge graph: build Perceiver: %w", err)
	}
	outputTokens, matrix := matrixExtents(output.Shape, p.targetSize)
	if output.Type != dtype.F32 || !matrix || outputTokens != p.definition.LatentCount {
		return nil, errors.New("bridge graph: Perceiver output differs from fixed target contract")
	}
	return output, nil
}

// BuildConditionalLoRA applies fixed target-space low-rank bases with a
// per-position scale bounded by tanh and the compiled positive limit.
func (p Program) BuildConditionalLoRA(
	builder *tensor.Builder,
	condition, target *tensor.Tensor,
	weights ConditionalLoRAWeights,
) (*tensor.Tensor, error) {
	if p.definition.Operator != OperatorConditionalLoRA || builder == nil || condition == nil || target == nil {
		return nil, errors.New("bridge graph: conditional LoRA program or input is absent")
	}
	tokens, sourceMatrix := matrixExtents(condition.Shape, p.sourceSize)
	targetTokens, targetMatrix := matrixExtents(target.Shape, p.targetSize)
	if condition.Type != dtype.F32 || target.Type != dtype.F32 || !sourceMatrix || !targetMatrix ||
		tokens != targetTokens || tokens < p.sourceMin || tokens > p.sourceMax {
		return nil, errors.New("bridge graph: conditional LoRA input differs from contracted matrices")
	}
	checks := []struct {
		name          string
		value         *tensor.Tensor
		input, output uint64
	}{
		{"conditioner", weights.Conditioner, p.sourceSize, tensor.SingletonExtent},
		{"A", weights.A, p.targetSize, p.definition.LoRARank},
		{"B", weights.B, p.definition.LoRARank, p.targetSize},
	}
	for _, check := range checks {
		if err := validateProjection(check.value, nil, check.input, check.output, false); err != nil {
			return nil, fmt.Errorf("bridge graph: conditional LoRA %s: %w", check.name, err)
		}
	}
	scale := builder.Scale(
		builder.Tanh(builder.MulMat(weights.Conditioner, condition)),
		p.definition.ScaleLimit,
	)
	delta := builder.MulMat(weights.B, builder.MulMat(weights.A, target))
	output := builder.Add(target, builder.Multiply(delta, scale))
	if err := builder.Err(); err != nil {
		return nil, fmt.Errorf("bridge graph: build conditional LoRA: %w", err)
	}
	outputTokens, matrix := matrixExtents(output.Shape, p.targetSize)
	if output.Type != dtype.F32 || !matrix || outputTokens != tokens {
		return nil, errors.New("bridge graph: conditional LoRA output differs from target contract")
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
