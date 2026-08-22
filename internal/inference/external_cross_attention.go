package inference

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/composition"
	"overgo/internal/hostmath"
	"overgo/internal/model"
	"overgo/internal/recipe"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// ExternalCrossAttentionDefinition is the composition-owned immutable adapter
// and seam authority retained as an alias for low-level compiler callers.
type ExternalCrossAttentionDefinition = composition.ExternalCrossAttentionDefinition

// ExternalCrossAttentionCompiler admits seams from an already compiled target
// model plan. Its zero value has no ambient model authority.
type ExternalCrossAttentionCompiler struct{}

// OpenExternalCrossAttention is the production construction path. It resolves
// active recipe authority before admitting the runtime program.
func OpenExternalCrossAttention(
	ctx context.Context,
	reader artifact.Reader,
	source, target artifact.ID,
	task recipe.Task,
	targetPlan model.ModelPlan,
) (ExternalCrossAttentionProgram, error) {
	plan, err := composition.CompileExternalCrossAttentionPlan(
		ctx, reader, source, target, task, targetPlan,
	)
	if err != nil {
		return ExternalCrossAttentionProgram{}, err
	}
	return (ExternalCrossAttentionCompiler{}).CompilePlan(plan)
}

// ExternalCrossAttentionProgram is the immutable layer admission and geometry
// contract for an interleaved adapter.
type ExternalCrossAttentionProgram struct {
	identity       artifact.ID
	definition     ExternalCrossAttentionDefinition
	targetChannels uint64
	headChannels   uint64
	layers         map[uint32]struct{}
}

// ExternalCrossAttentionWeights are the adapter-owned projections and scalar
// residual gate. A zero gate is an exact identity operation.
type ExternalCrossAttentionWeights struct {
	Query  reference.Value
	Key    reference.Value
	Value  reference.Value
	Output reference.Value
	Gate   float32
}

// ExternalCrossAttentionInput supplies one target seam and either fresh source
// memory or a separately typed external cache.
type ExternalCrossAttentionInput struct {
	Target         reference.Value
	TargetTokens   uint64
	Source         *reference.Value
	SourceIdentity artifact.ID
	SourceTokens   uint64
	Mask           *reference.Value
}

// ExternalCrossAttentionCache is deliberately disjoint from KVCache: it owns
// fixed external K/V projections and cannot enter target self-attention state.
type ExternalCrossAttentionCache struct {
	Program              artifact.ID
	Source               artifact.ID
	SourceRepresentation artifact.ID
	Tokens               uint64
	Key                  reference.Value
	Value                reference.Value
}

var (
	externalMatrix  = reference.Value.IsMatrixWidth
	externalExtent  = reference.Value.Rows
	externalDeclare = tensor.NewShape
	externalReframe = (*tensor.Builder).Reshape
)

// Compile validates exact model identities, adapter identity, bounded head
// geometry, and every requested target layer-program seam.
func (ExternalCrossAttentionCompiler) Compile(
	target artifact.ID,
	plan model.ModelPlan,
	definition ExternalCrossAttentionDefinition,
) (ExternalCrossAttentionProgram, error) {
	if !plan.Compiled() || target.Kind() != artifact.KindModel || definition.Target != target ||
		definition.Source.Kind() != artifact.KindModel || definition.Source == target ||
		definition.Adapter.Kind() != artifact.KindAdapter ||
		!checked.Nonzero(definition.SourceChannels) || !checked.Nonzero(definition.HeadCount) ||
		!checked.Nonzero(definition.SourceTokenLimit) || !checked.Nonzero(len(definition.Layers)) {
		return ExternalCrossAttentionProgram{}, errors.New("inference: external cross-attention identity or bounds are invalid")
	}
	targetChannels := uint64(plan.Spec().EmbeddingLength)
	headChannels, divisible := checked.DivExact64(targetChannels, definition.HeadCount)
	if !checked.Nonzero(targetChannels) || !divisible {
		return ExternalCrossAttentionProgram{}, errors.New("inference: external cross-attention head geometry is invalid")
	}
	ordered := slices.Clone(definition.Layers)
	slices.Sort(ordered)
	unique := slices.Compact(ordered)
	if len(unique) != len(definition.Layers) {
		return ExternalCrossAttentionProgram{}, errors.New("inference: external cross-attention layer seam is duplicated")
	}
	layers := make(map[uint32]struct{}, len(unique))
	for _, layer := range unique {
		program, err := plan.LayerProgram(layer)
		if err != nil {
			return ExternalCrossAttentionProgram{}, fmt.Errorf("inference: external layer seam %d: %w", layer, err)
		}
		if program.Layer().Cache == model.CacheCrossAttention {
			return ExternalCrossAttentionProgram{}, errors.New("inference: external seam conflicts with model-owned cross-attention cache")
		}
		layers[layer] = struct{}{}
	}
	definition.Layers = unique
	identity, err := artifact.JSONID(artifact.KindProfile, definition)
	if err != nil {
		return ExternalCrossAttentionProgram{}, err
	}
	return ExternalCrossAttentionProgram{
		identity: identity, definition: definition,
		targetChannels: targetChannels, headChannels: headChannels, layers: layers,
	}, nil
}

// CompilePlan admits only a recipe-compiled external-attention plan. Its cache
// identity remains in the composition-owned external domain.
func (ExternalCrossAttentionCompiler) CompilePlan(
	plan composition.ExternalCrossAttentionPlan,
) (ExternalCrossAttentionProgram, error) {
	if err := plan.ValidateIdentity(); err != nil {
		return ExternalCrossAttentionProgram{}, err
	}
	layers := make(map[uint32]struct{}, len(plan.Layers))
	for _, layer := range plan.Layers {
		layers[layer] = struct{}{}
	}
	return ExternalCrossAttentionProgram{
		identity: plan.CacheIdentity,
		definition: ExternalCrossAttentionDefinition{
			Version: artifact.InitialDocumentVersion,
			Target:  plan.Target, Source: plan.Source, Adapter: plan.Adapter,
			Layers: slices.Clone(plan.Layers), SourceChannels: plan.SourceChannels,
			HeadCount: plan.HeadCount, SourceTokenLimit: plan.SourceTokenLimit,
		},
		targetChannels: plan.TargetChannels, headChannels: plan.HeadChannels, layers: layers,
	}, nil
}

// Identity returns the exact adapter/seam program identity used by its cache.
func (program ExternalCrossAttentionProgram) Identity() artifact.ID { return program.identity }

// Apply executes one admitted post-layer seam. External K/V is returned only
// through ExternalCrossAttentionCache; target KVCache is outside this API.
func (program ExternalCrossAttentionProgram) Apply(
	ctx context.Context,
	layer uint32,
	input ExternalCrossAttentionInput,
	weights ExternalCrossAttentionWeights,
	cache *ExternalCrossAttentionCache,
) (reference.Value, ExternalCrossAttentionCache, error) {
	if ctx == nil || !program.identity.Valid() {
		return reference.Value{}, ExternalCrossAttentionCache{}, errors.New("inference: external cross-attention program is absent")
	}
	if err := ctx.Err(); err != nil {
		return reference.Value{}, ExternalCrossAttentionCache{}, err
	}
	if _, admitted := program.layers[layer]; !admitted {
		return reference.Value{}, ExternalCrossAttentionCache{}, errors.New("inference: target layer is not an external cross-attention seam")
	}
	if err := validateExternalMatrix(input.Target, program.targetChannels, input.TargetTokens); err != nil {
		return reference.Value{}, ExternalCrossAttentionCache{}, fmt.Errorf("inference: target seam: %w", err)
	}
	if err := program.validateWeights(weights); err != nil {
		return reference.Value{}, ExternalCrossAttentionCache{}, err
	}
	builder := tensor.NewBuilder()
	feeds := make(map[*tensor.Tensor]reference.Value)
	bind := func(name string, value reference.Value, first, second uint64) (*tensor.Tensor, error) {
		shape, err := externalDeclare(first, second)
		if err != nil {
			return nil, err
		}
		node := builder.Input(name, dtype.F32, shape)
		feeds[node] = value
		return node, nil
	}
	targetNode, err := bind("external-cross.target", input.Target, program.targetChannels, input.TargetTokens)
	if err != nil {
		return reference.Value{}, ExternalCrossAttentionCache{}, err
	}
	queryWeight, err := bind("external-cross.query", weights.Query, program.targetChannels, program.targetChannels)
	if err != nil {
		return reference.Value{}, ExternalCrossAttentionCache{}, err
	}
	keyWeight, err := bind("external-cross.key", weights.Key, program.definition.SourceChannels, program.targetChannels)
	if err != nil {
		return reference.Value{}, ExternalCrossAttentionCache{}, err
	}
	valueWeight, err := bind("external-cross.value", weights.Value, program.definition.SourceChannels, program.targetChannels)
	if err != nil {
		return reference.Value{}, ExternalCrossAttentionCache{}, err
	}
	outputWeight, err := bind("external-cross.output", weights.Output, program.targetChannels, program.targetChannels)
	if err != nil {
		return reference.Value{}, ExternalCrossAttentionCache{}, err
	}
	var keyNode, valueNode *tensor.Tensor
	external := ExternalCrossAttentionCache{
		Program: program.identity, Source: program.definition.Source,
		SourceRepresentation: input.SourceIdentity,
	}
	if cache == nil {
		if input.Source == nil || !input.SourceIdentity.Valid() || !checked.Nonzero(input.SourceTokens) ||
			input.SourceTokens > program.definition.SourceTokenLimit {
			return reference.Value{}, ExternalCrossAttentionCache{}, errors.New("inference: external source is absent or exceeds its token limit")
		}
		if err := validateExternalMatrix(*input.Source, program.definition.SourceChannels, input.SourceTokens); err != nil {
			return reference.Value{}, ExternalCrossAttentionCache{}, fmt.Errorf("inference: external source: %w", err)
		}
		sourceNode, bindErr := bind(
			"external-cross.source", *input.Source,
			program.definition.SourceChannels, input.SourceTokens,
		)
		if bindErr != nil {
			return reference.Value{}, ExternalCrossAttentionCache{}, bindErr
		}
		keyNode = builder.MulMat(keyWeight, sourceNode)
		valueNode = builder.MulMat(valueWeight, sourceNode)
		external.Tokens = input.SourceTokens
	} else {
		if input.Source != nil || checked.Nonzero(input.SourceTokens) ||
			cache.Program != program.identity || cache.Source != program.definition.Source ||
			!input.SourceIdentity.Valid() || input.SourceIdentity != cache.SourceRepresentation ||
			!checked.Nonzero(cache.Tokens) || cache.Tokens > program.definition.SourceTokenLimit {
			return reference.Value{}, ExternalCrossAttentionCache{}, errors.New("inference: external cross-attention cache identity differs")
		}
		if err := validateExternalMatrix(cache.Key, program.targetChannels, cache.Tokens); err != nil {
			return reference.Value{}, ExternalCrossAttentionCache{}, err
		}
		if err := validateExternalMatrix(cache.Value, program.targetChannels, cache.Tokens); err != nil {
			return reference.Value{}, ExternalCrossAttentionCache{}, err
		}
		keyNode, err = bind("external-cross.cached-key", cache.Key, program.targetChannels, cache.Tokens)
		if err != nil {
			return reference.Value{}, ExternalCrossAttentionCache{}, err
		}
		valueNode, err = bind("external-cross.cached-value", cache.Value, program.targetChannels, cache.Tokens)
		if err != nil {
			return reference.Value{}, ExternalCrossAttentionCache{}, err
		}
		external = *cache
	}
	var maskNode *tensor.Tensor
	if input.Mask != nil {
		maskShape, maskErr := externalDeclare(external.Tokens)
		if maskErr != nil {
			return reference.Value{}, ExternalCrossAttentionCache{}, maskErr
		}
		maskNode = builder.Input("external-cross.mask", dtype.F32, maskShape)
		feeds[maskNode] = *input.Mask
	}
	query := externalReframe(
		builder, builder.MulMat(queryWeight, targetNode),
		program.headChannels, program.definition.HeadCount, input.TargetTokens,
	)
	key := externalReframe(
		builder, keyNode,
		program.headChannels, program.definition.HeadCount, external.Tokens,
	)
	value := externalReframe(
		builder, valueNode,
		program.headChannels, program.definition.HeadCount, external.Tokens,
	)
	attention := builder.AttentionWithOptions(query, key, value, tensor.AttentionOptions{
		KeyBias: maskNode, Scale: hostmath.InvSqrt32(program.headChannels),
	})
	attention = externalReframe(builder, attention, program.targetChannels, input.TargetTokens)
	output := builder.Add(targetNode, builder.Scale(builder.MulMat(outputWeight, attention), weights.Gate))
	if err := builder.Err(); err != nil {
		return reference.Value{}, ExternalCrossAttentionCache{}, err
	}
	outputs := []*tensor.Tensor{output}
	if cache == nil {
		outputs = append(outputs, keyNode, valueNode)
	}
	values, err := reference.Execute(outputs, feeds)
	if err != nil {
		return reference.Value{}, ExternalCrossAttentionCache{}, err
	}
	if cache == nil {
		external.Key, external.Value = values[keyNode], values[valueNode]
	}
	return values[output], external, nil
}

func (program ExternalCrossAttentionProgram) validateWeights(weights ExternalCrossAttentionWeights) error {
	checks := []struct {
		name          string
		value         reference.Value
		input, output uint64
	}{
		{"query", weights.Query, program.targetChannels, program.targetChannels},
		{"key", weights.Key, program.definition.SourceChannels, program.targetChannels},
		{"value", weights.Value, program.definition.SourceChannels, program.targetChannels},
		{"output", weights.Output, program.targetChannels, program.targetChannels},
	}
	for _, check := range checks {
		if err := validateExternalMatrix(check.value, check.input, check.output); err != nil {
			return fmt.Errorf("inference: external %s weight: %w", check.name, err)
		}
	}
	if !checked.Finite32(weights.Gate) {
		return errors.New("inference: external residual gate is not finite")
	}
	return nil
}

func validateExternalMatrix(value reference.Value, channels, tokens uint64) error {
	if !checked.Nonzero(channels) || !checked.Nonzero(tokens) || !externalMatrix(value, channels) {
		return errors.New("matrix geometry differs")
	}
	complete, err := externalExtent(value, tensor.FirstOffset, tokens)
	if err != nil || len(complete.Data) != len(value.Data) {
		return errors.New("matrix storage or token extent differs")
	}
	return nil
}
