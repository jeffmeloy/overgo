package inference

import (
	"context"
	"errors"
	"math"
	"slices"
	"sync"

	"overgo/internal/artifact"
	"overgo/internal/bridgegraph"
	"overgo/internal/checked"
	"overgo/internal/composition"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/representation"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

// RepresentationSource captures one exact pre-layer model representation.
// Runner implements this port.
type RepresentationSource interface {
	ModelID() artifact.ID
	ExtractLayerInputs(context.Context, []tokenizer.TokenID, []int32) (reference.Value, error)
}

// RepresentationTarget admits a complete soft-token replacement at its
// embedding lookup boundary. Runner implements this port.
type RepresentationTarget interface {
	ModelID() artifact.ID
	ForwardWithEmbeddingOverrides(context.Context, []tokenizer.TokenID, []EmbeddingOverride) (reference.Value, error)
}

// RepresentationBridgeWeights are the host values bound to a compiled bridge
// graph. Optional bias and second-projection values remain nil when absent.
type RepresentationBridgeWeights struct {
	First      *reference.Value
	FirstBias  *reference.Value
	Second     *reference.Value
	SecondBias *reference.Value
}

// CompositionRuntimeResources resolves live sessions and bridge weights only
// after an active execution plan has fixed their immutable authorities.
type CompositionRuntimeResources interface {
	Source(context.Context, modelrecipe.ComponentSession) (RepresentationSource, error)
	Target(context.Context, modelrecipe.ComponentSession) (RepresentationTarget, error)
	BridgeWeights(context.Context, modelrecipe.ComponentSession, artifact.ID, composition.BridgeWeightAuthority) (RepresentationBridgeWeights, error)
}

// ProductionComposition is an active-recipe-owned embedding-injection
// runtime. Its bridge session cannot be constructed directly.
type ProductionComposition struct {
	plan                 composition.CompositionExecutionPlan
	session              representationBridgeSession
	cacheMu              sync.Mutex
	cachedSource         artifact.ID
	cachedRepresentation reference.Value
}

type representationBridgeSession struct {
	source  RepresentationSource
	target  RepresentationTarget
	program bridgegraph.Program
	weights RepresentationBridgeWeights
}

var (
	representationMatrix = reference.Value.IsMatrixWidth
	representationExtent = reference.Value.Rows
	compositionTensor    = tensor.NewShape
)

// OpenProductionComposition is the only embedding-injection construction
// path. It resolves the active recipe, recompiles exact contract seams, and
// admits resources solely against the resulting plan.
func OpenProductionComposition(
	ctx context.Context,
	reader artifact.Reader,
	source, target artifact.ID,
	task recipe.Task,
	resources CompositionRuntimeResources,
) (*ProductionComposition, error) {
	if resources == nil {
		return nil, errors.New("inference: composition runtime resources are absent")
	}
	plan, err := composition.CompileCompositionExecutionPlan(ctx, reader, source, target, task)
	if err != nil {
		return nil, err
	}
	if len(plan.BridgeDefinitions) != tensor.SingletonExtent ||
		len(plan.BridgeWeights) != tensor.SingletonExtent || len(plan.Sessions.Components) != tensor.PairedExtent {
		return nil, errors.New("inference: embedding injection requires one sealed bridge")
	}
	bridge, err := composition.LoadBridgeDefinition(ctx, reader, plan.BridgeDefinitions[tensor.FirstOffset])
	if err != nil || bridge.Weights != plan.BridgeWeights[tensor.FirstOffset] {
		return nil, errors.Join(err, errors.New("inference: bridge authority differs from execution plan"))
	}
	sourceContract, err := representation.LoadContract(ctx, reader, plan.SourceContract)
	if err != nil {
		return nil, err
	}
	targetContract, err := representation.LoadContract(ctx, reader, plan.TargetContract)
	if err != nil {
		return nil, err
	}
	sourceContent, err := sourceContract.Content()
	if err != nil {
		return nil, err
	}
	targetContent, err := targetContract.Content()
	if err != nil {
		return nil, err
	}
	program, err := (bridgegraph.Compiler{}).Compile(bridge.Graph, sourceContent.Data, targetContent.Data)
	if err != nil {
		return nil, err
	}
	sourceComponent := plan.Sessions.Components[tensor.FirstOffset]
	targetComponent := plan.Sessions.Components[tensor.SingletonExtent]
	sourceSession, err := resources.Source(ctx, sourceComponent)
	if err != nil {
		return nil, err
	}
	targetSession, err := resources.Target(ctx, targetComponent)
	if err != nil {
		return nil, err
	}
	weightAuthority, err := composition.LoadBridgeWeights(ctx, reader, bridge.ID)
	if err != nil {
		return nil, err
	}
	weights, err := resources.BridgeWeights(ctx, sourceComponent, bridge.ID, weightAuthority)
	if err != nil {
		return nil, err
	}
	if sourceSession == nil || targetSession == nil ||
		sourceSession.ModelID() != plan.SourceModel || targetSession.ModelID() != plan.TargetModel {
		return nil, errors.New("inference: composition session identity differs from execution plan")
	}
	return &ProductionComposition{
		plan: plan,
		session: representationBridgeSession{
			source: sourceSession, target: targetSession, program: program, weights: weights,
		},
	}, nil
}

// PlanIdentity returns the exact compiled authority used by this runtime.
func (runtime *ProductionComposition) PlanIdentity() artifact.ID {
	if runtime == nil {
		return artifact.ID{}
	}
	return runtime.plan.ID
}

// Forward captures the contracted source layer, evaluates the bridge, and
// replaces every target token embedding before target inference.
func (runtime *ProductionComposition) Forward(
	ctx context.Context,
	sourceTokens, targetTokens []tokenizer.TokenID,
) (reference.Value, error) {
	if runtime == nil {
		return reference.Value{}, errors.New("inference: production composition is absent")
	}
	if ctx == nil {
		return reference.Value{}, errors.New("inference: production composition context is absent")
	}
	if err := ctx.Err(); err != nil {
		return reference.Value{}, err
	}
	identity, err := runtime.transformedRepresentationCacheIdentity(sourceTokens)
	if err != nil {
		return reference.Value{}, err
	}
	runtime.cacheMu.Lock()
	bridged, found := runtime.cachedRepresentation, runtime.cachedSource == identity
	runtime.cacheMu.Unlock()
	if found {
		return runtime.session.inject(ctx, targetTokens, bridged)
	}
	bridged, err = runtime.session.captureAndTransform(ctx, sourceTokens)
	if err != nil {
		return reference.Value{}, err
	}
	runtime.cacheMu.Lock()
	runtime.cachedSource, runtime.cachedRepresentation = identity, bridged
	runtime.cacheMu.Unlock()
	return runtime.session.inject(ctx, targetTokens, bridged)
}

// ForwardArm executes one causal generation arm through this active-plan-owned
// runtime. Ablations preserve the same source and target sessions and differ
// only at the declared representation or bridge boundary.
func (runtime *ProductionComposition) ForwardArm(
	ctx context.Context,
	arm composition.CompositeGenerationArm,
	sourceTokens, targetTokens []tokenizer.TokenID,
) (reference.Value, error) {
	if runtime == nil {
		return reference.Value{}, errors.New("inference: production composition is absent")
	}
	if ctx == nil {
		return reference.Value{}, errors.New("inference: composite generation context is absent")
	}
	if err := ctx.Err(); err != nil {
		return reference.Value{}, err
	}
	switch arm {
	case composition.CompositeGenerationTargetBaseline:
		return runtime.session.target.ForwardWithEmbeddingOverrides(ctx, targetTokens, nil)
	case composition.CompositeGenerationComposed:
		return runtime.Forward(ctx, sourceTokens, targetTokens)
	case composition.CompositeGenerationSourceAblated:
		captured, err := runtime.session.capture(ctx, sourceTokens)
		if err != nil {
			return reference.Value{}, err
		}
		zero, err := materializedZero(captured.Shape)
		if err != nil {
			return reference.Value{}, err
		}
		bridged, err := runtime.session.transform(zero)
		if err != nil {
			return reference.Value{}, err
		}
		return runtime.session.inject(ctx, targetTokens, bridged)
	case composition.CompositeGenerationBridgeAblated:
		if _, err := runtime.session.capture(ctx, sourceTokens); err != nil {
			return reference.Value{}, err
		}
		_, targetBoundary := runtime.session.program.Boundaries()
		shape, err := compositionTensor(targetBoundary.Channels, uint64(len(targetTokens)))
		if err != nil {
			return reference.Value{}, err
		}
		zero, err := materializedZero(shape)
		if err != nil {
			return reference.Value{}, err
		}
		return runtime.session.inject(ctx, targetTokens, zero)
	default:
		return reference.Value{}, errors.New("inference: composite generation arm is invalid")
	}
}

func (runtime *ProductionComposition) transformedRepresentationCacheIdentity(sourceTokens []tokenizer.TokenID) (artifact.ID, error) {
	if runtime == nil || !runtime.plan.CacheIdentity.Valid() {
		return artifact.ID{}, errors.New("inference: transformed representation cache authority is absent")
	}
	return artifact.JSONID(artifact.KindProfile, struct {
		Authority artifact.ID         `json:"authority"`
		Source    []tokenizer.TokenID `json:"source"`
	}{Authority: runtime.plan.CacheIdentity, Source: sourceTokens})
}

func (session representationBridgeSession) captureAndTransform(
	ctx context.Context,
	sourceTokens []tokenizer.TokenID,
) (reference.Value, error) {
	captured, err := session.capture(ctx, sourceTokens)
	if err != nil {
		return reference.Value{}, err
	}
	return session.transform(captured)
}

func (session representationBridgeSession) capture(
	ctx context.Context,
	sourceTokens []tokenizer.TokenID,
) (reference.Value, error) {
	if ctx == nil || session.source == nil || session.target == nil {
		return reference.Value{}, errors.New("inference: representation bridge authority is absent")
	}
	if err := ctx.Err(); err != nil {
		return reference.Value{}, err
	}
	sourceBoundary, _ := session.program.Boundaries()
	if session.source.ModelID() != sourceBoundary.Model {
		return reference.Value{}, errors.New("inference: representation bridge model identity differs")
	}
	if sourceBoundary.Tap != representation.TapLayerInput || sourceBoundary.Layer == nil ||
		*sourceBoundary.Layer > math.MaxInt32 {
		return reference.Value{}, errors.New("inference: representation source tap is not an addressable layer input")
	}
	return session.source.ExtractLayerInputs(
		ctx, sourceTokens, []int32{int32(*sourceBoundary.Layer)},
	)
}

func materializedZero(shape tensor.Shape) (reference.Value, error) {
	elements, err := shape.Elements()
	if err != nil {
		return reference.Value{}, err
	}
	count, ok := checked.Int(elements)
	if !ok {
		return reference.Value{}, errors.New("inference: ablation representation exceeds host address space")
	}
	return reference.NewValue(shape, make([]float32, count))
}

func (session representationBridgeSession) inject(
	ctx context.Context,
	targetTokens []tokenizer.TokenID,
	bridged reference.Value,
) (reference.Value, error) {
	if ctx == nil || session.target == nil {
		return reference.Value{}, errors.New("inference: representation target authority is absent")
	}
	if err := ctx.Err(); err != nil {
		return reference.Value{}, err
	}
	_, targetBoundary := session.program.Boundaries()
	if session.target.ModelID() != targetBoundary.Model ||
		targetBoundary.Tap != representation.TapEmbeddingOutput || targetBoundary.Layer != nil {
		return reference.Value{}, errors.New("inference: representation target tap is not the embedding boundary")
	}
	rowCount := len(targetTokens)
	if uint64(rowCount) > math.MaxUint32 || !representationMatrix(bridged, targetBoundary.Channels) {
		return reference.Value{}, errors.New("inference: bridge output cannot replace the target token sequence")
	}
	complete, completeErr := representationExtent(bridged, tensor.FirstOffset, uint64(rowCount))
	if completeErr != nil || len(complete.Data) != len(bridged.Data) {
		return reference.Value{}, errors.New("inference: bridge output cannot replace the target token sequence")
	}
	overrides := make([]EmbeddingOverride, rowCount)
	for index := range overrides {
		row, rowErr := representationExtent(bridged, uint64(index), tensor.SingletonExtent)
		if rowErr != nil {
			return reference.Value{}, rowErr
		}
		overrides[index] = EmbeddingOverride{
			TokenIndex: uint32(index), Embedding: slices.Clone(row.Data),
		}
	}
	return session.target.ForwardWithEmbeddingOverrides(ctx, targetTokens, overrides)
}

func (session representationBridgeSession) transform(input reference.Value) (reference.Value, error) {
	builder := tensor.NewBuilder()
	feeds := make(map[*tensor.Tensor]reference.Value)
	bind := func(name string, value *reference.Value) *tensor.Tensor {
		if value == nil {
			return nil
		}
		node := builder.Input(name, dtype.F32, value.Shape)
		feeds[node] = *value
		return node
	}
	inputNode := builder.Input("representation-bridge.source", dtype.F32, input.Shape)
	feeds[inputNode] = input
	output, err := session.program.Build(builder, inputNode, bridgegraph.Weights{
		First:      bind("representation-bridge.first", session.weights.First),
		FirstBias:  bind("representation-bridge.first-bias", session.weights.FirstBias),
		Second:     bind("representation-bridge.second", session.weights.Second),
		SecondBias: bind("representation-bridge.second-bias", session.weights.SecondBias),
	})
	if err != nil {
		return reference.Value{}, err
	}
	values, err := reference.Execute([]*tensor.Tensor{output}, feeds)
	if err != nil {
		return reference.Value{}, err
	}
	return values[output], nil
}
