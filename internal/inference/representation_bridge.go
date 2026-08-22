package inference

import (
	"context"
	"errors"
	"math"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/bridgegraph"
	"overgo/internal/composition"
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
	Source(context.Context, composition.CompositionComponentPlan) (RepresentationSource, error)
	Target(context.Context, composition.CompositionComponentPlan) (RepresentationTarget, error)
	BridgeWeights(context.Context, artifact.ID, composition.BridgeWeightAuthority) (RepresentationBridgeWeights, error)
}

// ProductionComposition is an active-recipe-owned embedding-injection
// runtime. Its bridge session cannot be constructed directly.
type ProductionComposition struct {
	plan    composition.CompositionExecutionPlan
	session representationBridgeSession
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
		len(plan.BridgeWeights) != tensor.SingletonExtent || len(plan.Components) != tensor.PairedExtent {
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
	sourceSession, err := resources.Source(ctx, plan.Components[tensor.FirstOffset])
	if err != nil {
		return nil, err
	}
	targetSession, err := resources.Target(ctx, plan.Components[tensor.SingletonExtent])
	if err != nil {
		return nil, err
	}
	weightAuthority, err := composition.LoadBridgeWeights(ctx, reader, bridge.ID)
	if err != nil {
		return nil, err
	}
	weights, err := resources.BridgeWeights(ctx, bridge.ID, weightAuthority)
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
	return runtime.session.forward(ctx, sourceTokens, targetTokens)
}

func (session representationBridgeSession) forward(
	ctx context.Context,
	sourceTokens, targetTokens []tokenizer.TokenID,
) (reference.Value, error) {
	if ctx == nil || session.source == nil || session.target == nil {
		return reference.Value{}, errors.New("inference: representation bridge authority is absent")
	}
	if err := ctx.Err(); err != nil {
		return reference.Value{}, err
	}
	sourceBoundary, targetBoundary := session.program.Boundaries()
	if session.source.ModelID() != sourceBoundary.Model || session.target.ModelID() != targetBoundary.Model {
		return reference.Value{}, errors.New("inference: representation bridge model identity differs")
	}
	if sourceBoundary.Tap != representation.TapLayerInput || sourceBoundary.Layer == nil ||
		*sourceBoundary.Layer > math.MaxInt32 {
		return reference.Value{}, errors.New("inference: representation source tap is not an addressable layer input")
	}
	if targetBoundary.Tap != representation.TapEmbeddingOutput || targetBoundary.Layer != nil {
		return reference.Value{}, errors.New("inference: representation target tap is not the embedding boundary")
	}
	captured, err := session.source.ExtractLayerInputs(
		ctx, sourceTokens, []int32{int32(*sourceBoundary.Layer)},
	)
	if err != nil {
		return reference.Value{}, err
	}
	bridged, err := session.transform(captured)
	if err != nil {
		return reference.Value{}, err
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
