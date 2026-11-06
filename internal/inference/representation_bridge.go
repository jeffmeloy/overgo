package inference

import (
	"context"
	"errors"
	"image"

	"overgo/internal/artifact"
	"overgo/internal/bridgegraph"
	"overgo/internal/composition"
	"overgo/internal/modelrecipe"
	"overgo/internal/projector"
	"overgo/internal/recipe"
	"overgo/internal/representation"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// RepresentationSource encodes an image into the representation a composition's source contract names: an image
// encoder's output, its channels by its tokens.
type RepresentationSource interface {
	ModelID() artifact.ID
	EncodeImage(context.Context, image.Image) (reference.Value, error)
}

// RepresentationTarget is the text model a composition's bridged rows enter: its identity, its embedding width and the
// placeholder token a row takes.
type RepresentationTarget interface {
	ModelID() artifact.ID
	EmbeddingWidth() int
	ImagePlaceholder() string
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

// compositionFamily names a composition's prompt plan beside the projectors' families.
const compositionFamily = "composition"

// ProductionComposition is an active composition's image prompt session: its source encodes each image, its bridge
// carries the encoding to the text model's width, and the rows take the text model's placeholder tokens, the path a
// native projector's rows take to the runner. Both models stay as their authorities name them.
type ProductionComposition struct {
	projector.Session
	plan composition.CompositionExecutionPlan
}

type representationBridgeSession struct {
	source  RepresentationSource
	program bridgegraph.Program
	weights RepresentationBridgeWeights
}

// OpenProductionComposition is the only composition construction path. It resolves the active recipe, recompiles
// exact contract seams, and admits resources solely against the resulting plan.
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
		return nil, errors.New("inference: a composition requires one sealed bridge")
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
	sourceBoundary, targetBoundary := program.Boundaries()
	if sourceBoundary.Tap != representation.TapEncoderOutput || sourceBoundary.Layer != nil ||
		targetBoundary.Tap != representation.TapEmbeddingOutput || targetBoundary.Layer != nil {
		return nil, errors.New("inference: a composition bridges an encoder's output to a text model's embeddings")
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
	if uint64(targetSession.EmbeddingWidth()) != targetBoundary.Channels {
		return nil, errors.New("inference: the bridge's output width differs from the text model's embeddings")
	}
	session := representationBridgeSession{source: sourceSession, program: program, weights: weights}
	vision, err := projector.NewEncodedImageSession(
		compositionFamily, targetSession.ImagePlaceholder(), targetSession.EmbeddingWidth(), session.captureAndTransform,
	)
	if err != nil {
		return nil, err
	}
	return &ProductionComposition{Session: vision, plan: plan}, nil
}

// Recipe is the composition recipe the runtime was compiled from.
func (runtime *ProductionComposition) Recipe() artifact.ID {
	if runtime == nil {
		return artifact.ID{}
	}
	return runtime.plan.CompositionRecipe
}

// captureAndTransform encodes one image through the source and carries it across the bridge: the rows the text model's
// placeholders take, its embedding width by the encoder's tokens.
func (session representationBridgeSession) captureAndTransform(ctx context.Context, picture image.Image) (reference.Value, error) {
	if ctx == nil || session.source == nil {
		return reference.Value{}, errors.New("inference: representation bridge authority is absent")
	}
	if err := ctx.Err(); err != nil {
		return reference.Value{}, err
	}
	sourceBoundary, _ := session.program.Boundaries()
	if session.source.ModelID() != sourceBoundary.Model {
		return reference.Value{}, errors.New("inference: representation bridge model identity differs")
	}
	captured, err := session.source.EncodeImage(ctx, picture)
	if err != nil {
		return reference.Value{}, err
	}
	return session.transform(captured)
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
