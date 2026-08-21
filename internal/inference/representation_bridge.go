package inference

import (
	"context"
	"errors"
	"math"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/bridgegraph"
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

// RepresentationBridgeSession owns one source-capture, bridge-transform, and
// target-injection operation over exact model identities.
type RepresentationBridgeSession struct {
	Source  RepresentationSource
	Target  RepresentationTarget
	Program bridgegraph.Program
	Weights RepresentationBridgeWeights
}

var (
	representationMatrix = reference.Value.IsMatrixWidth
	representationExtent = reference.Value.Rows
)

// Forward captures the contracted source layer, evaluates the bridge, and
// replaces every target token embedding with the validated bridge output.
func (session RepresentationBridgeSession) Forward(
	ctx context.Context,
	sourceTokens, targetTokens []tokenizer.TokenID,
) (reference.Value, error) {
	if ctx == nil || session.Source == nil || session.Target == nil {
		return reference.Value{}, errors.New("inference: representation bridge authority is absent")
	}
	if err := ctx.Err(); err != nil {
		return reference.Value{}, err
	}
	sourceBoundary, targetBoundary := session.Program.Boundaries()
	if session.Source.ModelID() != sourceBoundary.Model || session.Target.ModelID() != targetBoundary.Model {
		return reference.Value{}, errors.New("inference: representation bridge model identity differs")
	}
	if sourceBoundary.Tap != representation.TapLayerInput || sourceBoundary.Layer == nil ||
		*sourceBoundary.Layer > math.MaxInt32 {
		return reference.Value{}, errors.New("inference: representation source tap is not an addressable layer input")
	}
	if targetBoundary.Tap != representation.TapEmbeddingOutput || targetBoundary.Layer != nil {
		return reference.Value{}, errors.New("inference: representation target tap is not the embedding boundary")
	}
	captured, err := session.Source.ExtractLayerInputs(
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
	return session.Target.ForwardWithEmbeddingOverrides(ctx, targetTokens, overrides)
}

func (session RepresentationBridgeSession) transform(input reference.Value) (reference.Value, error) {
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
	output, err := session.Program.Build(builder, inputNode, bridgegraph.Weights{
		First:      bind("representation-bridge.first", session.Weights.First),
		FirstBias:  bind("representation-bridge.first-bias", session.Weights.FirstBias),
		Second:     bind("representation-bridge.second", session.Weights.Second),
		SecondBias: bind("representation-bridge.second-bias", session.Weights.SecondBias),
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
