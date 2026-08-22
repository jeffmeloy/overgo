package inference

import (
	"context"
	"encoding/json"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/bridgegraph"
	"overgo/internal/representation"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

type bridgeSourceFixture struct {
	model  artifact.ID
	value  reference.Value
	layers []int32
}

func (source *bridgeSourceFixture) ModelID() artifact.ID { return source.model }

func (source *bridgeSourceFixture) ExtractLayerInputs(
	_ context.Context,
	_ []tokenizer.TokenID,
	layers []int32,
) (reference.Value, error) {
	source.layers = slices.Clone(layers)
	return source.value.Clone(), nil
}

type bridgeTargetFixture struct {
	model     artifact.ID
	embedding reference.Value
	overrides []EmbeddingOverride
}

func (target *bridgeTargetFixture) ModelID() artifact.ID { return target.model }

func (target *bridgeTargetFixture) ForwardWithEmbeddingOverrides(
	_ context.Context,
	_ []tokenizer.TokenID,
	overrides []EmbeddingOverride,
) (reference.Value, error) {
	target.overrides = slices.Clone(overrides)
	result := target.embedding.Clone()
	return result, applyEmbeddingOverrides(&result, overrides)
}

func TestRepresentationBridgeEmbeddingInjection(t *testing.T) {
	sourceModel := testutil.ArtifactID(t, artifact.KindModel, "bridge-source-model")
	targetModel := testutil.ArtifactID(t, artifact.KindModel, "bridge-target-model")
	layer := uint32(0)
	sourceContent, sourceContract := inferenceBridgeContract(
		t, sourceModel, representation.TapLayerInput, &layer, representation.ModalityAudio, 2,
	)
	targetContent, targetContract := inferenceBridgeContract(
		t, targetModel, representation.TapEmbeddingOutput, nil, representation.ModalityText, 3,
	)
	program, err := (bridgegraph.Compiler{}).Compile(bridgegraph.Definition{
		Source: sourceContract.ID, Target: targetContract.ID,
		Operator: bridgegraph.OperatorLinear, Bias: true,
	}, sourceContent, targetContent)
	if err != nil {
		t.Fatal(err)
	}
	source := &bridgeSourceFixture{
		model: sourceModel,
		value: inferenceBridgeValue(t, tensor.MustShape(2, 2), []float32{1, 2, 3, 4}),
	}
	target := &bridgeTargetFixture{
		model:     targetModel,
		embedding: inferenceBridgeValue(t, tensor.MustShape(3, 2), []float32{0, 0, 0, 0, 0, 0}),
	}
	first := reference.ZeroValue(tensor.MustShape(2, 3))
	bias := inferenceBridgeValue(t, tensor.MustShape(3), []float32{10, 20, 30})
	session := RepresentationBridgeSession{
		Source: source, Target: target, Program: program,
		Weights: RepresentationBridgeWeights{First: &first, FirstBias: &bias},
	}
	result, err := session.Forward(
		context.Background(),
		[]tokenizer.TokenID{1, 2}, []tokenizer.TokenID{3, 4},
	)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{10, 20, 30, 10, 20, 30}
	if !slices.Equal(result.Data, want) || len(target.overrides) != len(want)/len(bias.Data) ||
		!slices.Equal(source.layers, []int32{int32(layer)}) {
		t.Fatalf("bridge result=%v overrides=%v layers=%v", result.Data, target.overrides, source.layers)
	}
	wrongTarget := *target
	wrongTarget.model = sourceModel
	session.Target = &wrongTarget
	if _, err := session.Forward(
		context.Background(), []tokenizer.TokenID{1, 2}, []tokenizer.TokenID{3, 4},
	); err == nil {
		t.Fatal("mismatched target model accepted")
	}
}

func inferenceBridgeContract(
	t *testing.T,
	modelID artifact.ID,
	tap representation.TapPoint,
	layer *uint32,
	modality representation.Modality,
	width uint64,
) ([]byte, representation.Contract) {
	t.Helper()
	value := representation.Contract{
		Version: representation.ContractVersion,
		Producer: representation.Producer{
			Model:      modelID,
			Definition: testutil.ArtifactID(t, artifact.KindModelDefinition, string(modality)+"-definition"),
			Tap:        tap, Layer: layer,
		},
		Modality: modality,
		Tensor: representation.TensorContract{DataType: dtype.F32, Axes: []representation.Axis{
			{Kind: representation.AxisChannel, Bounds: representation.AxisBounds{Extent: width}},
			{Kind: representation.AxisSequence, Bounds: representation.AxisBounds{Minimum: 1, Maximum: 4}},
		}},
		Sequence: representation.SequenceContract{
			Axis: representation.AxisSequence, Mask: representation.MaskPrefix,
			Padding: representation.PaddingSuffix, Position: representation.PositionSequential,
			PositionAxes: []representation.AxisKind{representation.AxisSequence},
		},
		Normalization: representation.NormalizationContract{
			Kind: representation.NormalizationNone, Magnitude: representation.MagnitudeNative,
		},
	}
	content, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := representation.ParseContract(content)
	if err != nil {
		t.Fatal(err)
	}
	return content, parsed
}

func inferenceBridgeValue(t *testing.T, shape tensor.Shape, data []float32) reference.Value {
	t.Helper()
	value, err := reference.NewValue(shape, data)
	if err != nil {
		t.Fatal(err)
	}
	return value
}
