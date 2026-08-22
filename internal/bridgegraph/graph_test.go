package bridgegraph

import (
	"encoding/json"
	"math"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/hostmath"
	"overgo/internal/representation"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func TestMLPBridgeForwardAndValidation(t *testing.T) {
	sourceContent, source := bridgeContract(t, "source", representation.ModalityAudio, 2, 1, 3,
		representation.NormalizationContract{Kind: representation.NormalizationNone, Magnitude: representation.MagnitudeNative})
	targetContent, target := bridgeContract(t, "target", representation.ModalityText, 3, 1, 4,
		representation.NormalizationContract{Kind: representation.NormalizationLayer, Epsilon: 1e-5, Magnitude: representation.MagnitudeUnit})
	definition := Definition{Source: source.ID, Target: target.ID, Operator: OperatorMLPGELU, Intermediate: 2, Bias: true}
	program, err := (Compiler{}).Compile(definition, sourceContent, targetContent)
	if err != nil {
		t.Fatal(err)
	}

	builder := tensor.NewBuilder()
	input := builder.Input("source", dtype.F32, tensor.MustShape(2, 2))
	weights := Weights{
		First:      builder.Input("first.weight", dtype.F32, tensor.MustShape(2, 2)),
		FirstBias:  builder.Input("first.bias", dtype.F32, tensor.MustShape(2)),
		Second:     builder.Input("second.weight", dtype.F32, tensor.MustShape(2, 3)),
		SecondBias: builder.Input("second.bias", dtype.F32, tensor.MustShape(3)),
	}
	output, err := program.Build(builder, input, weights)
	if err != nil {
		t.Fatal(err)
	}
	ops := make([]tensor.Op, 0, len(builder.Nodes()))
	for _, node := range builder.Nodes() {
		ops = append(ops, node.Op)
	}
	if !slices.Contains(ops, tensor.OpGELU) || !slices.Contains(ops, tensor.OpLayerNorm) {
		t.Fatalf("compiled operations = %v", ops)
	}
	inputValue := mustReferenceValue(t, input.Shape, []float32{1, -1, 2, -2})
	zeroFirst := reference.ZeroValue(weights.First.Shape)
	zeroFirstBias := reference.ZeroValue(weights.FirstBias.Shape)
	zeroSecond := reference.ZeroValue(weights.Second.Shape)
	secondBias := mustReferenceValue(t, weights.SecondBias.Shape, []float32{1, 2, 3})
	values, err := reference.Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]reference.Value{
		input: inputValue, weights.First: zeroFirst, weights.FirstBias: zeroFirstBias,
		weights.Second: zeroSecond, weights.SecondBias: secondBias,
	})
	if err != nil {
		t.Fatal(err)
	}
	wantInput := []float32{1, 2, 3, 1, 2, 3}
	want := make([]float32, len(wantInput))
	hostmath.LayerNormInto(want, wantInput, nil, nil, 2, 3, float64(target.Normalization.Epsilon))
	for index, value := range values[output].Data {
		if math.Abs(float64(value-want[index])) > 1e-6 {
			t.Fatalf("output[%d] = %g, want %g", index, value, want[index])
		}
	}

	t.Run("linear", func(t *testing.T) {
		linear, err := (Compiler{}).Compile(Definition{Source: source.ID, Target: target.ID, Operator: OperatorLinear}, sourceContent, targetContent)
		if err != nil {
			t.Fatal(err)
		}
		builder := tensor.NewBuilder()
		input := builder.Input("input", dtype.F32, tensor.MustShape(2, 1))
		weight := builder.Input("weight", dtype.F32, tensor.MustShape(2, 3))
		if _, err := linear.Build(builder, input, Weights{First: weight}); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("rejects contract and operator mismatch", func(t *testing.T) {
		wrong := definition
		wrong.Source = target.ID
		if _, err := (Compiler{}).Compile(wrong, sourceContent, targetContent); err == nil {
			t.Fatal("wrong source identity accepted")
		}
		wrong = definition
		wrong.Operator = "unknown"
		if _, err := (Compiler{}).Compile(wrong, sourceContent, targetContent); err == nil {
			t.Fatal("unknown operator accepted")
		}
		wrong = definition
		wrong.Intermediate = 0
		if _, err := (Compiler{}).Compile(wrong, sourceContent, targetContent); err == nil {
			t.Fatal("zero MLP width accepted")
		}
		wrong = definition
		wrong.Operator = OperatorLinear
		if _, err := (Compiler{}).Compile(wrong, sourceContent, targetContent); err == nil {
			t.Fatal("linear intermediate width accepted")
		}
	})

	t.Run("rejects sequence change", func(t *testing.T) {
		narrowContent, narrow := bridgeContract(t, "narrow", representation.ModalityText, 3, 1, 2,
			representation.NormalizationContract{Kind: representation.NormalizationNone, Magnitude: representation.MagnitudeNative})
		if _, err := (Compiler{}).Compile(Definition{
			Source: source.ID, Target: narrow.ID, Operator: OperatorLinear,
		}, sourceContent, narrowContent); err == nil {
			t.Fatal("narrow target sequence accepted")
		}
	})

	t.Run("rejects runtime geometry", func(t *testing.T) {
		builder := tensor.NewBuilder()
		short := builder.Input("short", dtype.F32, tensor.MustShape(2, 4))
		badWeight := builder.Input("bad", dtype.F32, tensor.MustShape(3, 2))
		if _, err := program.Build(builder, short, Weights{First: badWeight}); err == nil {
			t.Fatal("out-of-range input and invalid weights accepted")
		}
		builder = tensor.NewBuilder()
		validInput := builder.Input("input", dtype.F32, tensor.MustShape(2, 2))
		first := builder.Input("first", dtype.F32, tensor.MustShape(2, 2))
		second := builder.Input("second", dtype.F32, tensor.MustShape(2, 3))
		if _, err := program.Build(builder, validInput, Weights{First: first, Second: second}); err == nil {
			t.Fatal("missing configured biases accepted")
		}
	})
}

func TestExternalAttentionBridgeAdmission(t *testing.T) {
	sourceContent, source := bridgeContract(t, "external-source", representation.ModalityAudio, 3, 1, 4,
		representation.NormalizationContract{Kind: representation.NormalizationNone, Magnitude: representation.MagnitudeNative})
	_, target := bridgeContract(t, "external-target", representation.ModalityText, 6, 1, 8,
		representation.NormalizationContract{Kind: representation.NormalizationNone, Magnitude: representation.MagnitudeNative})
	layer := uint32(2)
	target.Producer.Tap, target.Producer.Layer, target.ID = representation.TapLayerOutput, &layer, artifact.ID{}
	targetContent, err := json.Marshal(target)
	if err != nil {
		t.Fatal(err)
	}
	target, err = representation.ParseContract(targetContent)
	if err != nil {
		t.Fatal(err)
	}
	targetContent, err = json.Marshal(target)
	if err != nil {
		t.Fatal(err)
	}
	program, err := (Compiler{}).Compile(Definition{
		Source: source.ID, Target: target.ID, Operator: OperatorExternalAttention,
		HeadCount: 2, SourceTokenLimit: 4,
	}, sourceContent, targetContent)
	if err != nil {
		t.Fatal(err)
	}
	builder := tensor.NewBuilder()
	input := builder.Input("source", dtype.F32, tensor.MustShape(3, 2))
	if _, err := program.Build(builder, input, Weights{}); err == nil {
		t.Fatal("external attention was admitted as a unary representation projection")
	}
}

func bridgeContract(
	t *testing.T,
	name string,
	modality representation.Modality,
	width, minimum, maximum uint64,
	normalization representation.NormalizationContract,
) ([]byte, representation.Contract) {
	t.Helper()
	value := representation.Contract{
		Version: representation.ContractVersion,
		Producer: representation.Producer{
			Model:      bridgeTestID(t, artifact.KindModel, name+"-model"),
			Definition: bridgeTestID(t, artifact.KindModelDefinition, name+"-definition"),
			Tap:        representation.TapEncoderOutput,
		},
		Modality: modality,
		Tensor: representation.TensorContract{DataType: dtype.F32, Axes: []representation.Axis{
			{Kind: representation.AxisChannel, Bounds: representation.AxisBounds{Extent: width}},
			{Kind: representation.AxisSequence, Bounds: representation.AxisBounds{Minimum: minimum, Maximum: maximum}},
		}},
		Sequence: representation.SequenceContract{
			Axis: representation.AxisSequence, Mask: representation.MaskPrefix,
			Padding: representation.PaddingSuffix, Position: representation.PositionSequential,
			PositionAxes: []representation.AxisKind{representation.AxisSequence},
		},
		Normalization: normalization,
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

func bridgeTestID(t *testing.T, kind artifact.Kind, value string) artifact.ID {
	t.Helper()
	id, err := artifact.IdentifyBytes(kind, []byte(value))
	if err != nil {
		t.Fatal(err)
	}
	return id
}

func mustReferenceValue(t *testing.T, shape tensor.Shape, data []float32) reference.Value {
	t.Helper()
	value, err := reference.NewValue(shape, data)
	if err != nil {
		t.Fatal(err)
	}
	return value
}
