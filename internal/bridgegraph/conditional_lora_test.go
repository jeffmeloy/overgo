package bridgegraph

import (
	"math"
	"slices"
	"testing"

	"overgo/internal/representation"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func TestConditionalLoRAScale(t *testing.T) {
	sourceContent, source := bridgeContract(
		t, "conditional-source", representation.ModalityAudio, 2, 1, 2,
		representation.NormalizationContract{Kind: representation.NormalizationNone, Magnitude: representation.MagnitudeNative},
	)
	targetContent, target := bridgeContract(
		t, "conditional-target", representation.ModalityText, 3, 1, 2,
		representation.NormalizationContract{Kind: representation.NormalizationNone, Magnitude: representation.MagnitudeNative},
	)
	definition := Definition{
		Source: source.ID, Target: target.ID, Operator: OperatorConditionalLoRA,
		LoRARank: 1, ScaleLimit: 0.5,
	}
	program, err := (Compiler{}).Compile(definition, sourceContent, targetContent)
	if err != nil {
		t.Fatal(err)
	}
	builder := tensor.NewBuilder()
	condition := builder.Input("condition", dtype.F32, tensor.MustShape(2, 2))
	targetInput := builder.Input("target", dtype.F32, tensor.MustShape(3, 2))
	conditioner := builder.Input("conditioner", dtype.F32, tensor.MustShape(2, 1))
	a := builder.Input("a", dtype.F32, tensor.MustShape(3, 1))
	b := builder.Input("b", dtype.F32, tensor.MustShape(1, 3))
	output, err := program.BuildConditionalLoRA(builder, condition, targetInput, ConditionalLoRAWeights{
		Conditioner: conditioner, A: a, B: b,
	})
	if err != nil {
		t.Fatal(err)
	}
	ops := make([]tensor.Op, 0, len(builder.Nodes()))
	for _, node := range builder.Nodes() {
		ops = append(ops, node.Op)
	}
	if !slices.Contains(ops, tensor.OpTanh) || !slices.Contains(ops, tensor.OpScale) {
		t.Fatalf("conditional LoRA operations=%v", ops)
	}
	baseFeeds := map[*tensor.Tensor]reference.Value{
		targetInput: inferenceValue(t, targetInput.Shape, []float32{1, 1, 1, 1, 1, 1}),
		conditioner: inferenceValue(t, conditioner.Shape, []float32{1, 1}),
		a:           inferenceValue(t, a.Shape, []float32{1, 1, 1}),
		b:           inferenceValue(t, b.Shape, []float32{1, 2, 3}),
	}
	baseFeeds[condition] = inferenceValue(t, condition.Shape, []float32{100, 100, -100, -100})
	conditioned, err := reference.Execute([]*tensor.Tensor{output}, baseFeeds)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{2.5, 4, 5.5, -0.5, -2, -3.5}
	for index, value := range conditioned[output].Data {
		if math.Abs(float64(value-want[index])) > 1e-6 {
			t.Fatalf("conditional output=%v want=%v", conditioned[output].Data, want)
		}
	}
	baseFeeds[condition] = reference.ZeroValue(condition.Shape)
	identity, err := reference.Execute([]*tensor.Tensor{output}, baseFeeds)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(identity[output].Data, baseFeeds[targetInput].Data) {
		t.Fatalf("zero-conditioned LoRA changed target: %v", identity[output].Data)
	}

	unbounded := definition
	unbounded.ScaleLimit = float32(math.Inf(1))
	if _, err := (Compiler{}).Compile(unbounded, sourceContent, targetContent); err == nil {
		t.Fatal("non-finite conditional scale bound accepted")
	}
}
