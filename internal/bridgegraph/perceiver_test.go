package bridgegraph

import (
	"encoding/json"
	"math"
	"slices"
	"testing"

	"overgo/internal/representation"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func TestPerceiverResampler(t *testing.T) {
	sourceContent, source := bridgeContract(
		t, "perceiver-source", representation.ModalityVideo, 2, 1, 4,
		representation.NormalizationContract{Kind: representation.NormalizationNone, Magnitude: representation.MagnitudeNative},
	)
	targetContent, target := bridgeContract(
		t, "perceiver-target", representation.ModalityText, 2, 2, 2,
		representation.NormalizationContract{Kind: representation.NormalizationNone, Magnitude: representation.MagnitudeNative},
	)
	target.Tensor.Axes[tensor.SingletonExtent].Bounds = representation.AxisBounds{Extent: 2}
	target.Sequence.Mask = representation.MaskNone
	target.Sequence.Padding = representation.PaddingNone
	encodedTarget, err := json.Marshal(target)
	if err != nil {
		t.Fatal(err)
	}
	target, err = representation.ParseContract(encodedTarget)
	if err != nil {
		t.Fatal(err)
	}
	targetContent = encodedTarget
	definition := Definition{
		Source: source.ID, Target: target.ID, Operator: OperatorPerceiver,
		LatentCount: 2, HeadCount: 1, SourceTokenLimit: 3,
	}
	program, err := (Compiler{}).Compile(definition, sourceContent, targetContent)
	if err != nil {
		t.Fatal(err)
	}
	builder := tensor.NewBuilder()
	input := builder.Input("source", dtype.F32, tensor.MustShape(2, 3))
	mask := builder.Input("mask", dtype.F32, tensor.MustShape(3))
	latents := builder.Input("latents", dtype.F32, tensor.MustShape(2, 2))
	query := builder.Input("query", dtype.F32, tensor.MustShape(2, 2))
	key := builder.Input("key", dtype.F32, tensor.MustShape(2, 2))
	value := builder.Input("value", dtype.F32, tensor.MustShape(2, 2))
	projection := builder.Input("output", dtype.F32, tensor.MustShape(2, 2))
	output, err := program.Build(builder, input, Weights{
		Mask: mask, Latents: latents, Query: query, Key: key, Value: value, Output: projection,
	})
	if err != nil {
		t.Fatal(err)
	}
	attentionFound := false
	for _, node := range builder.Nodes() {
		if node.Op == tensor.OpAttention && node.Attrs.(tensor.AttentionAttributes).HasKeyBias {
			attentionFound = true
		}
	}
	if !attentionFound {
		t.Fatal("masked Perceiver attention is absent")
	}
	identity := inferenceValue(t, tensor.MustShape(2, 2), []float32{1, 0, 0, 1})
	feeds := map[*tensor.Tensor]reference.Value{
		mask:    inferenceValue(t, mask.Shape, []float32{0, 0, -10000}),
		latents: identity, query: identity, key: identity, value: identity, projection: identity,
	}
	firstFeeds := make(map[*tensor.Tensor]reference.Value, len(feeds)+1)
	secondFeeds := make(map[*tensor.Tensor]reference.Value, len(feeds)+1)
	for node, value := range feeds {
		firstFeeds[node], secondFeeds[node] = value, value
	}
	firstFeeds[input] = inferenceValue(t, input.Shape, []float32{1, 0, 0, 1, 10, 10})
	secondFeeds[input] = inferenceValue(t, input.Shape, []float32{1, 0, 0, 1, 1000, -1000})
	first, err := reference.Execute([]*tensor.Tensor{output}, firstFeeds)
	if err != nil {
		t.Fatal(err)
	}
	second, err := reference.Execute([]*tensor.Tensor{output}, secondFeeds)
	if err != nil {
		t.Fatal(err)
	}
	if len(first[output].Data) != len(second[output].Data) {
		t.Fatalf("Perceiver output extents differ: %v %v", first[output], second[output])
	}
	for index, value := range first[output].Data {
		if math.Abs(float64(value-second[output].Data[index])) > 1e-6 {
			t.Fatalf("masked source changed output: first=%v second=%v", first[output].Data, second[output].Data)
		}
	}
	if slices.Equal(first[output].Data, latentsFixture(identity)) {
		t.Fatal("Perceiver attention did not update learned latents")
	}

	overflowBuilder := tensor.NewBuilder()
	overflow := overflowBuilder.Input("source", dtype.F32, tensor.MustShape(2, 4))
	if _, err := program.Build(overflowBuilder, overflow, Weights{
		Mask: mask, Latents: latents, Query: query, Key: key, Value: value, Output: projection,
	}); err == nil {
		t.Fatal("source above declared token limit accepted")
	}
}

func latentsFixture(value reference.Value) []float32 { return slices.Clone(value.Data) }
