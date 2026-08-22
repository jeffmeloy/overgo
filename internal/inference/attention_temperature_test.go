package inference

import (
	"math"
	"strings"
	"testing"

	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func TestMistral3AttentionTemperatureInput(t *testing.T) {
	const fixtureAttentionLayerCount = 1
	builder := tensor.NewBuilder()
	feeds := make(map[*tensor.Tensor]reference.Value)
	weights := model.LayerGraphWeights{}
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "mistral3", BlockCount: fixtureAttentionLayerCount}, AttentionSpec: model.AttentionSpec{AttentionTempScale: 0.1, AttentionTempFloor: 8}}
	spec = bindFixtureSpec(spec)
	plan := fixtureLayerPlan(spec, 0)
	if _, err := bindLayerSideInputs(
		builder, spec, []uint32{0, 7, 8, 16}, plan, feeds, &weights, layerSideInputs{},
	); err != nil {
		t.Fatal(err)
	}
	value, ok := feeds[weights.AttentionTemperatureScale]
	if !ok || value.Shape != tensor.MustShape(1, 1, 4) {
		t.Fatalf("temperature input = %+v", value)
	}
	want := []float32{
		1,
		1,
		1 + 0.1*float32(math.Log(2)),
		1 + 0.1*float32(math.Log(3)),
	}
	for index := range want {
		if math.Abs(float64(value.Data[index]-want[index])) > 1e-7 {
			t.Fatalf("temperature[%d] = %g, want %g", index, value.Data[index], want[index])
		}
	}
}

func TestDeepSeek4PositionRequiresExactF32Representation(t *testing.T) {
	maximumExactPosition := uint32(1 << 24)
	builder := tensor.NewBuilder()
	feeds := make(map[*tensor.Tensor]reference.Value)
	weights := model.LayerGraphWeights{}
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "deepseek4"}}
	plan := model.LayerPlan{Cache: model.CacheCompressedAttention}
	if _, err := bindLayerSideInputs(
		builder,
		spec,
		[]uint32{maximumExactPosition},
		plan,
		feeds,
		&weights,
		layerSideInputs{},
	); err != nil {
		t.Fatalf("maximum exact position rejected: %v", err)
	}
	if _, err := bindLayerSideInputs(
		tensor.NewBuilder(),
		spec,
		[]uint32{maximumExactPosition + 1},
		plan,
		make(map[*tensor.Tensor]reference.Value),
		&model.LayerGraphWeights{},
		layerSideInputs{},
	); err == nil || !strings.Contains(err.Error(), "exact F32") {
		t.Fatalf("inexact position error = %v", err)
	}
}

func TestBindLayerSideInputs(t *testing.T) {
	builder := tensor.NewBuilder()
	feeds := make(map[*tensor.Tensor]reference.Value)
	weights := model.LayerGraphWeights{}
	skip := builder.Input("skip", dtype.F32, tensor.MustShape(2, 3))
	perLayer := builder.Input("per_layer", dtype.F32, tensor.MustShape(2, 3))
	blocks := builder.Input("blocks", dtype.F32, tensor.MustShape(3))
	bound, err := bindLayerSideInputs(
		builder,
		model.Spec{CommonSpec: model.CommonSpec{Architecture: "deepseek4"}},
		[]uint32{4, 9},
		model.LayerPlan{
			Layer: 3, Cache: model.CacheCompressedAttention, EmbeddingSkip: true,
			PerLayerInput: true, AttentionBlocks: model.AttentionBlocksUncached,
		},
		feeds,
		&weights,
		layerSideInputs{embeddingSkip: skip, perLayerInput: perLayer, attentionBlock: blocks},
	)
	if err != nil {
		t.Fatal(err)
	}
	if weights.EmbeddingSkip != skip || weights.PerLayerInput != perLayer ||
		weights.AttentionBlockIDs != blocks || bound.currentPositions == nil {
		t.Fatalf("bound side inputs = %+v, weights = %+v", bound, weights)
	}
	positionValue := feeds[bound.currentPositions]
	if !positionValue.Shape.Equal(tensor.MustShape(1, 1, 2)) ||
		len(positionValue.Data) != 2 || positionValue.Data[0] != 4 || positionValue.Data[1] != 9 {
		t.Fatalf("current positions = %+v", positionValue)
	}
}
