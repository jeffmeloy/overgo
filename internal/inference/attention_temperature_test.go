package inference

import (
	"math"
	"testing"

	"llamacpp2go/internal/model"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/reference"
)

func TestMistral3AttentionTemperatureInput(t *testing.T) {
	builder := tensor.NewBuilder()
	feeds := make(map[*tensor.Tensor]reference.Value)
	weights := model.LayerGraphWeights{}
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "mistral3"}, AttentionSpec: model.AttentionSpec{AttentionTempScale: 0.1, AttentionTempFloor: 8}}
	if err := addAttentionTemperatureInput(
		builder, spec, []uint32{0, 7, 8, 16}, 0, feeds, &weights,
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
