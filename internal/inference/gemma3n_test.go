package inference

import (
	"math"
	"testing"

	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

func TestGemma3nPredictCoefficientLayout(t *testing.T) {
	value := func(shape tensor.Shape, data ...float32) *reference.Value {
		result := reference.Value{Shape: shape, Data: data}
		return &result
	}
	layer := model.HostLayer{
		AltUpRouterNorm: value(tensor.MustShape(2), 1, 1),
		AltUpRouter: value(
			tensor.MustShape(2, 2),
			1, 0,
			0, 0,
		),
		AltUpPredictCoefficient: value(
			tensor.MustShape(2, 4),
			1, 0,
			0, 0,
			0, 0,
			2, 0,
		),
	}
	states := []reference.Value{
		{Shape: tensor.MustShape(2, 1), Data: []float32{1, 0}},
		{Shape: tensor.MustShape(2, 1), Data: []float32{0, 1}},
	}
	spec := model.Spec{CommonSpec: model.CommonSpec{EmbeddingLength: 2, RMSNormEpsilon: 1e-6}, MultimodalSpec: model.MultimodalSpec{AltUpActive: 0}}
	got, err := gemma3nPredict(
		states, layer, spec.AltUpActive, spec.EmbeddingLength, spec.RMSNormEpsilon,
	)
	if err != nil {
		t.Fatal(err)
	}
	modality := float32(math.Tanh(float64((1 / float32(math.Sqrt(0.5+1e-6))) / 2)))
	if math.Abs(float64(got[0].Data[0]-(1+modality))) > 1e-6 || got[0].Data[1] != 0 ||
		got[1].Data[0] != 0 || math.Abs(float64(got[1].Data[1]-(1+2*modality))) > 1e-6 {
		t.Fatalf("predictions = %v / %v", got[0].Data, got[1].Data)
	}
}

func TestGemma3nMagnitudeMatchAndMerge(t *testing.T) {
	active := reference.Value{Shape: tensor.MustShape(2, 1), Data: []float32{3, 4}}
	projected := reference.Value{Shape: tensor.MustShape(2, 1), Data: []float32{0, 2}}
	matched := gemma3nMatchMagnitude(projected, active)
	if matched.Data[0] != 0 || math.Abs(float64(matched.Data[1]-5)) > 1e-6 {
		t.Fatalf("matched = %v", matched.Data)
	}
	unembedding := reference.Value{
		Shape: tensor.MustShape(2, 2, 1),
		Data:  []float32{1, 0, 0, 1},
	}
	merged, err := gemma3nMergeAltUp([]reference.Value{active, projected}, unembedding, 0)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(merged.Data[0]-1.5)) > 1e-6 || math.Abs(float64(merged.Data[1]-4.5)) > 1e-6 {
		t.Fatalf("merged = %v", merged.Data)
	}
}

func TestGemma3nGaussianSparsity(t *testing.T) {
	gate := reference.Value{Shape: tensor.MustShape(3, 1), Data: []float32{0, 1, 10}}
	up := reference.Value{Shape: tensor.MustShape(3, 1), Data: []float32{1, 1, 1}}
	got, err := gemma3nActivateFFN(gate, up, true, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got.Data[0] != 0 || got.Data[1] != 0 || got.Data[2] < 6.3 || got.Data[2] > 6.34 {
		t.Fatalf("sparse activation = %v", got.Data)
	}
}
