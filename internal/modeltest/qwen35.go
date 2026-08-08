package modeltest

import (
	"overgo/internal/gguf"
	"overgo/internal/model"
)

const fixtureTokenEmbedding = "token_embd.weight"

// ModelFixture: coherent specification and layer inventory.
type ModelFixture struct {
	Spec    model.Spec
	Weights model.Weights
}

// Qwen35: compact valid hybrid model facts.
func Qwen35() ModelFixture {
	return ModelFixture{Spec: model.Spec{
		CommonSpec: model.CommonSpec{
			Architecture: "qwen35", EmbeddingLength: 8, FeedForwardLength: 12,
			RMSNormEpsilon: 1e-6,
		},
		AttentionSpec: model.AttentionSpec{
			HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
			RopeFrequencyBase: 10_000, RopeDimensionCount: 4,
			RopeSections: [4]int32{1, 1, 0, 0},
		},
		RecurrentSpec: model.RecurrentSpec{
			SSMConvKernel: 3, SSMInnerSize: 4, SSMStateSize: 2,
			SSMTimeStepRank: 2, SSMGroupCount: 1, FullAttentionInterval: 4,
		},
	}}
}

// Qwen35DenseRecurrentPair: one recurrent layer followed by one attention layer.
func Qwen35DenseRecurrentPair() ModelFixture {
	fixture := Qwen35()
	fixture.Spec.BlockCount = 2
	fixture.Spec.FullAttentionInterval = fixture.Spec.BlockCount
	fixture.Weights.Layers = []model.LayerWeights{{Recurrent: true}, {}}
	return fixture
}

// ServingWeights: physical marker required by serving-plan validation.
func (f ModelFixture) ServingWeights() model.Weights {
	weights := f.Weights
	weights.Layers = append([]model.LayerWeights(nil), f.Weights.Layers...)
	weights.TokenEmbedding = gguf.TensorInfo{Name: fixtureTokenEmbedding}
	return weights
}
