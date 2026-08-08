package modeltest

import "overgo/internal/model"

const (
	nemotronEmbeddingWidth   = 8
	nemotronAttentionHeads   = 2
	nemotronKVHeads          = 1
	nemotronHeadWidth        = 4
	nemotronFeedForwardWidth = 6
)

// NemotronHAttention: compact attention-only layer facts.
func NemotronHAttention() ModelFixture {
	return ModelFixture{Spec: model.Spec{
		CommonSpec: model.CommonSpec{
			Architecture: "nemotron_h", BlockCount: 1,
			EmbeddingLength: nemotronEmbeddingWidth, FeedForwardLength: 1,
			RMSNormEpsilon: 1e-5,
		},
		AttentionSpec: model.AttentionSpec{
			HeadCount: nemotronAttentionHeads, HeadCountKV: nemotronKVHeads,
			LayerHeadCounts:   []uint32{nemotronAttentionHeads},
			LayerKVHeadCounts: []uint32{nemotronKVHeads},
			KeyLength:         nemotronHeadWidth, ValueLength: nemotronHeadWidth,
		},
		MoESpec:       model.MoESpec{LayerFeedForward: []uint32{0}},
		RecurrentSpec: model.RecurrentSpec{RecurrentLayers: []bool{false}},
	}}
}

// NemotronHRecurrent: compact recurrent-only layer facts.
func NemotronHRecurrent() ModelFixture {
	fixture := NemotronHAttention()
	fixture.Spec.EmbeddingLength = nemotronHeadWidth
	fixture.Spec.HeadCount = 1
	fixture.Spec.HeadCountKV = 1
	fixture.Spec.LayerHeadCounts = []uint32{0}
	fixture.Spec.LayerKVHeadCounts = []uint32{0}
	fixture.Spec.RecurrentSpec = model.RecurrentSpec{
		RecurrentLayers: []bool{true}, SSMConvKernel: 3,
		SSMInnerSize: 8, SSMStateSize: 2, SSMTimeStepRank: 4, SSMGroupCount: 2,
	}
	return fixture
}

// NemotronHMoE: compact routed feed-forward layer facts.
func NemotronHMoE() ModelFixture {
	fixture := NemotronHAttention()
	fixture.Spec.Architecture = "nemotron_h_moe"
	fixture.Spec.FeedForwardLength = nemotronFeedForwardWidth
	fixture.Spec.MoESpec = model.MoESpec{
		LayerFeedForward: []uint32{nemotronFeedForwardWidth},
		ExpertCount:      4, ExpertUsedCount: 2,
		ExpertFeedForward: nemotronFeedForwardWidth, SharedExpertFF: 5,
		ExpertWeightsNorm: true, ExpertWeightsScale: 1.25, MoELatentSize: 4,
	}
	return fixture
}
