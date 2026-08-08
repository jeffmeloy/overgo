package modeltest

import "overgo/internal/model"

const (
	dsaEmbeddingWidth   = 8
	dsaFeedForwardWidth = 12
	dsaAttentionHeads   = 2
	dsaKVHeads          = 1
	dsaKeyWidth         = 6
	dsaValueWidth       = 4
	dsaQueryRank        = 3
	dsaKVRank           = 3
	dsaRopeWidth        = 2
	dsaIndexerHeads     = 2
	dsaIndexerKeyWidth  = 8
	dsaIndexerTopK      = 2
	dsaRopeBase         = 10_000
	deepSeek32Blocks    = 62
)

// GLMDSA: compact full-indexer profile facts.
func GLMDSA() ModelFixture {
	return dsaFixture("glm-dsa", 1, 1e-6)
}

// DeepSeek32: compact full-indexer YaRN profile facts.
func DeepSeek32() ModelFixture {
	fixture := dsaFixture("deepseek32", deepSeek32Blocks, 1e-5)
	fixture.Spec.LayerNormEpsilon = 1e-6
	fixture.Spec.RopeScalingType = "yarn"
	fixture.Spec.RopeScalingFactor = 4
	fixture.Spec.OriginalContextLength = 16
	fixture.Spec.YaRNExtFactor = 1
	fixture.Spec.YaRNAttentionFactor = 1
	fixture.Spec.YaRNBetaFast = 32
	fixture.Spec.YaRNBetaSlow = 1
	return fixture
}

func dsaFixture(architecture string, blocks uint32, epsilon float32) ModelFixture {
	fullIndexer := make([]bool, blocks)
	for index := range fullIndexer {
		fullIndexer[index] = true
	}
	return ModelFixture{Spec: model.Spec{
		CommonSpec: model.CommonSpec{
			Architecture: architecture, BlockCount: blocks,
			EmbeddingLength: dsaEmbeddingWidth, FeedForwardLength: dsaFeedForwardWidth,
			RMSNormEpsilon: epsilon,
		},
		AttentionSpec: model.AttentionSpec{
			HeadCount: dsaAttentionHeads, HeadCountKV: dsaKVHeads,
			KeyLength: dsaKeyWidth, ValueLength: dsaValueWidth,
			QLoRARank: dsaQueryRank, KVLoRARank: dsaKVRank,
			RopeDimensionCount: dsaRopeWidth, RopeFrequencyBase: dsaRopeBase,
			IndexerHeadCount: dsaIndexerHeads, IndexerKeyLength: dsaIndexerKeyWidth,
			IndexerTopK: dsaIndexerTopK, IndexerFullLayers: fullIndexer,
		},
		MoESpec: model.MoESpec{LeadingDenseBlocks: blocks},
	}}
}
