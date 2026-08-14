package model

import (
	"testing"

	"overgo/internal/testevidence"
)

func requireIntegration(t *testing.T) {
	t.Helper()
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip)
	}
}

const (
	fixtureEmbeddingWidth   = uint32(8)
	fixtureFeedForwardWidth = uint32(12)
	fixtureAttentionHeads   = uint32(2)
	fixtureKVHeads          = uint32(1)
	fixtureHeadWidth        = uint32(4)
	fixtureRMSNormEpsilon   = float32(1e-6)
	fixtureRoPEFrequency    = float32(10000)
)

type modelFixtureDimensions struct {
	embedding   uint32
	feedForward uint32
	heads       uint32
	kvHeads     uint32
	headWidth   uint32
}

var standardDecoderFixture = modelFixtureDimensions{
	embedding: fixtureEmbeddingWidth, feedForward: fixtureFeedForwardWidth,
	heads: fixtureAttentionHeads, kvHeads: fixtureKVHeads, headWidth: fixtureHeadWidth,
}

func (d modelFixtureDimensions) decoderSpec(architecture string) Spec {
	return Spec{
		CommonSpec: CommonSpec{
			Architecture: architecture, EmbeddingLength: d.embedding,
			FeedForwardLength: d.feedForward, RMSNormEpsilon: fixtureRMSNormEpsilon,
		},
		AttentionSpec: AttentionSpec{
			HeadCount: d.heads, HeadCountKV: d.kvHeads,
			KeyLength: d.headWidth, ValueLength: d.headWidth,
			RopeFrequencyBase: fixtureRoPEFrequency,
		},
	}
}
