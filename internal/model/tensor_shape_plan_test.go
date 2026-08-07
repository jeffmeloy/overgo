package model

import "testing"

const (
	shapeFixtureEmbedding   = 24
	shapeFixtureFeedForward = 48
	shapeFixtureQueryHeads  = 6
	shapeFixtureKVHeads     = 2
	shapeFixtureKey         = 4
	shapeFixtureValue       = 3
)

func TestTensorShapePlanDerivesAttentionWidths(t *testing.T) {
	plan := TensorShapePlan{
		Embedding: shapeFixtureEmbedding, FeedForward: shapeFixtureFeedForward,
		QueryHeads: shapeFixtureQueryHeads, KVHeads: shapeFixtureKVHeads,
		Key: shapeFixtureKey, Value: shapeFixtureValue,
	}
	if got, want := plan.QueryProjectionWidth(), uint64(shapeFixtureQueryHeads*shapeFixtureKey); got != want {
		t.Fatalf("query projection width = %d, want %d", got, want)
	}
	if got, want := plan.KeyProjectionWidth(), uint64(shapeFixtureKVHeads*shapeFixtureKey); got != want {
		t.Fatalf("key projection width = %d, want %d", got, want)
	}
	if got, want := plan.ValueProjectionWidth(), uint64(shapeFixtureKVHeads*shapeFixtureValue); got != want {
		t.Fatalf("value projection width = %d, want %d", got, want)
	}
	if got, want := plan.AttentionOutputWidth(), uint64(shapeFixtureQueryHeads*shapeFixtureValue); got != want {
		t.Fatalf("attention output width = %d, want %d", got, want)
	}
}
