package model

import (
	"testing"

	"overgo/internal/tensor"
)

func TestLayerCacheSchemaMaterializesTokenDimensions(t *testing.T) {
	const (
		fixtureWidth   = 3
		fixtureHeads   = 1
		templateTokens = 1
		fixtureTokens  = 7
	)
	tokenShape := tensor.MustShape(fixtureWidth, fixtureHeads, templateTokens)
	fixedShape := tensor.MustShape(fixtureWidth)
	schema := LayerCacheSchema{
		Primary: NewCachePair(
			CacheState[CacheValueSchema]{
				Mode: CacheStateToken, Value: CacheValueSchema{Shape: tokenShape},
			},
			CacheState[CacheValueSchema]{
				Mode: CacheStateFixed, Value: CacheValueSchema{Shape: fixedShape},
			},
		),
		States: CacheStates[CacheValueSchema]{
			CacheStatePositions: {
				Mode: CacheStateToken, Value: CacheValueSchema{Shape: tokenShape},
			},
		},
	}
	materialized := schema.WithTokenCount(fixtureTokens)
	if materialized.Primary.Key.Value.Shape.Dims[2] != fixtureTokens ||
		materialized.States[CacheStatePositions].Value.Shape.Dims[2] != fixtureTokens {
		t.Fatalf("materialized schema = %+v", materialized)
	}
	if !materialized.Primary.Value.Value.Shape.Equal(fixedShape) ||
		schema.Primary.Key.Value.Shape.Dims[2] != templateTokens {
		t.Fatal("materialization changed fixed or source schema")
	}
}

func TestCompileCacheSchemasRejectsLayerCountMismatch(t *testing.T) {
	const fixtureLayer = 0
	plan := ModelPlan{Layers: []LayerPlan{{Layer: fixtureLayer}}}
	if _, err := CompileCacheSchemas(Spec{}, plan, nil); err == nil {
		t.Fatal("cache catalog accepted mismatched layer counts")
	}
}
