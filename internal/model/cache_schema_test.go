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
	}
	schema.addState(CacheStatePositions, CacheState[CacheValueSchema]{
		Mode: CacheStateToken, Value: CacheValueSchema{Shape: tokenShape},
	})
	materialized := schema.WithTokenCount(fixtureTokens)
	positions, present := materialized.State(CacheStatePositions)
	if !present {
		t.Fatal("position state is missing")
	}
	if materialized.Primary.Key.Value.Shape.Dims[2] != fixtureTokens ||
		positions.Value.Shape.Dims[2] != fixtureTokens {
		t.Fatalf("materialized schema = %+v", materialized)
	}
	if !materialized.Primary.Value.Value.Shape.Equal(fixedShape) ||
		schema.Primary.Key.Value.Shape.Dims[2] != templateTokens {
		t.Fatal("materialization changed fixed or source schema")
	}
}

func TestCompileCacheSchemasRejectsLayerCountMismatch(t *testing.T) {
	const fixtureLayer = 0
	if _, err := compileCacheSchemas(
		Spec{}, []LayerPlan{{Layer: fixtureLayer}}, []LayerWeights{{}, {}},
	); err == nil {
		t.Fatal("cache catalog accepted mismatched layer counts")
	}
}

func TestCacheSchemaReportsInvalidRecurrentDimensions(t *testing.T) {
	const (
		fixtureLayerCount      = 1
		fixtureEmbeddingWidth  = 2
		fixtureNoHistoryKernel = 1
		fixtureInnerWidth      = 2
		fixtureStateWidth      = 2
		fixtureTokenCount      = 1
	)
	spec := Spec{
		CommonSpec: CommonSpec{
			Architecture: "mamba", BlockCount: fixtureLayerCount,
			EmbeddingLength: fixtureEmbeddingWidth,
		},
		RecurrentSpec: RecurrentSpec{
			SSMConvKernel: fixtureNoHistoryKernel,
			SSMInnerSize:  fixtureInnerWidth, SSMStateSize: fixtureStateWidth,
		},
	}
	plan := LayerPlan{
		Layer: 0, Block: BlockMamba, Cache: CacheMamba,
		CacheMode: CacheStateFixed,
	}
	if _, err := cacheSchemaForPlan(spec, plan, LayerWeights{}, fixtureTokenCount); err == nil {
		t.Fatal("zero-history recurrent cache shape was accepted")
	}
}
