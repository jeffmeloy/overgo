package model

import (
	"fmt"
	"hash/fnv"
	"math"
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// sharedKVCapacitySpec is the Gemma 4 E4B layout at fixture size: layers 2 and
// 3 own no KV and read the last owning layer of their kind, the sliding layer 0
// and the full layer 1.
func sharedKVCapacitySpec() Spec {
	return Spec{CommonSpec: CommonSpec{Architecture: "gemma4", BlockCount: 4, EmbeddingLength: 4,
		FeedForwardLength: 6, RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 2, ValueLength: 2, KeyLengthSWA: 2, ValueLengthSWA: 2,
		RopeDimensionCount: 2, RopeDimensionSWA: 2, RopeFrequencyBase: 10000, RopeFrequencySWA: 10000,
		AttentionScale: 1, SlidingWindow: 4,
		SlidingLayers: []bool{true, false, true, false}}, MultimodalSpec: MultimodalSpec{SharedKVLayers: 2,
		EmbeddingPerLayer: 1},
	}
}

// TestSupportsCapacityCacheAdmitsSharedKV pins which shared-KV plans take the
// capacity session: a dense decoder whose shared layers read a bounded-append
// source does, and an alternate-state decoder, which never builds its graph
// through the persistent device cache, keeps the request session.
func TestSupportsCapacityCacheAdmitsSharedKV(t *testing.T) {
	if plan := fixtureModelPlan(t, sharedKVCapacitySpec(), Weights{}); !plan.SupportsCapacityCache() {
		t.Fatal("a dense shared-KV decoder was refused the capacity session")
	}
	altUp := Spec{CommonSpec: CommonSpec{Architecture: "gemma3n", BlockCount: 21, EmbeddingLength: 4,
		FeedForwardLength: 6, RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 2, ValueLength: 2, RopeDimensionCount: 2, RopeFrequencyBase: 10000, RopeFrequencySWA: 10000,
		SlidingWindow: 4}, MultimodalSpec: MultimodalSpec{KVFromStart: 20, SharedKVLayers: 1},
	}
	if plan := fixtureModelPlan(t, altUp, Weights{}); plan.SupportsCapacityCache() {
		t.Fatal("an alternate-state shared-KV decoder was admitted to the capacity session")
	}
	llama := Spec{CommonSpec: CommonSpec{Architecture: "llama", BlockCount: 2}}
	if plan := fixtureModelPlan(t, llama, Weights{}); !plan.SupportsCapacityCache() {
		t.Fatal("a dense decoder without shared KV was refused the capacity session")
	}
}

type sharedKVPair struct {
	owner, shared DenseBlockResult
	outputs       []*tensor.Tensor
}

// buildSharedKVPair builds an owning layer and a layer sharing its KV in one
// graph over a past of the given token axis. Under CacheWriteAppend the past is
// a capacity page holding active valid tokens; under CacheWriteConcat it holds
// exactly the active tokens.
func buildSharedKVPair(
	spec Spec,
	owner, shared uint32,
	active, page uint32,
	write tensor.CacheWriteMode,
) (sharedKVPair, error) {
	builder := tensor.NewBuilder()
	if write == tensor.CacheWriteAppend {
		builder.SetCacheAppendPlan(tensor.CacheAppendPlan{
			ActiveTokens: active, SourceCapacityTokens: page, CapacityTokens: page,
		})
	}
	bound := bindFixtureSpec(spec)
	positions := []uint32{active}
	pastKey := builder.Input("past_key", dtype.F32, tensor.MustShape(2, 1, uint64(page)))
	pastValue := builder.Input("past_value", dtype.F32, tensor.MustShape(2, 1, uint64(page)))
	layerWeights := func(hasKV bool) LayerGraphWeights {
		weights := gemma4BlockInputs(builder, spec, hasKV, false)
		weights.PerLayerInput = builder.Input("per_layer_input", dtype.F32, tensor.MustShape(1, 1))
		return weights
	}
	ownerPlan := bound.PlanLayer(owner, false)
	ownerResult, err := executeCompiledLayer(blockDispatchOptions{
		Context: CachedBlockContext{
			Builder: builder, Input: builder.Input("owner_input", dtype.F32, tensor.MustShape(4, 1)),
			Positions: positions, PastKey: pastKey, PastValue: pastValue,
			Layer: owner, CacheWrite: write,
		},
		Spec: bound, Weights: layerWeights(true), Plan: &ownerPlan,
	})
	if err != nil {
		return sharedKVPair{}, err
	}
	sharedPlan := bound.PlanLayer(shared, false)
	if !sharedPlan.SharedKV || sharedPlan.KVSource != owner {
		return sharedKVPair{}, fmt.Errorf("layer %d does not share layer %d: %+v", shared, owner, sharedPlan)
	}
	sharedResult, err := executeCompiledLayer(blockDispatchOptions{
		Context: CachedBlockContext{
			Builder: builder, Input: builder.Input("shared_input", dtype.F32, tensor.MustShape(4, 1)),
			Positions: positions, PastKey: ownerResult.Key, PastValue: ownerResult.Value,
			Layer: shared, CacheWrite: write,
		},
		Spec: bound, Weights: layerWeights(false), Plan: &sharedPlan,
	})
	if err != nil {
		return sharedKVPair{}, err
	}
	return sharedKVPair{
		owner: ownerResult, shared: sharedResult,
		outputs: []*tensor.Tensor{ownerResult.Output, sharedResult.Output, ownerResult.Key, ownerResult.Value},
	}, nil
}

// TestBuildGemma4SharedKVReadsSourceAtAppendOffset pins the capacity-session
// contract of a shared layer: its attention reads the source layer's
// capacity-wide append output from the offset the source appended at, with
// the logical token count from the append plan, never from the page width.
func TestBuildGemma4SharedKVReadsSourceAtAppendOffset(t *testing.T) {
	const (
		active   = uint32(6)
		capacity = uint32(8)
	)
	spec := sharedKVCapacitySpec()
	for _, test := range []struct {
		name          string
		owner, shared uint32
	}{{"sliding", 0, 2}, {"full", 1, 3}} {
		t.Run(test.name, func(t *testing.T) {
			pair, err := buildSharedKVPair(spec, test.owner, test.shared, active, capacity, tensor.CacheWriteAppend)
			if err != nil {
				t.Fatal(err)
			}
			if pair.shared.Key != pair.owner.Key || pair.owner.Key.Op != tensor.OpCacheAppend ||
				pair.owner.Key.Shape.Dims[2] != uint64(capacity) {
				t.Fatalf("shared key = %v (%s), want the owner's capacity-wide append", pair.shared.Key.Shape.Slice(), pair.shared.Key.Op)
			}
			nodes, err := tensor.Topological(pair.shared.Output)
			if err != nil {
				t.Fatal(err)
			}
			var found bool
			for _, node := range nodes {
				if node.Op != tensor.OpAttention {
					continue
				}
				found = true
				attributes := node.Attrs.(tensor.AttentionAttributes)
				if attributes.QueryStart != active || attributes.KeyValueTokens != active+1 {
					t.Fatalf("shared attention QueryStart=%d KeyValueTokens=%d, want %d/%d",
						attributes.QueryStart, attributes.KeyValueTokens, active, active+1)
				}
			}
			if !found {
				t.Fatal("shared attention node missing")
			}
		})
	}
}

// sharedKVFeeds fills every input of the graph from its name and occurrence, so
// two builds of one structure read identical weights. The past of an append
// build holds the concat build's tokens, then poison: a read past the logical
// count changes the output.
func sharedKVFeeds(t *testing.T, outputs []*tensor.Tensor, active uint32) map[*tensor.Tensor]reference.Value {
	t.Helper()
	nodes, err := tensor.Topological(outputs...)
	if err != nil {
		t.Fatal(err)
	}
	feeds := make(map[*tensor.Tensor]reference.Value)
	seen := make(map[string]int)
	for _, node := range nodes {
		if node.Op != tensor.OpInput {
			continue
		}
		occurrence := seen[node.Name]
		seen[node.Name]++
		elements, err := node.Shape.Elements()
		if err != nil {
			t.Fatal(err)
		}
		hash := fnv.New32a()
		fmt.Fprintf(hash, "%s#%d", node.Name, occurrence)
		seed := float64(hash.Sum32() % 997)
		pastWidth := 2 * uint64(active)
		past := node.Name == "past_key" || node.Name == "past_value"
		data := make([]float32, elements)
		for index := range data {
			data[index] = float32(math.Sin(seed + float64(index)*0.73))
			if past && uint64(index) >= pastWidth {
				data[index] = 1e6
			}
		}
		feeds[node] = reference.Value{Shape: node.Shape, Data: data}
	}
	return feeds
}

// TestSharedKVAppendMatchesConcatBitForBit proves on the host reference that a
// shared layer decoding against the owner's capacity page computes the bits it
// computes against the concatenated cache: the same keys and values in the same
// order, the poison beyond the logical count never read, for the sliding window
// and the full layer alike.
func TestSharedKVAppendMatchesConcatBitForBit(t *testing.T) {
	const (
		active   = uint32(6)
		capacity = uint32(8)
	)
	spec := sharedKVCapacitySpec()
	for _, test := range []struct {
		name          string
		owner, shared uint32
	}{{"sliding", 0, 2}, {"full", 1, 3}} {
		t.Run(test.name, func(t *testing.T) {
			concat, err := buildSharedKVPair(spec, test.owner, test.shared, active, active, tensor.CacheWriteConcat)
			if err != nil {
				t.Fatal(err)
			}
			appended, err := buildSharedKVPair(spec, test.owner, test.shared, active, capacity, tensor.CacheWriteAppend)
			if err != nil {
				t.Fatal(err)
			}
			want, err := reference.Execute(concat.outputs, sharedKVFeeds(t, concat.outputs, active))
			if err != nil {
				t.Fatal(err)
			}
			got, err := reference.Execute(appended.outputs, sharedKVFeeds(t, appended.outputs, active))
			if err != nil {
				t.Fatalf("append graph: %v", err)
			}
			for index, name := range []string{"owner output", "shared output"} {
				wantValue, gotValue := want[concat.outputs[index]], got[appended.outputs[index]]
				if len(wantValue.Data) == 0 || len(wantValue.Data) != len(gotValue.Data) {
					t.Fatalf("%s lengths = %d/%d", name, len(wantValue.Data), len(gotValue.Data))
				}
				for element := range wantValue.Data {
					if math.Float32bits(wantValue.Data[element]) != math.Float32bits(gotValue.Data[element]) {
						t.Fatalf("%s[%d] = %v, concat computes %v", name, element,
							gotValue.Data[element], wantValue.Data[element])
					}
				}
			}
			// The page's valid prefix is the concat cache.
			wantKey, gotKey := want[concat.outputs[2]], got[appended.outputs[2]]
			for element := range 2 * (active + 1) {
				if math.Float32bits(wantKey.Data[element]) != math.Float32bits(gotKey.Data[element]) {
					t.Fatalf("key[%d] = %v, concat holds %v", element, gotKey.Data[element], wantKey.Data[element])
				}
			}
		})
	}
}
