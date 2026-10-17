package model

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func TestFlashNextSparseAttention(t *testing.T) {
	RunFlashNextSparseAttention(t, reference.Execute)
}

// RunFlashNextSparseAttention shares original-model inputs and assertions between host and CUDA tests.
func RunFlashNextSparseAttention(t *testing.T, run func([]*tensor.Tensor, map[*tensor.Tensor]reference.Value) (map[*tensor.Tensor]reference.Value, error)) {
	t.Helper()
	data, err := os.ReadFile("testdata/flashnext_sparse.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Tolerance     float64
		Ratio, Budget uint32
		Weights       map[string]struct {
			Shape []uint64
			Data  []float32
		}
		Cases []struct {
			Name                           string
			Start, Tokens                  uint32
			Positions                      [tensor.MaxDimensions][]uint32
			Input, Output, Raw, Key, Value []float32
			Selected                       []bool
		}
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	spec := qwen35TestSpec()
	spec.Architecture, spec.BlockCount = "qwen4exp", 1
	spec.RecurrentLayers = []bool{false}
	spec.CompressRatios = []uint32{fixture.Ratio}
	spec.IndexerHeadCount, spec.IndexerKeyLength, spec.IndexerTopK = 2, 4, fixture.Budget
	spec.HyperConnectionCount, spec.HyperConnectionRank = 4, 3
	spec = bindFixtureSpec(spec)
	for _, mode := range []tensor.CacheWriteMode{tensor.CacheWriteConcat, tensor.CacheWriteAppend} {
		var pastKey, pastValue reference.Value
		states := map[CacheStateName]reference.Value{}
		for _, test := range fixture.Cases {
			if !t.Run(fmt.Sprintf("cache-%d/%s/%d", mode, test.Name, test.Start), func(t *testing.T) {
				builder := tensor.NewBuilder()
				if mode == tensor.CacheWriteAppend {
					capacity := fixture.Cases[0].Tokens
					builder.SetCacheAppendPlan(tensor.CacheAppendPlan{ActiveTokens: test.Start, CapacityTokens: capacity})
					if test.Start == 0 {
						pastKey = reference.ZeroValue(tensor.MustShape(uint64(spec.KeyLength), uint64(spec.HeadCountKV), uint64(capacity)))
						pastValue = reference.ZeroValue(pastKey.Shape)
						states[CacheStateIndexerKey] = reference.ZeroValue(tensor.MustShape(uint64(spec.IndexerKeyLength), 1, uint64(capacity)))
						states[CacheStatePositions] = reference.ZeroValue(tensor.MustShape(tensor.MaxDimensions, 1, uint64(capacity)))
					}
				}
				feeds := map[*tensor.Tensor]reference.Value{}
				input := func(name string, value reference.Value) *tensor.Tensor {
					node := builder.Input(name, dtype.F32, value.Shape)
					feeds[node] = value
					return node
				}
				weight := func(name string) *tensor.Tensor {
					value := fixture.Weights[name+".weight"]
					return input(name, reference.Value{Shape: tensor.MustShape(value.Shape...), Data: value.Data})
				}
				weights := LayerGraphWeights{
					AttentionQ: weight("attn_q"), AttentionK: weight("attn_k"), AttentionV: weight("attn_v"),
					AttentionOutput: weight("attn_output"), AttentionQNorm: weight("attn_q_norm"), AttentionKNorm: weight("attn_k_norm"),
					IndexerAttentionQB: weight("index_q_proj"), IndexerAttentionK: weight("index_k_proj"),
					IndexerQNorm: weight("index_q_norm"), IndexerKNorm: weight("index_k_norm"),
				}
				positions := make([]uint32, test.Tokens)
				coordinates := make([]float32, int(test.Tokens)*tensor.MaxDimensions)
				for token := range positions {
					positions[token] = test.Start + uint32(token)
					for axis := range tensor.MaxDimensions {
						coordinates[token*tensor.MaxDimensions+axis] = float32(test.Positions[axis][token])
					}
				}
				context := CachedBlockContext{Builder: builder, Input: input("input", reference.Value{Shape: tensor.MustShape(8, uint64(test.Tokens)), Data: test.Input}),
					Positions: positions, MultiPositions: &test.Positions, Sequences: 1, CacheWrite: mode,
					CurrentPositions: input("positions", reference.Value{Shape: tensor.MustShape(tensor.MaxDimensions, 1, uint64(test.Tokens)), Data: coordinates}),
					PastStates:       CacheStates[*tensor.Tensor]{},
				}
				if test.Start != 0 || mode == tensor.CacheWriteAppend {
					context.PastKey, context.PastValue = input("past_key", pastKey), input("past_value", pastValue)
					for name, value := range states {
						context.PastStates[name] = CacheState[*tensor.Tensor]{Mode: CacheStateToken, Value: input(string(name), value)}
					}
				}
				plan := spec.PlanLayer(0, false)
				options := blockDispatchOptions{Context: context, Spec: spec, Weights: weights, Plan: &plan}
				operands := layerOperands{}
				operands.caches[runtimeCachePrimaryKey], operands.caches[runtimeCachePrimaryValue] = context.PastKey, context.PastValue
				result, err := executeAttentionOperator(options, plan, layerOperatorAttentionGatedProjection, operands, &layerExecution{current: context.Input})
				if err != nil {
					t.Fatal(err)
				}
				outputs := result.States.AppendValues([]*tensor.Tensor{result.Output, result.Key, result.Value})
				var selected *tensor.Tensor
				for _, node := range builder.Nodes() {
					if node.Op == tensor.OpTopK && node.Attrs.(tensor.TopKAttributes).BlockSize != 0 {
						selected = node
						outputs = append(outputs, node)
					}
				}
				values, err := run(outputs, feeds)
				if err != nil {
					t.Fatal(err)
				}
				for _, check := range []struct {
					name string
					node *tensor.Tensor
					want []float32
				}{
					{"output", result.Output, test.Output}, {"key", result.Key, test.Key}, {"value", result.Value, test.Value},
					{"raw indexer key", result.States[CacheStateIndexerKey].Value, test.Raw},
				} {
					got := values[check.node].Data
					if mode == tensor.CacheWriteAppend && check.node != result.Output && len(got) >= len(check.want) {
						got = got[:len(check.want)]
					}
					if len(got) != len(check.want) {
						t.Fatalf("%s length %d, want %d", check.name, len(got), len(check.want))
					}
					for i, want := range check.want {
						difference := math.Abs(float64(got[i] - want))
						if math.IsNaN(difference) || difference > fixture.Tolerance*(1+math.Abs(float64(want))) {
							t.Fatalf("%s[%d] = %.9g, original %.9g", check.name, i, got[i], want)
						}
					}
				}
				total := int(test.Start + test.Tokens)
				mask := make([]bool, int(test.Tokens)*total)
				for query := range int(test.Tokens) {
					if selected == nil {
						for key := 0; key <= int(test.Start)+query; key++ {
							mask[query*total+key] = true
						}
					} else {
						width := int(selected.Shape.Dims[0])
						for _, raw := range values[selected].Data[query*width : (query+1)*width] {
							if raw == -1 {
								continue
							}
							key := int(raw)
							if float32(key) != raw || key < 0 || key > int(test.Start)+query || mask[query*total+key] {
								t.Fatalf("invalid or duplicate selected token %g for query %d", raw, query)
							}
							mask[query*total+key] = true
						}
					}
				}
				if len(mask) != len(test.Selected) {
					t.Fatalf("selection length %d, want %d", len(mask), len(test.Selected))
				}
				for i, want := range test.Selected {
					if mask[i] != want {
						t.Fatalf("selected[%d] = %t, original %t", i, mask[i], want)
					}
				}
				pastKey, pastValue = values[result.Key], values[result.Value]
				for name, state := range result.States {
					states[name] = values[state.Value]
				}
			}) {
				return
			}
		}
	}
}
