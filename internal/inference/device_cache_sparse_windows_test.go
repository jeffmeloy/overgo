//go:build windows

package inference

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/model"
	"overgo/internal/modelrecipe"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

func TestDeviceSparseCacheTargets(t *testing.T) {
	cudatest.Require(t)
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "qwen4exp", BlockCount: 1, ContextLength: 16, EmbeddingLength: 4, VocabularySize: 1}, AttentionSpec: model.AttentionSpec{KeyLength: 2, ValueLength: 3, HeadCountKV: 1}}
	spec.RecurrentLayers = []bool{false}
	spec.CompressRatios = []uint32{4}
	spec.IndexerKeyLength = 2
	spec.HyperConnectionCount = 4
	spec.NGram = &model.NGramSpec{Size: 3, ConvKernel: 4}
	r := fixtureRunner(spec, model.Weights{Layers: []model.LayerWeights{{}}})
	r.program.Decode.Session = modelrecipe.DecodeSessionCapacity
	var err error
	r.cuda, err = executor.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer r.cuda.Close()
	_, schema, err := r.cacheSchema(0, 2)
	if err != nil {
		t.Fatal(err)
	}
	filled := func(shape tensor.Shape, value float32) reference.Value {
		elements, err := shape.Elements()
		if err != nil {
			t.Fatal(err)
		}
		result := reference.Value{Shape: shape, Data: make([]float32, elements)}
		for i := range result.Data {
			result.Data[i] = value
		}
		return result
	}
	expected := &KVCache{Tokens: 2, Position: 2, Layers: []LayerCache{{Key: filled(schema.Primary.Key.Value.Shape, 1), Value: filled(schema.Primary.Value.Value.Shape, 2), States: make(LayerStates)}}}
	schema.RangeStates(func(name model.CacheStateName, state model.CacheState[model.CacheValueSchema]) {
		expected.Layers[0].States[name] = LayerState{Mode: state.Mode, Value: filled(state.Value.Shape, 3)}
	})
	past, err := r.uploadDeviceCache(t.Context(), expected)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if past != nil {
			if err := past.Release(context.Background()); err != nil {
				t.Error(err)
			}
		}
	}()
	check := func(cache *deviceKVCache, want *KVCache) {
		t.Helper()
		got, err := r.downloadDeviceCache(t.Context(), cache)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("active cache differs: got=%+v want=%+v", got, want)
		}
	}
	appendOne := func(cancel bool) *deviceKVCache {
		t.Helper()
		appends := []deviceBatchAppend{{Tokens: []tokenizer.TokenID{0}, Past: past}}
		appendPlan := r.cacheAppendPlan(appends, deviceOutputPlan{})
		if appendPlan.CapacityTokens == 0 {
			t.Fatalf("sparse append plan = %+v", appendPlan)
		}
		builder := tensor.NewBuilder()
		builder.SetCacheAppendPlan(appendPlan.CacheAppendPlan)
		feeds := make(map[*tensor.Tensor]reference.Value)
		var bindings tensor.InputBindings[driver.DevicePtr]
		input, err := r.deviceBatchLayerCacheInputs(builder, "", 0, past, feeds, &bindings)
		if err != nil {
			t.Fatal(err)
		}
		add := func(name string, node *tensor.Tensor) *tensor.Tensor {
			shape, _ := tensor.WithTrailingExtent(node.Shape, 1)
			next := builder.Input(name, dtype.F32, shape)
			feeds[next] = filled(shape, float32(past.Tokens+1))
			return builder.AppendCache(node, next, uint32(node.Shape.Rank-1))
		}
		graph := deviceBatchGraph{keys: []*tensor.Tensor{add("key_new", input.key)}, values: []*tensor.Tensor{add("value_new", input.value)}, states: []deviceGraphStates{make(deviceGraphStates)}, pastTokens: past.Tokens, nextPosition: past.Position, tokenCount: 1}
		for name, state := range input.states {
			node := state.Value
			if state.Mode.TokenAligned() {
				node = add(string(name)+"_new", node)
			} else {
				node = builder.Scale(node, 2)
			}
			graph.states[0][name] = model.CacheState[*tensor.Tensor]{Mode: state.Mode, Value: node}
		}
		graph.logits = builder.Input("logits", dtype.F32, tensor.MustShape(1))
		feeds[graph.logits] = filled(graph.logits.Shape, 0)
		if err := builder.Err(); err != nil {
			t.Fatal(err)
		}
		execution, err := executor.CompileIndexed(decodeGraphOutputs(graph, deviceOutputPlan{})...)
		if err != nil {
			t.Fatal(err)
		}
		if err := execution.Inputs.Bind(bindings); err != nil {
			t.Fatal(err)
		}
		graphs := []deviceBatchGraph{graph}
		plans, err := r.compileDeviceCacheTargetPlans(execution.Graph, graphs, appends)
		if err != nil {
			t.Fatal(err)
		}
		before := past.storage.refs
		targets, storages, err := r.prepareDeviceCacheTargets(t.Context(), execution.Graph, graphs, appends, plans)
		if err != nil {
			t.Fatal(err)
		}
		ctx := t.Context()
		if cancel {
			var stop context.CancelFunc
			ctx, stop = context.WithCancel(ctx)
			stop()
		}
		retained, err := r.cuda.ExecuteRetainedCompiledOnce(ctx, execution.Graph, feeds, execution.Inputs, targets, nil)
		if cancel {
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("canceled append error=%v", err)
			}
			for _, storage := range storages {
				if err := storage.release(context.Background()); err != nil {
					t.Fatal(err)
				}
			}
			if past.storage.refs != before {
				t.Fatalf("canceled append retained storage refs=%d want=%d", past.storage.refs, before)
			}
			check(past, expected)
			return nil
		}
		if err != nil {
			t.Fatal(err)
		}
		next, err := r.assembleDeviceBatchCaches(ctx, retained, storages, graphs, appends, plans, nil, nil, deviceOutputPlan{})
		if err != nil {
			t.Fatal(err)
		}
		return next[0]
	}
	for total := uint32(3); total <= 6; total++ {
		if total == 6 {
			appendOne(true)
			if !past.storage.retain() {
				t.Fatal("retain shared storage")
			}
		}
		next := appendOne(false)
		reuse := total == 4
		if (next.storage == past.storage) != reuse {
			t.Fatalf("tokens=%d reuse=%t want=%t", total, next.storage == past.storage, reuse)
		}
		if total == 6 {
			check(past, expected)
			if err := past.storage.release(t.Context()); err != nil {
				t.Fatal(err)
			}
		}
		advance := func(value reference.Value) reference.Value {
			shape, _ := tensor.WithTrailingExtent(value.Shape, uint64(total))
			count, _ := value.Shape.Elements()
			result := filled(shape, float32(total))
			copy(result.Data[:count], value.Data)
			return result
		}
		expected.Tokens, expected.Position = total, total
		expected.Layers[0].Key = advance(expected.Layers[0].Key)
		expected.Layers[0].Value = advance(expected.Layers[0].Value)
		for name, state := range expected.Layers[0].States {
			if state.Mode.TokenAligned() {
				state.Value = advance(state.Value)
			} else {
				for i := range state.Value.Data {
					state.Value.Data[i] *= 2
				}
			}
			expected.Layers[0].States[name] = state
		}
		check(next, expected)
		if err := past.Release(t.Context()); err != nil {
			t.Fatal(err)
		}
		past = next
	}
}
