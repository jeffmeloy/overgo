package inference

import (
	"testing"

	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/model"
	"overgo/internal/modelrecipe"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tokenizer"
)

func TestParameterizedDecodeCapacityStaysWithinPageClass(t *testing.T) {
	const (
		pageTokens    = uint32(4)
		contextTokens = uint32(16)
	)
	spec := model.Spec{CommonSpec: model.CommonSpec{
		Architecture: "llama", BlockCount: 1, ContextLength: contextTokens,
	}}
	runner := &Runner{preparedModel: preparedModel{
		spec: spec,
		program: modelrecipe.Plan{Model: fixtureProgram(spec, model.Weights{}).Model, Decode: modelrecipe.DecodePlan{
			Session: modelrecipe.DecodeSessionCapacity,
		}},
	}}
	appendFor := func(tokens uint32) []deviceBatchAppend {
		return []deviceBatchAppend{{
			Tokens: []tokenizer.TokenID{1}, PageTokens: pageTokens,
			Past: &deviceKVCache{
				storage: &deviceCacheStorage{refs: 1}, Tokens: tokens,
				Selection: executor.DeviceValue{Pointer: driver.DevicePtr(1)},
			},
		}}
	}
	plan := deviceOutputPlan{mode: deviceOutputGreedy}
	if capacity, ok := runner.parameterizedDecodeCapacity(appendFor(2), plan); !ok || capacity != pageTokens {
		t.Fatalf("within-page capacity = %d/%t", capacity, ok)
	}
	if capacity, ok := runner.parameterizedDecodeCapacity(appendFor(pageTokens), plan); !ok || capacity != pageTokens*2 {
		t.Fatalf("boundary capacity = %d/%t", capacity, ok)
	}
	if capacity, ok := runner.parameterizedDecodeCapacity(appendFor(2), deviceOutputPlan{}); !ok || capacity != pageTokens {
		t.Fatalf("buffered decode capacity = %d/%t", capacity, ok)
	}
}

func TestParameterizedDecodeAttributesCoverDynamicNodes(t *testing.T) {
	const (
		activeTokens   = uint32(2)
		capacityTokens = uint32(4)
	)
	builder := tensor.NewBuilder()
	builder.SetCacheAppendPlan(tensor.CacheAppendPlan{
		ActiveTokens: activeTokens, CapacityTokens: capacityTokens,
	})
	cache := builder.Input("cache", dtype.F32, tensor.MustShape(2, 1, uint64(capacityTokens)))
	appendValue := builder.Input("append", dtype.F32, tensor.MustShape(2, 1, 1))
	key := builder.AppendCache(cache, appendValue, 2)
	table := builder.Input("table", dtype.F32, tensor.MustShape(2, 4))
	tokenRows := builder.GetRows(table, []uint32{1})
	query := builder.Reshape(tokenRows, 2, 1, 1)
	query = builder.RoPENeoX(query, []uint32{activeTokens}, 2, 10_000)
	output := builder.AttentionWithOffset(query, key, key, 1, true, activeTokens)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	compiled, err := executor.Compile(output, key)
	if err != nil {
		t.Fatal(err)
	}
	program, err := compileDecodeSessionPlan(
		compiled,
		[]deviceBatchGraph{{
			logits: output, tokenRows: tokenRows,
			keys: []*tensor.Tensor{key}, values: []*tensor.Tensor{key},
			states: []deviceGraphStates{nil},
		}},
		[]deviceCacheTargetPlan{{}},
		decodeSessionIdentity{
			capacity: capacityTokens, pageTokens: capacityTokens,
			branches: 1, tokenCount: 1,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if err = program.updateBranch(0, 2, 3, 3); err != nil {
		t.Fatal(err)
	}
	var updateErr error
	allocations := testing.AllocsPerRun(100, func() {
		updateErr = program.updateBranch(0, 2, 3, 3)
	})
	if updateErr != nil {
		t.Fatal(updateErr)
	}
	if allocations != 0 {
		t.Fatalf("decode attribute update allocations = %g", allocations)
	}
}
