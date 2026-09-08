package inference

import (
	"testing"

	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/model"
	"overgo/internal/modelrecipe"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/planner"
	"overgo/internal/tokenizer"
)

func TestDecodeStateOutputsDoNotRetainLayerHistories(t *testing.T) {
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(1024, 1))
	firstHistory := builder.Scale(input, 2)
	firstState := builder.GroupSlice(firstHistory, 0, 1, 1, 1)
	secondHistory := builder.Scale(builder.Scale(firstHistory, 3), 4)
	secondState := builder.GroupSlice(secondHistory, 0, 1, 1, 1)
	logits := builder.Scale(secondHistory, 5)
	outputs := decodeGraphOutputs(deviceBatchGraph{
		logits: logits, keys: []*tensor.Tensor{firstState, secondState},
		values: []*tensor.Tensor{firstState, secondState}, states: make([]deviceGraphStates, 2),
	}, deviceOutputPlan{mode: deviceOutputLogits})
	if len(outputs) != 3 {
		t.Fatalf("decode outputs duplicated or omitted: %d", len(outputs))
	}
	contract, err := tensor.CompileOutputTargetContract(logits)
	if err != nil {
		t.Fatal(err)
	}
	retained := make(map[*tensor.Tensor]struct{}, len(outputs))
	for _, output := range outputs {
		retained[output] = struct{}{}
	}
	plan, err := planner.BuildWithRewrites(outputs, contract.Alignment, nil, retained)
	if err != nil {
		t.Fatal(err)
	}
	historyBytes, err := firstHistory.Shape.Bytes(dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	stateBytes, err := firstState.Shape.Bytes(dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	if limit := 2*historyBytes + 2*stateBytes; plan.ArenaSize > limit {
		t.Fatalf("state collection retains full histories: arena=%d, live working-set bound=%d", plan.ArenaSize, limit)
	}
}

func TestParameterizedDecodeCapacityStaysWithinPageClass(t *testing.T) {
	const (
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
			Tokens: []tokenizer.TokenID{1},
			Past: &deviceKVCache{
				storage: &deviceCacheStorage{refs: 1}, Tokens: tokens,
				Selection: executor.DeviceValue{Pointer: driver.DevicePtr(1)},
			},
		}}
	}
	plan := deviceOutputPlan{mode: deviceOutputGreedy}
	if capacity, _, ok := runner.parameterizedDecodeCapacity(appendFor(2), plan); !ok ||
		capacity != cachePageCapacity(3, contextTokens, modelrecipe.DecodeSessionCapacity) {
		t.Fatalf("within-page capacity = %d/%t", capacity, ok)
	}
	if capacity, _, ok := runner.parameterizedDecodeCapacity(appendFor(4), plan); !ok ||
		capacity != cachePageCapacity(5, contextTokens, modelrecipe.DecodeSessionCapacity) {
		t.Fatalf("boundary capacity = %d/%t", capacity, ok)
	}
	if capacity, _, ok := runner.parameterizedDecodeCapacity(appendFor(2), deviceOutputPlan{}); !ok ||
		capacity != cachePageCapacity(3, contextTokens, modelrecipe.DecodeSessionCapacity) {
		t.Fatalf("buffered decode capacity = %d/%t", capacity, ok)
	}
	// Two pasts that append into the same target page can sit on pages of
	// different capacity: 4 tokens on a 4-token page and 6 on an 8-token
	// page both append into 8. The derivation reports the source page so
	// the session identity keeps them apart (the BBH chat fault: a session
	// compiled over a 512-token page served a 256-token past).
	onBand, onBandSource, ok := runner.parameterizedDecodeCapacity(appendFor(4), plan)
	if !ok || onBand != 8 || onBandSource != 4 {
		t.Fatalf("on-band past = capacity %d source %d ok %t, want 8/4", onBand, onBandSource, ok)
	}
	beyond, beyondSource, ok := runner.parameterizedDecodeCapacity(appendFor(6), plan)
	if !ok || beyond != 8 || beyondSource != 8 {
		t.Fatalf("beyond-band past = capacity %d source %d ok %t, want 8/8", beyond, beyondSource, ok)
	}
	var lora [32]byte
	compiledBeyond := decodeSessionIdentity{capacity: beyond, source: beyondSource, branches: 1, tokenCount: 1, output: plan, lora: lora}
	if compiledBeyond.matches(onBand, onBandSource, 1, plan, lora) {
		t.Fatal("a session compiled over the larger page matched the on-band past")
	}
	if !compiledBeyond.matches(beyond, beyondSource, 1, plan, lora) {
		t.Fatal("the session does not match its own past class")
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
	query = builder.RoPEWithOptions(query, tensor.RoPEOptions{Layout: tensor.RoPELayoutNeoX, Positions: []uint32{activeTokens}, RotaryDimensions: 2, FrequencyBase: 10_000, FrequencyScale: 1})
	output := builder.AttentionWithOptions(query, key, key, tensor.AttentionOptions{Scale: 1, Causal: true, QueryStart: activeTokens})
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
			logits: output, tokenInput: tokenRows,
			keys: []*tensor.Tensor{key}, values: []*tensor.Tensor{key},
			states:      []deviceGraphStates{nil},
			cacheInputs: []layerGraphCacheInputs{{key: cache, value: cache}},
		}},
		[]deviceCacheTargetPlan{{}},
		decodeSessionIdentity{
			capacity: capacityTokens,
			branches: 1, tokenCount: 1,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	rows := []uint32{2}
	if err = program.updateBranch(0, rows, 3, 3); err != nil {
		t.Fatal(err)
	}
	var updateErr error
	allocations := testing.AllocsPerRun(100, func() {
		updateErr = program.updateBranch(0, rows, 3, 3)
	})
	if updateErr != nil {
		t.Fatal(updateErr)
	}
	if allocations != 0 {
		t.Fatalf("decode attribute update allocations = %g", allocations)
	}

	// Binding a past whose page is smaller than the page the session was
	// compiled to read refuses: the append copy would otherwise read the
	// compiled extent off the end of the page (the BBH chat fault).
	session := &deviceDecodeSession{
		execution: &executor.IndexedGraph{Graph: compiled, Inputs: compiled.NewDeviceInputs()},
		program:   program,
	}
	needed, ok := compiled.InputBytes(program.branches[0].cacheInputs[0].key.slot)
	if !ok || needed != 2*1*uint64(capacityTokens)*4 {
		t.Fatalf("compiled cache input bytes = %d/%t", needed, ok)
	}
	pastFor := func(capacityBytes uint64) []deviceBatchAppend {
		value := executor.DeviceValue{Pointer: driver.DevicePtr(1), CapacityBytes: capacityBytes}
		return []deviceBatchAppend{{
			Tokens: []tokenizer.TokenID{1},
			Past: &deviceKVCache{
				Keys: []executor.DeviceValue{value}, Values: []executor.DeviceValue{value},
				States: []deviceLayerStates{nil}, Selection: value,
			},
		}}
	}
	if err := bindParameterizedDecodeInputs(session, pastFor(needed/2)); err == nil {
		t.Fatal("a past page half the compiled extent was bound")
	}
	if err := bindParameterizedDecodeInputs(session, pastFor(needed)); err != nil {
		t.Fatalf("a past page of the compiled extent was refused: %v", err)
	}
}
