//go:build windows

package executor

import (
	"bytes"
	"cmp"
	"errors"
	"maps"
	"math"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/gguf"
	"overgo/internal/modeldevice"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// decodeStepGraph is one compiled decode step of Flash-Next's parts: a
// token row, rotary positions, an in-place cache append, decode attention,
// a block selection with sparse attention over the page, and routed experts
// of IQ4_XS and Q8_0 banks.
type decodeStepGraph struct {
	compiled                                  *CompiledGraph
	inputs                                    map[string]*tensor.Tensor
	logits, key, value                        *tensor.Tensor
	embedded, attention, indices, sparse      *tensor.Tensor
	rotations, appends                        []*tensor.Tensor
	scores, selected                          *tensor.Tensor
	width, capacity, blockSize, experts, topK uint64
}

// buildDecodeStepGraph compiles the step of the token row at position, every attribute fixed in the graph.
func buildDecodeStepGraph(t *testing.T, row, position uint32) decodeStepGraph {
	t.Helper()
	const (
		// Two query heads of 128 channels fill one 256-weight IQ4_XS block of the hidden width.
		width, heads, hidden, intermediate = 128, 2, 256, 256
		// Eight blocks of four tokens; a budget of two blocks selects 11 tokens.
		capacity, blockSize, budget = 32, 4, 8
		vocabulary, experts, topK   = 16, 8, 2
	)
	builder := tensor.NewBuilder()
	builder.SetMulMatCompute(tensor.MulMatComputeNativeTensorCore)
	builder.SetCacheAppendPlan(tensor.CacheAppendPlan{ActiveTokens: position, CapacityTokens: capacity})
	g := decodeStepGraph{inputs: make(map[string]*tensor.Tensor), width: width, capacity: capacity, blockSize: blockSize, experts: experts, topK: topK}
	input := func(name string, storage dtype.Type, dims ...uint64) *tensor.Tensor {
		g.inputs[name] = builder.Input(name, storage, tensor.MustShape(dims...))
		return g.inputs[name]
	}
	table, router := input("token_embd", dtype.F32, hidden, vocabulary), input("router", dtype.F32, hidden, experts)
	wq, wk, wv := input("wq", dtype.F32, hidden, hidden), input("wk", dtype.F32, hidden, width), input("wv", dtype.F32, hidden, width)
	wout := input("output", dtype.F32, hidden, vocabulary)
	keyCache, valueCache := input("cache_k", dtype.F32, width, 1, capacity), input("cache_v", dtype.F32, width, 1, capacity)
	g.scores, g.selected = input("block_scores", dtype.F32, capacity/blockSize, 1), input("selected", dtype.F32, topK, 1)
	gate := input("gate", dtype.IQ4XS, hidden, intermediate, experts)
	up := input("up", dtype.IQ4XS, hidden, intermediate, experts)
	down := input("down", dtype.Q8_0, intermediate, hidden, experts)
	rope := func(node *tensor.Tensor) *tensor.Tensor {
		rotated := builder.RoPEWithOptions(node, tensor.RoPEOptions{
			Layout: tensor.RoPELayoutNeoX, Positions: []uint32{position}, RotaryDimensions: width,
			FrequencyBase: 10000, FrequencyScale: tensor.UnitFrequencyScale,
		})
		g.rotations = append(g.rotations, rotated)
		return rotated
	}
	appendCache := func(cache, node *tensor.Tensor) *tensor.Tensor {
		appended := builder.AppendCache(cache, node, 2)
		g.appends = append(g.appends, appended)
		return appended
	}
	g.embedded = builder.GetRows(table, []uint32{row})
	query := rope(builder.Reshape(builder.MulMat(wq, g.embedded), width, heads, 1))
	g.key = appendCache(keyCache, rope(builder.Reshape(builder.MulMat(wk, g.embedded), width, 1, 1)))
	g.value = appendCache(valueCache, builder.Reshape(builder.MulMat(wv, g.embedded), width, 1, 1))
	scale := float32(1 / math.Sqrt(width))
	g.attention = builder.AttentionWithOptions(query, g.key, g.value, tensor.AttentionOptions{Scale: scale, Causal: true, QueryStart: position})
	g.indices = builder.BlockTopK(g.scores, budget, blockSize, position, capacity)
	g.sparse = builder.SparseAttentionWithOffset(query, g.key, g.value, g.indices, scale, true, position)
	mixed := builder.Add(builder.Add(g.embedded, builder.Reshape(g.attention, hidden, 1)), builder.Reshape(g.sparse, hidden, 1))
	routed := builder.MoEWithOptions(mixed, router, up, down, tensor.MoEOptions{
		Gate: gate, SelectedExperts: g.selected, TopK: topK, NormalizeTopKProb: true, Scale: 1,
		Routing: tensor.MoERoutingSoftmax, Activation: tensor.MoEActivationSiLU, ExpertIndexDivisor: 1,
	})
	g.logits = builder.MulMat(wout, builder.Add(mixed, routed))
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	var err error
	if g.compiled, err = Compile(g.logits, g.key, g.value); err != nil {
		t.Fatal(err)
	}
	return g
}

// TestDecodeStepReplaysAsOneGraph decodes several tokens two ways. The
// launched step compiles each token's graph with its token row, position,
// first query and key count fixed, and runs it uncaptured with every expert
// on the card. The replayed step compiles the first token's graph once and
// passes each later token's values as runtime attributes, with only expert
// 0 of each bank on the card, so each token's routed experts are first read
// from mapped host memory and the residency moves experts between tokens.
// Every token after the first replays the captured graph with one graph
// launch, no kernel launch and one stream synchronization, and each token's
// logits and the final caches keep the launched step's bits.
func TestDecodeStepReplaysAsOneGraph(t *testing.T) {
	// The first query is 9 and six tokens follow, so the selection crosses blocks at 11 and 15.
	const past, steps, vocabulary = 9, 6, 16
	replayedGraph := buildDecodeStepGraph(t, 0, past)
	g := replayedGraph

	// Every weight and the past caches hold the same values on both executors.
	values := make(map[string][]byte)
	encoded := make(map[string][]byte)
	var fileBanks []gguf.TensorData
	var smallest uint64
	for ordinal, name := range []string{"token_embd", "router", "wq", "wk", "wv", "output", "cache_k", "cache_v", "gate", "up", "down"} {
		node := g.inputs[name]
		elements, _ := node.Shape.Elements()
		data := moePinnedValues("decode-graph", uint64(ordinal), elements, 3/math.Sqrt(float64(node.Shape.Dims[0])))
		if node.Type == dtype.F32 {
			values[name] = driver.Bytes(data)
			continue
		}
		bank, err := moePinnedQuantize(node.Type, "decode-graph/"+name, data)
		if err != nil {
			t.Fatal(err)
		}
		encoded[name] = bank
		fileBanks = append(fileBanks, gguf.TensorData{Name: name, Shape: node.Shape.Slice(), Type: node.Type, Data: bytes.NewReader(bank)})
		expert := uint64(len(bank)) / g.experts
		smallest = min(cmp.Or(smallest, expert), expert)
	}
	upload := func(cuda *Executor, data map[string][]byte) map[string]driver.DevicePtr {
		pointers := make(map[string]driver.DevicePtr)
		for name, bytes := range data {
			pointers[name] = copyFixtureDeviceBytes(t, cuda.worker, bytes)
		}
		return pointers
	}
	// bind binds a graph's inputs by name and its appended caches in place.
	bind := func(graph decodeStepGraph, pointers map[string]driver.DevicePtr) (*DeviceInputs, *RetainedTargets) {
		inputs, targets := graph.compiled.NewDeviceInputs(), graph.compiled.NewRetainedTargets()
		for name, pointer := range pointers {
			if err := inputs.Set(graph.inputs[name], pointer); err != nil {
				t.Fatal(err)
			}
		}
		for _, cache := range []struct {
			output *tensor.Tensor
			name   string
		}{{graph.key, "cache_k"}, {graph.value, "cache_v"}} {
			bytes, _ := cache.output.Shape.Bytes(dtype.F32)
			if err := targets.Set(cache.output, DeviceValue{Pointer: pointers[cache.name], Shape: cache.output.Shape, CapacityBytes: bytes}); err != nil {
				t.Fatal(err)
			}
		}
		return inputs, targets
	}
	stats := func(cuda *Executor) driver.ExecutionStats {
		var stats driver.ExecutionStats
		if err := cuda.worker.Do(t.Context(), func(state *device.State) error { stats = state.Driver.ExecutionStats(); return nil }); err != nil {
			t.Fatal(err)
		}
		return stats
	}
	readCaches := func(cuda *Executor, pointers map[string]driver.DevicePtr) []byte {
		var held []byte
		if err := cuda.worker.Do(t.Context(), func(state *device.State) error {
			for _, name := range []string{"cache_k", "cache_v"} {
				data := make([]byte, len(values[name]))
				if err := state.Driver.MemcpyDtoH(data, pointers[name]); err != nil {
					return err
				}
				held = append(held, data...)
			}
			return nil
		}); err != nil {
			t.Fatal(err)
		}
		return held
	}
	logits := func(retained *RetainedOutputs, err error, output *tensor.Tensor) []byte {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		host, copyErr := retained.CopyToHost(t.Context(), output)
		if err := errors.Join(copyErr, retained.Release(t.Context())); err != nil {
			t.Fatal(err)
		}
		return bytes.Clone(driver.Bytes(host.Data))
	}

	launched := newFixtureExecutor(t)
	launchedData := upload(launched, values)
	maps.Copy(launchedData, upload(launched, encoded))

	// The residency holds each bank's table and expert 0, so every routed expert starts in mapped host memory.
	replayed := newFixtureExecutor(t)
	replayedData := upload(replayed, values)
	var file bytes.Buffer
	if err := gguf.Write(&file, nil, fileBanks, gguf.WriteOptions{}); err != nil {
		t.Fatal(err)
	}
	parsed, err := gguf.Parse(bytes.NewReader(file.Bytes()), uint64(file.Len()), gguf.DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	weights, err := modeldevice.NewDeviceWeights(replayed.worker)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := errors.Join(replayed.SetExpertBanks(nil), weights.Close()); err != nil {
			t.Error(err)
		}
	})
	const pointerBytes, slotAlignment = 8, 256
	limit := uint64(len(encoded)) * g.experts * pointerBytes
	for _, bank := range encoded {
		limit += (uint64(len(bank))/g.experts + slotAlignment - 1) / slotAlignment * slotAlignment
	}
	loader := modeldevice.NewLoader(parsed)
	loader.Experts(weights, parsed.Tensors, limit)
	if err := loader.Load(t.Context(), nil); err != nil {
		t.Fatal(err)
	}
	if err := replayed.SetExpertBanks(weights); err != nil {
		t.Fatal(err)
	}
	for name := range encoded {
		if _, replayedData[name], err = weights.Input(tensor.NewBuilder(), name); err != nil {
			t.Fatal(err)
		}
	}
	inputs, targets := bind(g, replayedData)

	// The runtime attributes point at values each token rewrites, as a decode session's do.
	attributes := g.compiled.NewRuntimeAttributes()
	set := func(node *tensor.Tensor, value tensor.Attributes) {
		if err := attributes.Set(node, value); err != nil {
			t.Fatal(err)
		}
	}
	rows := g.embedded.Attrs.(tensor.GetRowsAttributes)
	rows.Rows = []uint32{0}
	set(g.embedded, &rows)
	var rotations []*tensor.RoPEAttributes
	for _, node := range g.rotations {
		rotation := node.Attrs.(tensor.RoPEAttributes)
		rotation.Positions = []uint32{past}
		rotations = append(rotations, &rotation)
		set(node, &rotation)
	}
	var appends []*tensor.CacheAppendAttributes
	for _, node := range g.appends {
		appended := node.Attrs.(tensor.CacheAppendAttributes)
		appends = append(appends, &appended)
		set(node, &appended)
	}
	attention := g.attention.Attrs.(tensor.AttentionAttributes)
	selection := g.indices.Attrs.(tensor.TopKAttributes)
	sparse := g.sparse.Attrs.(tensor.SparseAttentionAttributes)
	set(g.attention, &attention)
	set(g.indices, &selection)
	set(g.sparse, &sparse)

	var previous []byte
	for token := range uint32(steps) {
		position, row := past+token, (5*token+3)%vocabulary
		rows.Rows[0] = row
		for _, rotation := range rotations {
			rotation.Positions[0] = position
		}
		for _, appended := range appends {
			appended.Offset = position
		}
		attention.QueryStart, attention.KeyValueTokens = position, position+1
		selection.QueryStart, sparse.QueryStart = position, position
		// Token t routes to experts t%7+1 and (t+3)%7+1: never expert 0, and never an expert the last token routed.
		feed := func(graph decodeStepGraph) map[*tensor.Tensor]reference.Value {
			return map[*tensor.Tensor]reference.Value{
				graph.scores:   {Shape: graph.scores.Shape, Data: moePinnedValues("decode-graph/scores", uint64(token), graph.capacity/graph.blockSize, 1)},
				graph.selected: {Shape: graph.selected.Shape, Data: []float32{float32(token%7 + 1), float32((token+3)%7 + 1)}},
			}
		}
		fresh := buildDecodeStepGraph(t, row, position)
		freshInputs, freshTargets := bind(fresh, launchedData)
		retained, err := launched.ExecuteRetainedCompiledOnce(t.Context(), fresh.compiled, feed(fresh), freshInputs, freshTargets, nil)
		want := logits(retained, err, fresh.logits)
		before := stats(replayed)
		retained, err = replayed.ExecuteRetainedCompiled(t.Context(), g.compiled, feed(g), inputs, targets, attributes)
		got := logits(retained, err, g.logits)
		after := stats(replayed)
		if !bytes.Equal(got, want) {
			t.Fatalf("token %d: the replayed step's logits differ from the launched step's", token)
		}
		if bytes.Equal(got, previous) {
			t.Fatalf("token %d: the logits repeat the last token's, so the step ignored its inputs", token)
		}
		previous = got
		if graphs := after.GraphLaunches - before.GraphLaunches; graphs != 1 {
			t.Fatalf("token %d launched %d graphs, want one", token, graphs)
		}
		if token == 0 {
			continue
		}
		if kernels, waits := after.KernelLaunches-before.KernelLaunches, after.StreamSynchronizations-before.StreamSynchronizations; kernels != 0 || waits != 1 {
			t.Fatalf("token %d launched %d kernels and waited for the stream %d times, want a replay with one wait", token, kernels, waits)
		}
		if moved := after.HostToDeviceBytes - before.HostToDeviceBytes; moved < smallest {
			t.Fatalf("token %d moved %d bytes to the card, less than one routed expert of the last token", token, moved)
		}
	}
	if !bytes.Equal(readCaches(replayed, replayedData), readCaches(launched, launchedData)) {
		t.Fatal("the replayed steps' caches differ from the launched steps'")
	}
}
