//go:build windows

package executor

import (
	"bytes"
	"context"
	"errors"
	"math"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// verifySpanGraph is one compiled speculative verify span of Flash-Next's
// parts: token rows, rotary positions, in-place cache appends of the span,
// causal attention over the page, a block selection with sparse attention
// for each query, and routed experts of IQ4_XS and Q8_0 banks.
type verifySpanGraph struct {
	compiled            *CompiledGraph
	inputs              map[string]*tensor.Tensor
	logits, key, value  *tensor.Tensor
	embedded, attention *tensor.Tensor
	indices, sparse     *tensor.Tensor
	rotations, appends  []*tensor.Tensor
	scores, selected    *tensor.Tensor
	capacity, blockSize uint64
}

// buildVerifySpanGraph compiles the span of the token rows at position, every attribute fixed in the graph.
func buildVerifySpanGraph(t *testing.T, rows []uint32, position uint32) verifySpanGraph {
	t.Helper()
	const (
		// Two query heads of 128 channels fill one 256-weight IQ4_XS block of the hidden width.
		width, heads, hidden, intermediate = 128, 2, 256, 256
		// Eight blocks of four tokens; a budget of two blocks selects up to 11 tokens a query.
		capacity, blockSize, budget = 32, 4, 8
		vocabulary, experts, topK   = 16, 8, 2
	)
	span := uint64(len(rows))
	builder := tensor.NewBuilder()
	builder.SetMulMatCompute(tensor.MulMatComputeNativeTensorCore)
	builder.SetCacheAppendPlan(tensor.CacheAppendPlan{ActiveTokens: position, CapacityTokens: capacity})
	g := verifySpanGraph{inputs: make(map[string]*tensor.Tensor), capacity: capacity, blockSize: blockSize}
	input := func(name string, storage dtype.Type, dims ...uint64) *tensor.Tensor {
		g.inputs[name] = builder.Input(name, storage, tensor.MustShape(dims...))
		return g.inputs[name]
	}
	table, router := input("token_embd", dtype.F32, hidden, vocabulary), input("router", dtype.F32, hidden, experts)
	wq, wk, wv := input("wq", dtype.F32, hidden, hidden), input("wk", dtype.F32, hidden, width), input("wv", dtype.F32, hidden, width)
	wout := input("output", dtype.F32, hidden, vocabulary)
	keyCache, valueCache := input("cache_k", dtype.F32, width, 1, capacity), input("cache_v", dtype.F32, width, 1, capacity)
	g.scores, g.selected = input("block_scores", dtype.F32, capacity/blockSize, span), input("selected", dtype.F32, topK, span)
	gate := input("gate", dtype.IQ4XS, hidden, intermediate, experts)
	up := input("up", dtype.IQ4XS, hidden, intermediate, experts)
	down := input("down", dtype.Q8_0, intermediate, hidden, experts)
	positions := make([]uint32, span)
	for index := range positions {
		positions[index] = position + uint32(index)
	}
	rope := func(node *tensor.Tensor) *tensor.Tensor {
		rotated := builder.RoPEWithOptions(node, tensor.RoPEOptions{
			Layout: tensor.RoPELayoutNeoX, Positions: positions, RotaryDimensions: width,
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
	g.embedded = builder.GetRows(table, rows)
	query := rope(builder.Reshape(builder.MulMat(wq, g.embedded), width, heads, span))
	g.key = appendCache(keyCache, rope(builder.Reshape(builder.MulMat(wk, g.embedded), width, 1, span)))
	g.value = appendCache(valueCache, builder.Reshape(builder.MulMat(wv, g.embedded), width, 1, span))
	scale := float32(1 / math.Sqrt(width))
	g.attention = builder.AttentionWithOptions(query, g.key, g.value, tensor.AttentionOptions{Scale: scale, Causal: true, QueryStart: position})
	g.indices = builder.BlockTopK(g.scores, budget, blockSize, position, capacity)
	g.sparse = builder.SparseAttentionWithOffset(query, g.key, g.value, g.indices, scale, true, position)
	mixed := builder.Add(builder.Add(g.embedded, builder.Reshape(g.attention, hidden, span)), builder.Reshape(g.sparse, hidden, span))
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

// TestVerifySpanReplaysCaptured verifies three-token spans two ways. The
// launched span compiles each span's graph with its token rows, positions,
// first query and key count fixed, and runs it uncaptured. The replayed span
// compiles the first span's graph once and passes each later span's values
// as runtime attributes. The spans start at 9, 12, 13, 16 and 19, as verify
// spans do when a draft is rejected and the next span starts one token past
// the last: the selections cross blocks and the attention's first query
// moves back. Every span replays the captured graph with one graph launch
// (after the first, no kernel launch and one stream synchronization), and
// each span's logits and the final caches keep the launched span's bits.
func TestVerifySpanReplaysCaptured(t *testing.T) {
	const vocabulary = 16
	starts := []uint32{9, 12, 13, 16, 19}
	spanRows := func(span int) []uint32 {
		rows := make([]uint32, 3)
		for index := range rows {
			rows[index] = uint32(5*(span*3+index)+3) % vocabulary
		}
		return rows
	}
	g := buildVerifySpanGraph(t, []uint32{0, 0, 0}, starts[0])

	// Every weight and the past caches hold the same values on both executors.
	values := make(map[string][]byte)
	for ordinal, name := range []string{"token_embd", "router", "wq", "wk", "wv", "output", "cache_k", "cache_v", "gate", "up", "down"} {
		node := g.inputs[name]
		elements, _ := node.Shape.Elements()
		data := moePinnedValues("verify-span", uint64(ordinal), elements, 3/math.Sqrt(float64(node.Shape.Dims[0])))
		if node.Type == dtype.F32 {
			values[name] = driver.Bytes(data)
			continue
		}
		bank, err := moePinnedQuantize(node.Type, "verify-span/"+name, data)
		if err != nil {
			t.Fatal(err)
		}
		values[name] = bank
	}
	upload := func(cuda *Executor) map[string]driver.DevicePtr {
		pointers := make(map[string]driver.DevicePtr)
		for name, data := range values {
			pointers[name] = copyFixtureDeviceBytes(t, cuda.worker, data)
		}
		return pointers
	}
	// bind binds a graph's inputs by name and its appended caches in place.
	bind := func(graph verifySpanGraph, pointers map[string]driver.DevicePtr) (*DeviceInputs, *RetainedTargets) {
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
			size, _ := cache.output.Shape.Bytes(dtype.F32)
			if err := targets.Set(cache.output, DeviceValue{Pointer: pointers[cache.name], Shape: cache.output.Shape, CapacityBytes: size}); err != nil {
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

	launched, replayed := newFixtureExecutor(t), newFixtureExecutor(t)
	launchedData, replayedData := upload(launched), upload(replayed)
	inputs, targets := bind(g, replayedData)

	// The runtime attributes point at values each span rewrites, as a decode session's do.
	attributes := g.compiled.NewRuntimeAttributes()
	set := func(node *tensor.Tensor, value tensor.Attributes) {
		if err := attributes.Set(node, value); err != nil {
			t.Fatal(err)
		}
	}
	rows := g.embedded.Attrs.(tensor.GetRowsAttributes)
	rows.Rows = []uint32{0, 0, 0}
	set(g.embedded, &rows)
	var rotations []*tensor.RoPEAttributes
	for _, node := range g.rotations {
		rotation := node.Attrs.(tensor.RoPEAttributes)
		rotation.Positions = make([]uint32, 3)
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
	for span, start := range starts {
		copy(rows.Rows, spanRows(span))
		for _, rotation := range rotations {
			for index := range rotation.Positions {
				rotation.Positions[index] = start + uint32(index)
			}
		}
		for _, appended := range appends {
			appended.Offset = start
		}
		attention.QueryStart, attention.KeyValueTokens = start, start+3
		selection.QueryStart, sparse.QueryStart = start, start
		// Each token routes to two experts that change from span to span.
		feed := func(graph verifySpanGraph) map[*tensor.Tensor]reference.Value {
			selected := make([]float32, 0, 6)
			for index := range 3 {
				token := span*3 + index
				selected = append(selected, float32(token%7+1), float32((token+3)%7+1))
			}
			return map[*tensor.Tensor]reference.Value{
				graph.scores:   {Shape: graph.scores.Shape, Data: moePinnedValues("verify-span/scores", uint64(span), 3*graph.capacity/graph.blockSize, 1)},
				graph.selected: {Shape: graph.selected.Shape, Data: selected},
			}
		}
		fresh := buildVerifySpanGraph(t, spanRows(span), start)
		freshInputs, freshTargets := bind(fresh, launchedData)
		retained, err := launched.ExecuteRetainedCompiledOnce(t.Context(), fresh.compiled, feed(fresh), freshInputs, freshTargets, nil)
		want := logits(retained, err, fresh.logits)
		before := stats(replayed)
		retained, err = replayed.ExecuteRetainedCompiled(t.Context(), g.compiled, feed(g), inputs, targets, attributes)
		got := logits(retained, err, g.logits)
		after := stats(replayed)
		if graphs := after.GraphLaunches - before.GraphLaunches; graphs != 1 {
			t.Fatalf("span %d at %d launched %d graphs, want one", span, start, graphs)
		}
		if !bytes.Equal(got, want) {
			t.Fatalf("span %d at %d: the replayed span's logits differ from the launched span's", span, start)
		}
		if bytes.Equal(got, previous) {
			t.Fatalf("span %d at %d: the logits repeat the last span's, so the span ignored its inputs", span, start)
		}
		previous = got
		if span == 0 {
			continue
		}
		if kernels, waits := after.KernelLaunches-before.KernelLaunches, after.StreamSynchronizations-before.StreamSynchronizations; kernels != 0 || waits != 1 {
			t.Fatalf("span %d at %d launched %d kernels and waited for the stream %d times, want a replay with one wait", span, start, kernels, waits)
		}
	}
	if !bytes.Equal(readCaches(replayed, replayedData), readCaches(launched, launchedData)) {
		t.Fatal("the replayed spans' caches differ from the launched spans'")
	}
}

// TestExecutorSpanDecodeAttentionMatchesReference holds a causal span of up
// to eight queries, served by the decode kernel from the device key count,
// to the host reference, with and without a sliding window and a softcap,
// over a small page and over a cache whose keys split across blocks. Each
// query's row keeps, bit for bit, what the one-query decode step at its
// position gives it over the same keys.
func TestExecutorSpanDecodeAttentionMatchesReference(t *testing.T) {
	const (
		width      = 4
		queryHeads = 4
		keyHeads   = 2
	)
	cuda := newFixtureExecutor(t)
	for _, geometry := range []struct{ capacity, active uint32 }{{24, 9}, {20000, 16500}} {
		for _, queries := range []uint32{2, 5, 8} {
			for _, window := range []uint32{0, 3} {
				for _, softcap := range []float32{0, 1.5} {
					// attention builds the span's attention at position over the past and the appended keys.
					attention := func(position, count uint32) (*tensor.Tensor, map[*tensor.Tensor]reference.Value) {
						builder := tensor.NewBuilder()
						builder.SetCacheAppendPlan(tensor.CacheAppendPlan{CapacityTokens: geometry.capacity, ActiveTokens: position})
						query := builder.Input("query", dtype.F32, tensor.MustShape(width, queryHeads, uint64(count)))
						pastKey := builder.Input("past_key", dtype.F32, tensor.MustShape(width, keyHeads, uint64(geometry.capacity)))
						newKey := builder.Input("new_key", dtype.F32, tensor.MustShape(width, keyHeads, uint64(count)))
						pastValue := builder.Input("past_value", dtype.F32, tensor.MustShape(width, keyHeads, uint64(geometry.capacity)))
						newValue := builder.Input("new_value", dtype.F32, tensor.MustShape(width, keyHeads, uint64(count)))
						output := builder.AttentionWithOptions(builder.Reshape(query, width, queryHeads, uint64(count)),
							builder.AppendCache(pastKey, newKey, 2), builder.AppendCache(pastValue, newValue, 2),
							tensor.AttentionOptions{Scale: 0.5, Softcap: softcap, Causal: true, QueryStart: position, Window: window})
						if err := builder.Err(); err != nil {
							t.Fatal(err)
						}
						// The span's queries and keys, and the past keys, are one sequence of positions, so a later
						// one-query step at a span position reads the keys the span appended before it as its past.
						tokens := func(seed int, heads, first, count uint32) reference.Value {
							value := patternedValue(tensor.MustShape(width, uint64(heads), uint64(geometry.active+queries)), seed, 0.2, 0)
							start, end := first*width*heads, (first+count)*width*heads
							return reference.Value{Shape: tensor.MustShape(width, uint64(heads), uint64(count)), Data: value.Data[start:end]}
						}
						past := func(seed int) reference.Value {
							value := tokens(seed, keyHeads, 0, position)
							data := make([]float32, width*keyHeads*geometry.capacity)
							copy(data, value.Data)
							return reference.Value{Shape: pastKey.Shape, Data: data}
						}
						return output, map[*tensor.Tensor]reference.Value{
							query: tokens(1, queryHeads, position, count), newKey: tokens(3, keyHeads, position, count),
							newValue: tokens(5, keyHeads, position, count), pastKey: past(3), pastValue: past(5),
						}
					}
					run := func(output *tensor.Tensor, feeds map[*tensor.Tensor]reference.Value) []float32 {
						program, err := tensor.CompileProgram(output)
						if err != nil {
							t.Fatal(err)
						}
						got, err := cuda.ExecuteProgram(context.WithoutCancel(t.Context()), program, feeds)
						if err != nil {
							t.Fatal(err)
						}
						return got[output].Data
					}
					output, feeds := attention(geometry.active, queries)
					want, err := reference.Execute([]*tensor.Tensor{output}, feeds)
					if err != nil {
						t.Fatal(err)
					}
					got := run(output, feeds)
					compare(t, got, want[output].Data, 3e-5)
					rowWidth := uint32(width * queryHeads)
					for query := range queries {
						single, singleFeeds := attention(geometry.active+query, 1)
						compareBits(t, got[query*rowWidth:(query+1)*rowWidth], run(single, singleFeeds))
					}
				}
			}
		}
	}
}
