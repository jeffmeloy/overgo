//go:build windows

package executor

import (
	"bytes"
	"context"
	"math"
	"testing"

	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/testcheck"
)

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
	g := buildDecodeStepGraph(t, []uint32{0, 0, 0}, starts[0])

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

	launched, replayed := newFixtureExecutor(t), newFixtureExecutor(t)
	launchedData, replayedData := upload(launched), upload(replayed)
	inputs, targets := bindDecodeStep(t, g, replayedData)

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
		feed := func(graph decodeStepGraph) map[*tensor.Tensor]reference.Value {
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
		fresh := buildDecodeStepGraph(t, spanRows(span), start)
		freshInputs, freshTargets := bindDecodeStep(t, fresh, launchedData)
		retained, err := launched.ExecuteRetainedCompiledOnce(t.Context(), fresh.compiled, feed(fresh), freshInputs, freshTargets, nil)
		want := decodeLogits(t, retained, err, fresh.logits)
		before := testcheck.Must(replayed.worker.ExecutionStats(t.Context()))(t)
		retained, err = replayed.ExecuteRetainedCompiled(t.Context(), g.compiled, feed(g), inputs, targets, attributes)
		got := decodeLogits(t, retained, err, g.logits)
		after := testcheck.Must(replayed.worker.ExecutionStats(t.Context()))(t)
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
	if !bytes.Equal(readDecodeCaches(t, replayed, replayedData, values), readDecodeCaches(t, launched, launchedData, values)) {
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
