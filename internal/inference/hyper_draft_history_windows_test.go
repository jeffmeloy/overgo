//go:build windows

package inference

import (
	"context"
	"errors"
	"math"
	"testing"

	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/modeldevice"
	"overgo/internal/recipe"
	"overgo/internal/sampling"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

// TestFlashNextMTPHistoryDraftsAsOneRead: the hermetic Flash-Next's random draft layer, given a greedy answer's cells
// round by round, one, two and three a round, drafts each round's token as the layer reading every cell up to it at
// once drafts it, so the history holds each cell where the layer's own attention would, and a round that does not
// follow the history's last cell starts it again.
func TestFlashNextMTPHistoryDraftsAsOneRead(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	trunk := writeHermeticFlashNextGGUF(t)
	runner, err := openFixtureRunnerWithResidency(trunk, OpenOptions{Draft: writeHermeticFlashNextMTPGGUF(t, trunk, false)}, recipe.ResidencyDeviceF32)
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	greedy, err := sampling.New(sampling.Config{Temperature: 0})
	if err != nil {
		t.Fatal(err)
	}
	prompt := []tokenizer.TokenID{4, 9, 17}
	ids, _, err := runner.Generate(t.Context(), "", GenerateOptions{
		MaxNewTokens: 16, Sampler: greedy, DeviceGreedy: true, PromptTokenIDs: prompt, ContinueAfterEOG: true, PlainDecode: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	runner.mu.Lock()
	defer runner.mu.Unlock()
	ctx := t.Context()
	span, err := runner.forwardDeviceCachedModeLocked(ctx, deviceOutputGreedySpan, ids, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer span.Release(context.Background())
	first, cells := len(prompt)-1, len(ids)-1
	whole, wholeKeys, err := runner.draftOverCells(ctx, span.SpanHidden, ids, first, cells)
	if err != nil {
		t.Fatal(err)
	}
	if err := runner.ensureHyperDraftHistory(ctx, uint32(cells)); err != nil {
		t.Fatal(err)
	}
	width := runner.hyperDraftWidth()
	// round drafts cells [from, to) through the history and holds its draft to the one read's.
	round := func(from, to int) {
		t.Helper()
		rows := reference.Value{Shape: tensor.MustShape(width, uint64(to-from)), Data: span.SpanHidden.Data[uint64(from)*width : uint64(to)*width]}
		drafted, err := runner.draftHyperFromHistoryLocked(ctx, rows, ids[from+1:to+1], uint32(from))
		if err != nil {
			t.Fatal(err)
		}
		if drafted != whole[to-1] {
			t.Fatalf("cells %d to %d drafted %d through the history, %d read at once", from, to, drafted, whole[to-1])
		}
	}
	at := first
	for _, rows := range []int{1, 2, 3, 1, 2, 3, 1, 1} {
		if at+rows > cells {
			break
		}
		round(at, at+rows)
		at += rows
	}
	if runner.hyperDraft.history.cells != uint32(at-first) {
		t.Fatalf("the history holds %d cells after %d", runner.hyperDraft.history.cells, at-first)
	}
	// Each cell's keys lie where the one read puts them.
	stored, err := runner.cuda.ReadDeviceValues(ctx, []executor.DeviceValue{runner.hyperDraft.history.keys})
	if err != nil {
		t.Fatal(err)
	}
	cellElements := uint64(runner.spec.KeyLength) * uint64(runner.spec.HeadCountKV)
	for index := range uint64(at-first) * cellElements {
		if delta := math.Abs(float64(stored[0][index] - wholeKeys.Data[index])); !(delta <= 1e-3) {
			t.Fatalf("history cell %d holds key element %d off the one read's by %g", index/cellElements, index%cellElements, delta)
		}
	}
	// A round past the history's last cell starts it again, so it drafts as a read from that cell alone.
	alone, _, err := runner.draftOverCells(ctx, span.SpanHidden, ids, at+1, at+2)
	if err != nil {
		t.Fatal(err)
	}
	rows := reference.Value{Shape: tensor.MustShape(width, 1), Data: span.SpanHidden.Data[uint64(at+1)*width : uint64(at+2)*width]}
	drafted, err := runner.draftHyperFromHistoryLocked(ctx, rows, ids[at+2:at+3], uint32(at+1))
	if err != nil || drafted != alone[at+1] || runner.hyperDraft.history.cells != 1 {
		t.Fatalf("a round past the history drafted %d (%v), alone %d, the history holding %d cells", drafted, err, alone[at+1], runner.hyperDraft.history.cells)
	}
}

// draftOverCells runs the draft layer over cells first..cells-1 at once, each attending over itself and every earlier
// cell, and returns each cell's greedy draft.
func (r *Runner) draftOverCells(ctx context.Context, hidden reference.Value, ids []tokenizer.TokenID, first, cells int) ([]tokenizer.TokenID, reference.Value, error) {
	draft, builder := r.hyperDraft, r.newGraphBuilder()
	feeds := map[*tensor.Tensor]reference.Value{}
	var deviceFeeds tensor.InputBindings[driver.DevicePtr]
	var bindErr error
	weight := func(info gguf.TensorInfo) *tensor.Tensor {
		node, pointer, err := r.deviceInput(builder, info)
		bindErr = errors.Join(bindErr, err)
		deviceFeeds.Add(node, pointer)
		return node
	}
	host := func(name string, shape tensor.Shape, fill float32) *tensor.Tensor {
		node := builder.Input(name, dtype.F32, shape)
		elements, err := shape.Elements()
		bindErr = errors.Join(bindErr, err)
		data := make([]float32, elements)
		for index := range data {
			data[index] = fill
		}
		feeds[node] = reference.Value{Shape: shape, Data: data}
		return node
	}
	offset := func(node *tensor.Tensor) *tensor.Tensor {
		return builder.Add(node, host("reference.unit_offset", node.Shape, 1))
	}
	w := draft.weights
	count := cells - first
	rows := make([]uint32, count)
	positions := make([]uint32, count)
	for at := range count {
		rows[at], positions[at] = uint32(ids[first+at+1]), uint32(first+at)
	}
	embedding := builder.GetRows(weight(r.weights.TokenEmbedding), rows)
	width := r.hyperDraftWidth()
	streams := builder.Input("reference.streams", dtype.F32, tensor.MustShape(width, uint64(count)))
	feeds[streams] = reference.Value{Shape: tensor.MustShape(width, uint64(count)), Data: hidden.Data[width*uint64(first) : width*uint64(cells)]}
	current, err := model.BuildHyperDraftInput(builder, r.spec, embedding, streams, offset(weight(w.EmbeddingNorm)), offset(weight(w.HiddenNorm)),
		weight(w.EmbeddingProjection), weight(w.HiddenProjection))
	if err = errors.Join(err, bindErr); err != nil {
		return nil, reference.Value{}, err
	}
	layer, layerFeeds, err := modeldevice.BindDeviceLayerGraphInputs(builder, w.Layer, r.deviceInput)
	if err != nil {
		return nil, reference.Value{}, err
	}
	deviceFeeds = append(deviceFeeds, layerFeeds...)
	layer.AttentionQNorm, layer.AttentionKNorm = offset(layer.AttentionQNorm), offset(layer.AttentionKNorm)
	layer.HyperAttentionNorm, layer.HyperFeedForwardNorm = offset(layer.HyperAttentionNorm), offset(layer.HyperFeedForwardNorm)
	plan := draft.program.Layer()
	block, err := draft.program.Build(model.CachedBlockContext{
		Builder: builder, Input: current, Positions: positions, Layer: plan.Layer, Recurrent: plan.Recurrent,
	}, layer)
	if err != nil {
		return nil, reference.Value{}, err
	}
	mixed, err := model.BuildHyperDraftHeadInput(builder, r.spec, block.Output, offset(weight(w.MixerNorm)), weight(w.MixerDown), weight(w.MixerUp))
	if err != nil {
		return nil, reference.Value{}, err
	}
	head := r.outputTensor()
	if len(draft.vocabulary) != 0 {
		head = draft.head
	}
	selection := builder.TopK(builder.MulMat(weight(head), mixed), 1)
	if err := errors.Join(builder.Err(), bindErr); err != nil {
		return nil, reference.Value{}, err
	}
	graph, err := executor.CompileIndexed(selection, block.Key)
	if err != nil {
		return nil, reference.Value{}, err
	}
	if err := graph.Graph.ReadBack(selection, block.Key); err != nil {
		return nil, reference.Value{}, err
	}
	if err := graph.Inputs.Bind(deviceFeeds); err != nil {
		return nil, reference.Value{}, err
	}
	retained, err := r.cuda.ExecuteRetainedCompiled(ctx, graph.Graph, feeds, graph.Inputs, nil, nil)
	if err != nil {
		return nil, reference.Value{}, err
	}
	values, err := retained.CopyToHost(ctx, selection)
	keys, keysErr := retained.CopyToHost(ctx, block.Key)
	err = errors.Join(err, keysErr)
	if err = errors.Join(err, retained.Release(ctx)); err != nil {
		return nil, reference.Value{}, err
	}
	drafts := make([]tokenizer.TokenID, cells)
	for at := range count {
		index := int(values.Data[at])
		if len(draft.vocabulary) != 0 {
			drafts[first+at] = tokenizer.TokenID(draft.vocabulary[index])
		} else {
			drafts[first+at] = tokenizer.TokenID(index)
		}
	}
	return drafts, keys, nil
}
