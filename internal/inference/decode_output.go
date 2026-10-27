package inference

import (
	"context"
	"errors"
	"slices"

	"overgo/internal/cuda/executor"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

type deviceOutputMode uint8

const (
	deviceOutputLogits deviceOutputMode = iota
	deviceOutputGreedy
	deviceOutputTopK
	// deviceOutputGreedySpan keeps every position of a span append in
	// the output head and selects greedily per position — the MTP
	// verify batch: one weight read scores the whole draft span.
	deviceOutputGreedySpan
	// deviceOutputLogitsSpan keeps every position of a span append in the output head as the greedy span does and
	// publishes each row's logits, which a sampled request's verify draws its rows from.
	deviceOutputLogitsSpan
	// deviceOutputTopKSpan keeps every position of a span append in the output head as the greedy span does and
	// publishes each row's top-K candidates, which a sampled request's verify draws its rows from where its sampler cuts
	// to the top K first (sampling.Sampler.BoundedTopK), as continuous batching's bounded steps do.
	deviceOutputTopKSpan
)

// deviceOutputPlan: graph reduction and publication contract.
//   - streams keeps every row's draft streams on the card too, as a span does: a prompt read whose request drafts over it
type deviceOutputPlan struct {
	mode    deviceOutputMode
	topK    uint32
	streams bool
	// promptExperts marks a prompt read's routed nodes (tensor.Builder.SetPromptExperts): the second part of a
	// two-part read, whose few tokens follow the part before it.
	promptExperts bool
}

func compileDeviceOutputPlan(mode deviceOutputMode, topK, vocabulary uint32) (deviceOutputPlan, error) {
	switch mode {
	case deviceOutputLogits, deviceOutputGreedy, deviceOutputGreedySpan, deviceOutputLogitsSpan:
		if topK != 0 {
			return deviceOutputPlan{}, errors.New("inference: decode output count requires top-K mode")
		}
	case deviceOutputTopK, deviceOutputTopKSpan:
		if topK == 0 || topK > vocabulary {
			return deviceOutputPlan{}, errors.New("inference: device top-K count is invalid")
		}
	default:
		return deviceOutputPlan{}, errors.New("inference: decode output mode is invalid")
	}
	return deviceOutputPlan{mode: mode, topK: topK}, nil
}

// decodeStepPlan is a request's decode step's output plan: its greedy selection, the top K a sampled request draws from
// (sampledTopK), or the whole row.
func (r *Runner) decodeStepPlan(greedy bool, topK uint32) (deviceOutputPlan, error) {
	switch {
	case greedy:
		return compileDeviceOutputPlan(deviceOutputGreedy, 0, r.spec.VocabularySize)
	case topK != 0:
		return compileDeviceOutputPlan(deviceOutputTopK, topK, r.spec.VocabularySize)
	}
	return compileDeviceOutputPlan(deviceOutputLogits, 0, r.spec.VocabularySize)
}

// is names the plan's mode contracts at call sites: deviceOutputLogits
// publishes full logits, deviceOutputGreedy retains the device-resident
// selection for feedback, and deviceOutputGreedySpan keeps every span
// position in the output head with its own greedy selection — the MTP
// verify batch.
func (p deviceOutputPlan) is(mode deviceOutputMode) bool { return p.mode == mode }

// span reports whether the plan keeps every position of the append in the output head: a verify window's.
func (p deviceOutputPlan) span() bool {
	return p.mode == deviceOutputGreedySpan || p.mode == deviceOutputLogitsSpan || p.mode == deviceOutputTopKSpan
}

func (p deviceOutputPlan) reduce(builder *tensor.Builder, logits *tensor.Tensor) (selection, candidates *tensor.Tensor) {
	switch p.mode {
	case deviceOutputGreedy, deviceOutputGreedySpan:
		selection = builder.TopK(logits, 1)
	case deviceOutputTopK, deviceOutputTopKSpan:
		candidates = builder.TopKPairs(logits, p.topK)
	}
	return selection, candidates
}

func (p deviceOutputPlan) graphOutput(graph deviceBatchGraph) *tensor.Tensor {
	switch p.mode {
	case deviceOutputGreedy, deviceOutputGreedySpan:
		return graph.selection
	case deviceOutputTopK, deviceOutputTopKSpan:
		return graph.candidates
	default:
		return graph.logits
	}
}

// readBack names to a decode run's compiled graph the outputs collect reads on the host, which then come down with
// the run: each branch's selection, candidates or logits, and a span's hidden rows.
func (p deviceOutputPlan) readBack(compiled *executor.CompiledGraph, graphs ...deviceBatchGraph) error {
	var outputs []*tensor.Tensor
	for _, graph := range graphs {
		outputs = append(outputs, p.graphOutput(graph))
		if p.span() && graph.hidden != nil {
			outputs = append(outputs, graph.hidden)
		}
	}
	return compiled.ReadBack(outputs...)
}

func (p deviceOutputPlan) collect(ctx context.Context, r *Runner, retained *executor.RetainedOutputs, graph deviceBatchGraph, caches []*deviceKVCache) error {
	count := len(caches)
	vocabulary := int(r.spec.VocabularySize)
	switch p.mode {
	case deviceOutputGreedySpan:
		if count != 1 {
			return errors.New("inference: span selection covers exactly one sequence")
		}
		selected, _, err := retainedDeviceGreedySelections(
			ctx, retained, graph.selection, int(graph.tokenCount), vocabulary,
		)
		if err != nil {
			return err
		}
		caches[0].SpanSelected = selected
	case deviceOutputLogitsSpan:
		if count != 1 {
			return errors.New("inference: span logits cover exactly one sequence")
		}
		value, err := retained.CopyToHost(ctx, graph.logits)
		if err != nil {
			return err
		}
		rows := int(graph.tokenCount)
		if vocabulary <= 0 || len(value.Data) != vocabulary*rows {
			return errors.New("inference: span logits shape is incompatible")
		}
		caches[0].SpanLogits = make([][]float32, rows)
		for row := range rows {
			caches[0].SpanLogits[row] = r.finalizeLogits(value.Data[row*vocabulary : (row+1)*vocabulary])
		}
	case deviceOutputTopKSpan:
		if count != 1 {
			return errors.New("inference: span candidates cover exactly one sequence")
		}
		value, err := retained.CopyToHost(ctx, graph.candidates)
		if err != nil {
			return err
		}
		if caches[0].SpanCandidates, err = r.decodeCandidatePairs(value.Data, int(graph.tokenCount), p.topK); err != nil {
			return err
		}
	case deviceOutputGreedy:
		selected, device, err := retainedDeviceGreedySelections(
			ctx, retained, graph.selection, count, vocabulary,
		)
		if err != nil {
			return err
		}
		for index, cache := range caches {
			cache.Selection = device
			if count > 1 {
				cache.Selection, err = device.SliceLastAxis(dtype.F32, uint64(index), 1)
				if err != nil {
					return err
				}
			}
			cache.Selected = selected[index]
		}
	case deviceOutputTopK:
		value, err := retained.CopyToHost(ctx, graph.candidates)
		if err != nil {
			return err
		}
		sets, err := r.decodeCandidatePairs(value.Data, count, p.topK)
		if err != nil {
			return err
		}
		for index, cache := range caches {
			cache.Candidates = sets[index]
		}
	default:
		value, err := retained.CopyToHost(ctx, graph.logits)
		if err != nil {
			return err
		}
		if vocabulary <= 0 || len(value.Data) != vocabulary*count {
			return errors.New("inference: decode logits shape is incompatible")
		}
		for index, cache := range caches {
			logits := value.Data[index*vocabulary : (index+1)*vocabulary]
			if count > 1 {
				logits = slices.Clone(logits)
			}
			cache.Logits = r.finalizeLogits(logits)
		}
	}
	// A span's rows' hidden streams come down beside its outputs, for the next draft.
	if p.span() && graph.hidden != nil {
		hidden, err := retained.CopyToHost(ctx, graph.hidden)
		if err != nil {
			return err
		}
		caches[0].SpanHidden = hidden
	}
	return nil
}
