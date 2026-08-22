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
)

// deviceOutputPlan: graph reduction and publication contract.
type deviceOutputPlan struct {
	mode deviceOutputMode
	topK uint32
}

func compileDeviceOutputPlan(mode deviceOutputMode, topK, vocabulary uint32) (deviceOutputPlan, error) {
	switch mode {
	case deviceOutputLogits, deviceOutputGreedy:
		if topK != 0 {
			return deviceOutputPlan{}, errors.New("inference: decode output count requires top-K mode")
		}
	case deviceOutputTopK:
		if topK == 0 || topK > vocabulary {
			return deviceOutputPlan{}, errors.New("inference: device top-K count is invalid")
		}
	default:
		return deviceOutputPlan{}, errors.New("inference: decode output mode is invalid")
	}
	return deviceOutputPlan{mode: mode, topK: topK}, nil
}

func (p deviceOutputPlan) fullLogits() bool { return p.mode == deviceOutputLogits }

func (p deviceOutputPlan) retainsSelection() bool { return p.mode == deviceOutputGreedy }

func (p deviceOutputPlan) reduce(builder *tensor.Builder, logits *tensor.Tensor) (selection, candidates *tensor.Tensor) {
	switch p.mode {
	case deviceOutputGreedy:
		selection = builder.TopK(logits, 1)
	case deviceOutputTopK:
		candidates = builder.TopKPairs(logits, p.topK)
	}
	return selection, candidates
}

func (p deviceOutputPlan) graphOutput(graph deviceBatchGraph) *tensor.Tensor {
	switch p.mode {
	case deviceOutputGreedy:
		return graph.selection
	case deviceOutputTopK:
		return graph.candidates
	default:
		return graph.logits
	}
}

func (p deviceOutputPlan) collect(ctx context.Context, r *Runner, retained *executor.RetainedOutputs, graph deviceBatchGraph, caches []*deviceKVCache) error {
	count := len(caches)
	vocabulary := int(r.spec.VocabularySize)
	switch p.mode {
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
	return nil
}
