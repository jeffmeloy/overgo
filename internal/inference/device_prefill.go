package inference

import (
	"context"
	"errors"

	"overgo/internal/cuda/executor"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// prefillDeviceCachedLocked appends prompt chunks and retains final-position outputs.
// Only a memory refusal before execution reduces a chunk and retries it.
// The caller retains ownership of the supplied prefix on every exit.
func (r *Runner) prefillDeviceCachedLocked(ctx context.Context, appendInput deviceBatchAppend, plan deviceOutputPlan, chunk int) (_ *deviceKVCache, err error) {
	if chunk <= 0 || len(appendInput.Tokens) == 0 || plan.is(deviceOutputGreedySpan) {
		return nil, errors.New("inference: bounded prefill requires tokens, a positive chunk and final-position output")
	}
	past := appendInput.Past
	defer func() {
		if err != nil && past != appendInput.Past {
			err = errors.Join(err, past.Release(context.Background()))
		}
	}()
	for remaining := appendInput.Tokens; len(remaining) > 0; {
		chunk = min(chunk, len(remaining))
		source := past
		if past != nil && past.Selection.Pointer != 0 {
			// Prompt tokens override decode feedback, including a one-token tail.
			prefix := *past
			prefix.Selection = executor.DeviceValue{}
			source = &prefix
		}
		next, runErr := r.forwardDeviceCachedBranchedBatchLocked(ctx, []deviceBatchAppend{{Tokens: remaining[:chunk], Past: source}}, plan)
		var refused *executor.MemoryCapacityError
		if errors.As(runErr, &refused) && chunk > 1 {
			// Bisection reduces the candidate without assuming linear memory growth.
			chunk = (chunk + 1) / tensor.PairedExtent
			continue
		}
		if runErr != nil {
			return nil, runErr
		}
		previous := past
		past = next[0]
		if previous != appendInput.Past {
			if err := previous.Release(context.Background()); err != nil {
				return nil, err
			}
		}
		remaining = remaining[chunk:]
	}
	return past, nil
}

func reusableDeviceCacheStorage(past *deviceKVCache, plan deviceCacheTargetPlan) *deviceCacheStorage {
	if past != nil && past.storage.aliases(past) && past.storage.exclusive() && deviceCacheStorageFits(past.storage, plan) {
		return past.storage
	}
	return nil
}

func (r *Runner) checkDevicePrefillMemory(ctx context.Context, execution *executor.IndexedGraph, plan deviceCacheTargetPlan, past *deviceKVCache) error {
	var slots []executor.OutputSlot
	add := func(value deviceCacheTargetValue) {
		if value.enabled {
			slots = append(slots, value.slot)
		}
	}
	for _, layer := range plan.layers {
		add(layer.key)
		add(layer.value)
		for _, state := range layer.states {
			add(state)
		}
	}
	var capacities []uint64
	if plan.elements > 0 && reusableDeviceCacheStorage(past, plan) == nil {
		shape, err := tensor.NewShape(plan.elements)
		if err != nil {
			return err
		}
		bytes, err := shape.Bytes(dtype.F32)
		if err != nil {
			return err
		}
		capacities = []uint64{bytes}
	}
	return r.cuda.CheckRetainedMemory(ctx, execution.Graph, execution.Inputs, slots, capacities)
}
