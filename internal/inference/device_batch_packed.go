package inference

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

type deviceCohortKey struct {
	tokens, position uint32
}

func (r *Runner) forwardPackedDeviceCohortsLocked(
	ctx context.Context,
	appends []deviceBatchAppend,
	plan deviceOutputPlan,
) ([]*deviceKVCache, bool, error) {
	if len(appends) < 2 || !r.forwardProgram().DeviceBatchSelection() {
		return nil, false, nil
	}
	for _, item := range appends {
		if item.Past != nil && item.Past.storage != nil {
			return nil, false, nil
		}
	}
	packed, remainder := planDeviceCohorts(appends)
	if len(packed) == 0 {
		return nil, false, nil
	}
	next := make([]*deviceKVCache, len(appends))
	fail := func(cause error) ([]*deviceKVCache, bool, error) {
		for _, cache := range next {
			if cache != nil {
				_ = cache.Release(context.Background())
			}
		}
		return nil, true, cause
	}
	execute := func(indices []int, packed bool) error {
		items := make([]deviceBatchAppend, len(indices))
		for item, index := range indices {
			items[item] = appends[index]
		}
		var caches []*deviceKVCache
		var err error
		if packed {
			caches, err = r.forwardPackedDeviceBatchLocked(ctx, items, plan)
		} else {
			caches, err = r.forwardDeviceCachedBranchedBatchLocked(ctx, items, plan)
		}
		if err != nil {
			return err
		}
		if len(caches) != len(indices) {
			for _, cache := range caches {
				if cache != nil {
					_ = cache.Release(context.Background())
				}
			}
			return errors.New("inference: device cohort result count differs")
		}
		for item, index := range indices {
			next[index] = caches[item]
		}
		return nil
	}
	for _, cohort := range packed {
		if err := execute(cohort, true); err != nil {
			return fail(err)
		}
	}
	if len(remainder) > 0 {
		if err := execute(remainder, false); err != nil {
			return fail(err)
		}
	}
	return next, true, nil
}

func planDeviceCohorts(appends []deviceBatchAppend) ([][]int, []int) {
	cohorts := make([][]int, 0)
	byKey := make(map[deviceCohortKey]int)
	remainder := make([]int, 0)
	for index, item := range appends {
		if item.Past == nil || len(item.Tokens) != 1 {
			remainder = append(remainder, index)
			continue
		}
		key := deviceCohortKey{tokens: item.Past.Tokens, position: item.Past.Position}
		cohort, ok := byKey[key]
		if !ok {
			cohort = len(cohorts)
			byKey[key] = cohort
			cohorts = append(cohorts, nil)
		}
		cohorts[cohort] = append(cohorts[cohort], index)
	}
	packed := cohorts[:0]
	for _, cohort := range cohorts {
		if len(cohort) < 2 {
			remainder = append(remainder, cohort...)
			continue
		}
		packed = append(packed, cohort)
	}
	if len(packed) == 0 {
		return nil, remainder
	}
	slices.Sort(remainder)
	return packed, remainder
}

func (r *Runner) forwardPackedDeviceBatchLocked(
	ctx context.Context,
	appends []deviceBatchAppend,
	plan deviceOutputPlan,
) ([]*deviceKVCache, error) {
	packedPast, packedOwner, err := r.packDeviceBatchCaches(ctx, appends)
	if err != nil {
		return nil, err
	}
	if packedOwner != nil {
		defer packedOwner.Release(context.Background())
	}
	tokens := make([]tokenizer.TokenID, len(appends))
	for index, item := range appends {
		tokens[index] = item.Tokens[0]
	}
	builder := r.newGraphBuilder()
	hostFeeds := make(map[*tensor.Tensor]reference.Value)
	deviceFeeds := make(map[*tensor.Tensor]driver.DevicePtr)
	graph, err := r.buildDeviceCachedBatchBranch(
		builder, 0, tokens, packedPast, uint64(len(appends)), plan,
		tensor.CacheWriteConcat, hostFeeds, deviceFeeds,
	)
	if err != nil {
		return nil, fmt.Errorf("inference: packed device batch: %w", err)
	}
	outputs := decodeGraphOutputs(graph, plan)
	if err := builder.Err(); err != nil {
		return nil, err
	}
	compiled, err := executor.Compile(outputs...)
	if err != nil {
		return nil, err
	}
	inputs := compiled.NewDeviceInputs()
	for node, pointer := range deviceFeeds {
		if err := inputs.Set(node, pointer); err != nil {
			return nil, err
		}
	}
	retained, err := r.cuda.ExecuteRetainedCompiled(ctx, compiled, hostFeeds, inputs, nil, nil)
	if err != nil {
		return nil, err
	}
	fail := func(cause error) ([]*deviceKVCache, error) {
		_ = retained.Release(context.Background())
		return nil, cause
	}
	packedKeys := make([]executor.DeviceValue, len(graph.keys))
	packedValues := make([]executor.DeviceValue, len(graph.values))
	packedStates := make([]deviceLayerStates, len(graph.states))
	for layer := range graph.keys {
		var present bool
		packedKeys[layer], present = retained.Value(graph.keys[layer])
		if !present {
			return fail(fmt.Errorf("inference: packed key is missing for layer %d", layer))
		}
		packedValues[layer], present = retained.Value(graph.values[layer])
		if !present {
			return fail(fmt.Errorf("inference: packed value is missing for layer %d", layer))
		}
		if len(graph.states[layer]) != 0 {
			packedStates[layer] = make(deviceLayerStates, len(graph.states[layer]))
			for name, state := range graph.states[layer] {
				value, ok := retained.Value(state.Value)
				if !ok {
					return fail(fmt.Errorf("inference: packed state %q is missing for layer %d", name, layer))
				}
				packedStates[layer][name] = deviceLayerState{Mode: state.Mode, Value: value}
			}
		}
	}
	next := make([]*deviceKVCache, len(appends))
	for sequence, item := range appends {
		cache := &deviceKVCache{
			Keys: make([]executor.DeviceValue, len(graph.keys)), Values: make([]executor.DeviceValue, len(graph.values)),
			States: make([]deviceLayerStates, len(graph.states)),
			Tokens: graph.pastTokens + graph.tokenCount, Position: graph.nextPosition + graph.tokenCount,
		}
		for layer := range cache.Keys {
			cache.Keys[layer], err = executor.SplitPackedValue(
				packedKeys[layer], item.Past.Keys[layer].Shape, dtype.F32, uint64(sequence),
			)
			if err != nil {
				return fail(fmt.Errorf("inference: split packed key layer %d: %w", layer, err))
			}
			cache.Values[layer], err = executor.SplitPackedValue(
				packedValues[layer], item.Past.Values[layer].Shape, dtype.F32, uint64(sequence),
			)
			if err != nil {
				return fail(fmt.Errorf("inference: split packed value layer %d: %w", layer, err))
			}
			if len(packedStates[layer]) != 0 {
				cache.States[layer] = make(deviceLayerStates, len(packedStates[layer]))
				for name, state := range packedStates[layer] {
					template, ok := item.Past.States[layer][name]
					if !ok {
						return fail(fmt.Errorf("inference: packed state %q has no template at layer %d", name, layer))
					}
					value, splitErr := executor.SplitPackedValue(
						state.Value, template.Value.Shape, dtype.F32, uint64(sequence),
					)
					if splitErr != nil {
						return fail(fmt.Errorf("inference: split packed state %q layer %d: %w", name, layer, splitErr))
					}
					cache.States[layer][name] = deviceLayerState{Mode: state.Mode, Value: value}
				}
			}
		}
		if err = rebuildDeviceCachePages(cache, r.program.Decode.Session); err != nil {
			return fail(err)
		}
		next[sequence] = cache
	}
	if err = plan.collect(ctx, r, retained, graph, next); err != nil {
		return fail(err)
	}
	owner := newDeviceCacheOwner(retained, len(next))
	for _, cache := range next {
		cache.owner = owner
	}
	return next, nil
}

func (r *Runner) packDeviceBatchCaches(
	ctx context.Context,
	appends []deviceBatchAppend,
) (*deviceKVCache, *executor.RetainedOutputs, error) {
	first := appends[0].Past
	if first == nil || len(first.Keys) != len(r.weights.Layers) || len(first.Values) != len(r.weights.Layers) {
		return nil, nil, errors.New("inference: packed device cache layer count differs")
	}
	for _, item := range appends[1:] {
		if item.Past == nil || len(item.Past.Keys) != len(first.Keys) ||
			len(item.Past.Values) != len(first.Values) || len(item.Past.States) != len(first.States) {
			return nil, nil, errors.New("inference: packed device cache layout differs")
		}
	}
	hasSelection := first.Selection.Pointer != 0
	for _, item := range appends[1:] {
		hasSelection = hasSelection && item.Past.Selection.Pointer != 0
	}
	if packed, ok := packedDeviceBatchView(appends); ok {
		return packed, nil, nil
	}
	copies := make([]executor.DeviceCopy, 0, len(first.Keys)*2)
	stateNames := make([][]model.CacheStateName, len(first.Keys))
	collect := func(selectValue func(*deviceKVCache) executor.DeviceValue) error {
		values := make([]executor.DeviceValue, len(appends))
		for index, item := range appends {
			values[index] = selectValue(item.Past)
		}
		copySpec, copyErr := executor.PackedCopy(values, dtype.F32)
		if copyErr != nil {
			return copyErr
		}
		copies = append(copies, copySpec)
		return nil
	}
	for layer := range first.Keys {
		if err := collect(func(cache *deviceKVCache) executor.DeviceValue { return cache.Keys[layer] }); err != nil {
			return nil, nil, fmt.Errorf("inference: pack key layer %d: %w", layer, err)
		}
		if err := collect(func(cache *deviceKVCache) executor.DeviceValue { return cache.Values[layer] }); err != nil {
			return nil, nil, fmt.Errorf("inference: pack value layer %d: %w", layer, err)
		}
		if layer < len(first.States) {
			stateNames[layer] = first.States[layer].SortedNames()
		}
		for _, name := range stateNames[layer] {
			if err := collect(func(cache *deviceKVCache) executor.DeviceValue {
				return cache.States[layer][name].Value
			}); err != nil {
				return nil, nil, fmt.Errorf("inference: pack state %q layer %d: %w", name, layer, err)
			}
		}
	}
	if hasSelection {
		if err := collect(func(cache *deviceKVCache) executor.DeviceValue { return cache.Selection }); err != nil {
			return nil, nil, fmt.Errorf("inference: pack device selection: %w", err)
		}
	}
	owner, values, err := r.cuda.CopyDeviceValues(ctx, copies)
	if err != nil {
		return nil, nil, err
	}
	packed := &deviceKVCache{
		Keys: make([]executor.DeviceValue, len(first.Keys)), Values: make([]executor.DeviceValue, len(first.Values)),
		States: make([]deviceLayerStates, len(first.States)), Tokens: first.Tokens,
		Position: first.Position,
	}
	valueIndex := 0
	for layer := range packed.Keys {
		packed.Keys[layer], packed.Values[layer] = values[valueIndex], values[valueIndex+1]
		valueIndex += tensor.PairedExtent
		if len(stateNames[layer]) != 0 {
			packed.States[layer] = make(deviceLayerStates, len(stateNames[layer]))
			for _, name := range stateNames[layer] {
				mode := first.States[layer][name].Mode
				packed.States[layer][name] = deviceLayerState{Mode: mode, Value: values[valueIndex]}
				valueIndex++
			}
		}
	}
	if hasSelection {
		packed.Selection = values[valueIndex]
		packed.Selection.Shape = tensor.MustShape(uint64(len(appends)))
	}
	return packed, owner, nil
}

func packedDeviceBatchView(appends []deviceBatchAppend) (*deviceKVCache, bool) {
	first := appends[0].Past
	packed := &deviceKVCache{
		Keys: make([]executor.DeviceValue, len(first.Keys)), Values: make([]executor.DeviceValue, len(first.Values)),
		States: make([]deviceLayerStates, len(first.States)), Tokens: first.Tokens,
		Position: first.Position,
	}
	view := func(selectValue func(*deviceKVCache) (executor.DeviceValue, bool)) (executor.DeviceValue, bool) {
		values := make([]executor.DeviceValue, len(appends))
		for index, item := range appends {
			var ok bool
			values[index], ok = selectValue(item.Past)
			if !ok {
				return executor.DeviceValue{}, false
			}
		}
		return executor.PackedView(values, dtype.F32)
	}
	for layer := range packed.Keys {
		var ok bool
		packed.Keys[layer], ok = view(func(cache *deviceKVCache) (executor.DeviceValue, bool) {
			if layer >= len(cache.Keys) {
				return executor.DeviceValue{}, false
			}
			return cache.Keys[layer], true
		})
		if !ok {
			return nil, false
		}
		packed.Values[layer], ok = view(func(cache *deviceKVCache) (executor.DeviceValue, bool) {
			if layer >= len(cache.Values) {
				return executor.DeviceValue{}, false
			}
			return cache.Values[layer], true
		})
		if !ok {
			return nil, false
		}
		if layer >= len(first.States) {
			continue
		}
		for _, name := range first.States[layer].SortedNames() {
			state, stateOK := view(func(cache *deviceKVCache) (executor.DeviceValue, bool) {
				if layer >= len(cache.States) {
					return executor.DeviceValue{}, false
				}
				value, present := cache.States[layer][name]
				return value.Value, present
			})
			if !stateOK {
				return nil, false
			}
			if packed.States[layer] == nil {
				packed.States[layer] = make(deviceLayerStates)
			}
			packed.States[layer][name] = deviceLayerState{
				Mode: first.States[layer][name].Mode, Value: state,
			}
		}
	}
	if first.Selection.Pointer != 0 {
		selection, ok := view(func(cache *deviceKVCache) (executor.DeviceValue, bool) {
			return cache.Selection, cache.Selection.Pointer != 0
		})
		if !ok {
			return nil, false
		}
		selection.Shape = tensor.MustShape(uint64(len(appends)))
		packed.Selection = selection
	}
	return packed, true
}
