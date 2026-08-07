package inference

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"

	"llamacpp2go/internal/cuda/driver"
	"llamacpp2go/internal/cuda/executor"
	"llamacpp2go/internal/model"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/dtype"
	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
)

type qwen35DeviceCohortKey struct {
	tokens, position, pageTokens uint32
}

const f32DeviceStorageBytes = uint64(4)

func (r *Runner) forwardPackedQwen35CohortsLocked(
	ctx context.Context,
	appends []deviceBatchAppend,
) ([]*deviceKVCache, bool, error) {
	if len(appends) < 2 || r.profile().Attention != model.AttentionQwenGDN {
		return nil, false, nil
	}
	packed, fallback := planQwen35DeviceCohorts(appends)
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
			caches, err = r.forwardPackedQwen35DeviceBatchLocked(ctx, items)
		} else {
			caches, err = r.forwardDeviceCachedBranchedBatchLocked(ctx, items)
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
	if len(fallback) > 0 {
		if err := execute(fallback, false); err != nil {
			return fail(err)
		}
	}
	return next, true, nil
}

func planQwen35DeviceCohorts(appends []deviceBatchAppend) ([][]int, []int) {
	cohorts := make([][]int, 0)
	byKey := make(map[qwen35DeviceCohortKey]int)
	fallback := make([]int, 0)
	for index, item := range appends {
		if item.Past == nil || len(item.Tokens) != 1 {
			fallback = append(fallback, index)
			continue
		}
		key := qwen35DeviceCohortKey{
			tokens: item.Past.Tokens, position: item.Past.Position, pageTokens: item.PageTokens,
		}
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
			fallback = append(fallback, cohort...)
			continue
		}
		packed = append(packed, cohort)
	}
	if len(packed) == 0 {
		return nil, fallback
	}
	slices.Sort(fallback)
	return packed, fallback
}

func (r *Runner) forwardPackedQwen35DeviceBatchLocked(
	ctx context.Context,
	appends []deviceBatchAppend,
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
		builder, 0, tokens, packedPast, uint64(len(appends)), hostFeeds, deviceFeeds,
	)
	if err != nil {
		return nil, fmt.Errorf("inference: packed device batch: %w", err)
	}
	outputs := []*tensor.Tensor{graph.logits}
	for layer := range graph.keys {
		outputs = append(outputs, graph.keys[layer], graph.values[layer])
		outputs = graph.states[layer].AppendValues(outputs)
	}
	if err := builder.Err(); err != nil {
		return nil, err
	}
	retained, err := r.cuda.ExecuteRetainedWithDeviceFeeds(ctx, outputs, hostFeeds, deviceFeeds)
	if err != nil {
		return nil, err
	}
	fail := func(cause error) ([]*deviceKVCache, error) {
		_ = retained.Release(context.Background())
		return nil, cause
	}
	logits, err := retained.CopyToHost(ctx, graph.logits)
	if err != nil {
		return fail(err)
	}
	vocabulary := int(r.spec.VocabularySize)
	if vocabulary <= 0 || len(logits.Data) != vocabulary*len(appends) {
		return fail(errors.New("inference: packed logits shape is incompatible"))
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
			PageTokens: resolveCachePageTokens(item.PageTokens),
			Logits:     slices.Clone(logits.Data[sequence*vocabulary : (sequence+1)*vocabulary]),
		}
		cache.Logits = r.finalizeLogits(cache.Logits)
		for layer := range cache.Keys {
			cache.Keys[layer], err = splitPackedDeviceValue(
				packedKeys[layer], item.Past.Keys[layer].Shape, sequence, len(appends),
			)
			if err != nil {
				return fail(fmt.Errorf("inference: split packed key layer %d: %w", layer, err))
			}
			cache.Values[layer], err = splitPackedDeviceValue(
				packedValues[layer], item.Past.Values[layer].Shape, sequence, len(appends),
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
					value, splitErr := splitPackedDeviceValue(
						state.Value, template.Value.Shape, sequence, len(appends),
					)
					if splitErr != nil {
						return fail(fmt.Errorf("inference: split packed state %q layer %d: %w", name, layer, splitErr))
					}
					cache.States[layer][name] = deviceLayerState{Mode: state.Mode, Value: value}
				}
			}
		}
		if err = rebuildDeviceCachePages(cache, cache.PageTokens); err != nil {
			return fail(err)
		}
		next[sequence] = cache
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
		copySpec, copyErr := packedDeviceCopy(values)
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
	owner, values, err := r.cuda.CopyDeviceValues(ctx, copies)
	if err != nil {
		return nil, nil, err
	}
	packed := &deviceKVCache{
		Keys: make([]executor.DeviceValue, len(first.Keys)), Values: make([]executor.DeviceValue, len(first.Values)),
		States: make([]deviceLayerStates, len(first.States)), Tokens: first.Tokens,
		Position: first.Position, PageTokens: first.PageTokens,
	}
	valueIndex := 0
	for layer := range packed.Keys {
		packed.Keys[layer], packed.Values[layer] = values[valueIndex], values[valueIndex+1]
		valueIndex += 2
		if len(stateNames[layer]) != 0 {
			packed.States[layer] = make(deviceLayerStates, len(stateNames[layer]))
			for _, name := range stateNames[layer] {
				mode := first.States[layer][name].Mode
				packed.States[layer][name] = deviceLayerState{Mode: mode, Value: values[valueIndex]}
				valueIndex++
			}
		}
	}
	return packed, owner, nil
}

func packedDeviceCopy(values []executor.DeviceValue) (executor.DeviceCopy, error) {
	if len(values) < 2 {
		return executor.DeviceCopy{}, errors.New("packed device copy requires multiple values")
	}
	base := values[0].Shape
	shape, err := packedDeviceShape(base, len(values))
	if err != nil {
		return executor.DeviceCopy{}, err
	}
	bytes, err := base.Bytes(dtype.F32)
	if err != nil {
		return executor.DeviceCopy{}, err
	}
	segments := make([]executor.DeviceCopySegment, len(values))
	for index, value := range values {
		if !value.Shape.Equal(base) || value.Pointer == 0 {
			return executor.DeviceCopy{}, errors.New("packed device values have incompatible shapes or storage")
		}
		segments[index] = executor.DeviceCopySegment{Source: value.Pointer, Bytes: bytes}
	}
	return executor.DeviceCopy{Shape: shape, Segments: segments}, nil
}

func packedDeviceBatchView(appends []deviceBatchAppend) (*deviceKVCache, bool) {
	first := appends[0].Past
	packed := &deviceKVCache{
		Keys: make([]executor.DeviceValue, len(first.Keys)), Values: make([]executor.DeviceValue, len(first.Values)),
		States: make([]deviceLayerStates, len(first.States)), Tokens: first.Tokens,
		Position: first.Position, PageTokens: first.PageTokens,
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
		return packedDeviceView(values)
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
	return packed, true
}

func packedDeviceView(values []executor.DeviceValue) (executor.DeviceValue, bool) {
	if len(values) < 2 {
		return executor.DeviceValue{}, false
	}
	shape, err := packedDeviceShape(values[0].Shape, len(values))
	if err != nil {
		return executor.DeviceValue{}, false
	}
	bytes, err := values[0].Shape.Bytes(dtype.F32)
	last := uint64(len(values) - 1)
	if err != nil || bytes == 0 || last > math.MaxUint64/bytes ||
		uint64(values[0].Pointer) > math.MaxUint64-last*bytes {
		return executor.DeviceValue{}, false
	}
	for index, value := range values {
		want := values[0].Pointer + driver.DevicePtr(uint64(index)*bytes)
		if !value.Shape.Equal(values[0].Shape) || value.Pointer != want {
			return executor.DeviceValue{}, false
		}
	}
	return executor.DeviceValue{Pointer: values[0].Pointer, Shape: shape}, true
}

func packedDeviceShape(base tensor.Shape, sequences int) (tensor.Shape, error) {
	if sequences < 2 {
		return tensor.Shape{}, errors.New("packed device shape requires multiple sequences")
	}
	dimensions := base.Slice()
	if base.Rank == tensor.MaxDimensions {
		if dimensions[base.Rank-1] != 1 {
			return tensor.Shape{}, errors.New("packed rank-4 value has no singleton sequence dimension")
		}
		dimensions[base.Rank-1] = uint64(sequences)
	} else {
		dimensions = append(dimensions, uint64(sequences))
	}
	return tensor.NewShape(dimensions...)
}

func splitPackedDeviceValue(
	packed executor.DeviceValue,
	template tensor.Shape,
	sequence, sequences int,
) (executor.DeviceValue, error) {
	if sequence < 0 || sequence >= sequences || sequences < 2 {
		return executor.DeviceValue{}, errors.New("packed device split index is invalid")
	}
	elements, err := packed.Shape.Elements()
	if err != nil || elements%uint64(sequences) != 0 {
		return executor.DeviceValue{}, errors.New("packed device split size is invalid")
	}
	strideElements := elements / uint64(sequences)
	if strideElements > math.MaxUint64/f32DeviceStorageBytes ||
		uint64(sequence) > math.MaxUint64/(strideElements*f32DeviceStorageBytes) {
		return executor.DeviceValue{}, errors.New("packed device split offset overflows")
	}
	offset := uint64(sequence) * strideElements * f32DeviceStorageBytes
	if uint64(packed.Pointer) > math.MaxUint64-offset {
		return executor.DeviceValue{}, errors.New("packed device split pointer overflows")
	}
	var shape tensor.Shape
	switch {
	case packed.Shape.Rank == template.Rank+1:
		shape, err = tensor.NewShape(packed.Shape.Slice()[:template.Rank]...)
	case packed.Shape.Rank == template.Rank && template.Dims[template.Rank-1] == 1:
		dimensions := packed.Shape.Slice()
		dimensions[packed.Shape.Rank-1] = 1
		shape, err = tensor.NewShape(dimensions...)
	default:
		return executor.DeviceValue{}, errors.New("packed device split shape differs from template")
	}
	if err != nil {
		return executor.DeviceValue{}, err
	}
	return executor.DeviceValue{Pointer: packed.Pointer + driver.DevicePtr(offset), Shape: shape}, nil
}
