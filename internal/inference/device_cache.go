package inference

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sync"

	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

type deviceKVCache struct {
	owner      *deviceCacheOwner
	storage    *deviceCacheStorage
	Keys       []executor.DeviceValue
	Values     []executor.DeviceValue
	States     []deviceLayerStates
	Pages      []deviceKVPage
	PageTokens uint32
	Tokens     uint32
	Position   uint32
	Logits     []float32
	Candidates []LogitCandidate
	Selection  executor.DeviceValue
	Selected   tokenizer.TokenID
}

type deviceCacheStorage struct {
	mu        sync.Mutex
	refs      int
	releasing bool
	keys      []*executor.DeviceBuffer
	values    []*executor.DeviceBuffer
}

func (s *deviceCacheStorage) retain() bool {
	if s == nil {
		return false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.refs <= 0 || s.releasing {
		return false
	}
	s.refs++
	return true
}

func (s *deviceCacheStorage) exclusive() bool {
	if s == nil {
		return false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.refs == 1 && !s.releasing
}

func (s *deviceCacheStorage) release(ctx context.Context) error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	if s.refs <= 0 || s.releasing {
		s.mu.Unlock()
		return nil
	}
	if s.refs > 1 {
		s.refs--
		s.mu.Unlock()
		return nil
	}
	if err := ctx.Err(); err != nil {
		s.mu.Unlock()
		return err
	}
	keys, values := s.keys, s.values
	s.refs = 0
	s.releasing = true
	s.keys = nil
	s.values = nil
	s.mu.Unlock()

	buffers := make([]*executor.DeviceBuffer, 0, len(keys)+len(values))
	buffers = append(buffers, keys...)
	buffers = append(buffers, values...)
	err := executor.ReleaseDeviceBuffers(context.Background(), buffers...)

	s.mu.Lock()
	defer s.mu.Unlock()
	s.releasing = false
	if err != nil {
		s.refs = 1
		s.keys = keys
		s.values = values
		return err
	}
	s.refs = 0
	return nil
}

type deviceOutputMode uint8

const (
	deviceOutputLogits deviceOutputMode = iota
	deviceOutputGreedy
	deviceOutputTopK
)

type deviceOutputPlan struct {
	mode deviceOutputMode
	topK uint32
}

type deviceLayerState = model.CacheState[executor.DeviceValue]
type deviceLayerStates = model.CacheStates[executor.DeviceValue]

// deviceCacheOwner: shared fused-execution allocation owner.
type deviceCacheOwner struct {
	mu      sync.Mutex
	outputs *executor.RetainedOutputs
	refs    int
}

func newDeviceCacheOwner(outputs *executor.RetainedOutputs, refs int) *deviceCacheOwner {
	return &deviceCacheOwner{outputs: outputs, refs: refs}
}

func (o *deviceCacheOwner) retain() bool {
	if o == nil {
		return false
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.refs <= 0 || o.outputs == nil {
		return false
	}
	o.refs++
	return true
}

func (o *deviceCacheOwner) release(ctx context.Context) error {
	if o == nil {
		return nil
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.refs <= 0 {
		return nil
	}
	o.refs--
	if o.refs > 0 || o.outputs == nil {
		return nil
	}
	err := o.outputs.Release(ctx)
	if err != nil {
		o.refs = 1
	} else {
		o.outputs = nil
	}
	return err
}

type deviceKVPage struct {
	Start  uint32
	Tokens uint32
	Keys   []executor.DeviceValue
	Values []executor.DeviceValue
}

func (c *deviceKVCache) Release(ctx context.Context) error {
	if c == nil {
		return nil
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := c.owner.release(context.Background()); err != nil {
		return err
	}
	c.owner = nil
	if err := c.storage.release(context.Background()); err != nil {
		return err
	}
	c.storage = nil
	return nil
}

func (r *Runner) shiftDeviceCacheForAppend(
	cache *deviceKVCache,
	incoming int,
) error {
	return r.shiftDeviceCacheForAppendPolicy(cache, incoming, -1)
}

func (r *Runner) shiftDeviceCacheForAppendPolicy(
	cache *deviceKVCache,
	incoming int,
	requestedDiscard int,
) error {
	if cache == nil || incoming <= 0 {
		return nil
	}
	discardCount, needed, err := planContextShift(
		cache.Tokens, incoming, r.spec.ContextLength, 0, requestedDiscard, true,
	)
	if err != nil {
		return err
	}
	if !needed {
		return nil
	}
	discard := uint64(discardCount)
	remaining := uint64(cache.Tokens) - discard
	for layerIndex := range cache.Keys {
		recurrent := r.layerPlan(
			layerIndex, r.weights.Layers[layerIndex].Recurrent,
		).CacheMode == model.CacheStateFixed
		if recurrent {
			// Recurrent primary state: position-independent.
		} else {
			for _, value := range []*executor.DeviceValue{
				&cache.Keys[layerIndex],
				&cache.Values[layerIndex],
			} {
				if err := shiftDeviceTokenState(value, cache.Tokens, discard); err != nil {
					return fmt.Errorf("inference: layer %d device cache: %w", layerIndex, err)
				}
			}
		}
		if layerIndex < len(cache.States) {
			for name, state := range cache.States[layerIndex] {
				if !state.Mode.TokenAligned() {
					continue
				}
				if err := shiftDeviceTokenState(&state.Value, cache.Tokens, discard); err != nil {
					return fmt.Errorf("inference: layer %d state %q: %w", layerIndex, name, err)
				}
				cache.States[layerIndex][name] = state
			}
		}
	}
	cache.Tokens = uint32(remaining)
	return rebuildDeviceCachePages(cache, cache.PageTokens)
}

func shiftDeviceTokenState(
	value *executor.DeviceValue,
	tokens uint32,
	discard uint64,
) error {
	if value.Shape.Rank != 3 || value.Shape.Dims[2] != uint64(tokens) {
		return fmt.Errorf(
			"shape %v does not contain %d tokens",
			value.Shape.Slice(), tokens,
		)
	}
	stride := value.Shape.Dims[0] * value.Shape.Dims[1]
	if stride > math.MaxUint64/4 || discard > math.MaxUint64/(stride*4) {
		return errors.New("shifted device cache offset overflows")
	}
	offset := discard * stride * 4
	if uint64(value.Pointer) > math.MaxUint64-offset {
		return errors.New("shifted device cache pointer overflows")
	}
	value.Pointer += driver.DevicePtr(offset)
	value.Shape.Dims[2] = uint64(tokens) - discard
	return nil
}

func (r *Runner) compactDeviceCacheForAppend(
	ctx context.Context,
	cache *deviceKVCache,
	incoming int,
	keep uint32,
	requestedDiscard int,
	forceCopy bool,
) (*deviceKVCache, error) {
	if cache == nil || incoming <= 0 {
		return cache, nil
	}
	discard, needed, err := planContextShift(
		cache.Tokens, incoming, r.spec.ContextLength, keep, requestedDiscard, true,
	)
	if err != nil {
		return nil, err
	}
	if !needed {
		return cache, nil
	}
	if keep == 0 && !forceCopy {
		if err := r.shiftDeviceCacheForAppendPolicy(
			cache,
			incoming,
			requestedDiscard,
		); err != nil {
			return nil, err
		}
		return cache, nil
	}
	if len(cache.Keys) != len(r.weights.Layers) ||
		len(cache.Values) != len(r.weights.Layers) {
		return nil, errors.New("inference: device cache layer count differs")
	}
	copies := make([]executor.DeviceCopy, 0, len(cache.Keys)*2)
	type stateCopyTarget struct {
		layer int
		name  model.CacheStateName
		mode  CacheStateMode
	}
	stateTargets := make([]stateCopyTarget, 0)
	stateCopies := make([]executor.DeviceCopy, 0)
	for layerIndex := range cache.Keys {
		recurrent := r.layerPlan(
			layerIndex, r.weights.Layers[layerIndex].Recurrent,
		).CacheMode == model.CacheStateFixed
		for _, item := range []struct {
			label string
			value executor.DeviceValue
		}{
			{"key", cache.Keys[layerIndex]},
			{"value", cache.Values[layerIndex]},
		} {
			copySpec, copyErr := deviceCacheRangeCopy(
				item.value,
				cache.Tokens,
				keep,
				discard,
				recurrent,
			)
			if copyErr != nil {
				return nil, fmt.Errorf(
					"inference: layer %d device cache %s: %w",
					layerIndex,
					item.label,
					copyErr,
				)
			}
			copies = append(copies, copySpec)
		}
		if layerIndex < len(cache.States) && len(cache.States[layerIndex]) != 0 {
			for _, stateName := range cache.States[layerIndex].SortedNames() {
				state := cache.States[layerIndex][stateName]
				copySpec, copyErr := deviceCacheRangeCopy(
					state.Value, cache.Tokens, keep, discard,
					state.Mode == CacheStateFixed,
				)
				if copyErr != nil {
					return nil, fmt.Errorf(
						"inference: layer %d device cache state %q: %w",
						layerIndex, stateName, copyErr,
					)
				}
				stateCopies = append(stateCopies, copySpec)
				stateTargets = append(stateTargets, stateCopyTarget{
					layer: layerIndex, name: stateName, mode: state.Mode,
				})
			}
		}
	}
	copies = append(copies, stateCopies...)
	outputs, values, err := r.cuda.CopyDeviceValues(ctx, copies)
	if err != nil {
		return nil, err
	}
	next := &deviceKVCache{
		owner:      newDeviceCacheOwner(outputs, 1),
		Keys:       make([]executor.DeviceValue, len(cache.Keys)),
		Values:     make([]executor.DeviceValue, len(cache.Values)),
		States:     make([]deviceLayerStates, len(cache.States)),
		Tokens:     cache.Tokens - discard,
		Position:   cache.Position,
		PageTokens: cache.PageTokens,
	}
	for index := range next.Keys {
		next.Keys[index] = values[2*index]
		next.Values[index] = values[2*index+1]
	}
	stateOffset := 2 * len(next.Keys)
	for index, target := range stateTargets {
		if next.States[target.layer] == nil {
			next.States[target.layer] = make(deviceLayerStates)
		}
		next.States[target.layer][target.name] = deviceLayerState{
			Mode: target.mode, Value: values[stateOffset+index],
		}
	}
	if err := rebuildDeviceCachePages(next, next.PageTokens); err != nil {
		_ = next.Release(context.Background())
		return nil, err
	}
	return next, nil
}

func deviceCacheRangeCopy(
	value executor.DeviceValue,
	tokens, keep, discard uint32,
	recurrent bool,
) (executor.DeviceCopy, error) {
	if recurrent {
		elements, err := value.Shape.Elements()
		if err != nil {
			return executor.DeviceCopy{}, err
		}
		if elements > math.MaxUint64/4 {
			return executor.DeviceCopy{}, errors.New(
				"recurrent device cache byte size overflows",
			)
		}
		return executor.DeviceCopy{
			Shape: value.Shape,
			Segments: []executor.DeviceCopySegment{{
				Source: value.Pointer,
				Bytes:  elements * 4,
			}},
		}, nil
	}
	if value.Shape.Rank != 3 ||
		value.Shape.Dims[2] != uint64(tokens) {
		return executor.DeviceCopy{}, fmt.Errorf(
			"shape %v does not contain %d tokens",
			value.Shape.Slice(),
			tokens,
		)
	}
	stride := value.Shape.Dims[0] * value.Shape.Dims[1]
	if stride > math.MaxUint64/4 {
		return executor.DeviceCopy{}, errors.New(
			"attention device cache stride overflows",
		)
	}
	strideBytes := stride * 4
	if strideBytes == 0 {
		return executor.DeviceCopy{}, errors.New(
			"attention device cache stride is zero",
		)
	}
	suffixToken := uint64(keep) + uint64(discard)
	if suffixToken > uint64(tokens) ||
		suffixToken > math.MaxUint64/strideBytes {
		return executor.DeviceCopy{}, errors.New(
			"attention device cache range overflows",
		)
	}
	suffixSource := uint64(value.Pointer) + suffixToken*strideBytes
	if suffixSource < uint64(value.Pointer) {
		return executor.DeviceCopy{}, errors.New(
			"attention device cache source pointer overflows",
		)
	}
	shape := value.Shape
	shape.Dims[2] = uint64(tokens - discard)
	segments := make([]executor.DeviceCopySegment, 0, 2)
	if keep > 0 {
		segments = append(segments, executor.DeviceCopySegment{
			Source: value.Pointer,
			Bytes:  uint64(keep) * strideBytes,
		})
	}
	suffixTokens := tokens - keep - discard
	if suffixTokens > 0 {
		segments = append(segments, executor.DeviceCopySegment{
			Source: driver.DevicePtr(suffixSource),
			Bytes:  uint64(suffixTokens) * strideBytes,
		})
	}
	return executor.DeviceCopy{Shape: shape, Segments: segments}, nil
}

func (r *Runner) forwardDeviceCachedLocked(
	ctx context.Context,
	tokenIDs []tokenizer.TokenID,
	past *deviceKVCache,
) (reference.Value, *deviceKVCache, error) {
	next, err := r.forwardDeviceCachedBatchLocked(ctx, []deviceBatchAppend{{
		Tokens:     tokenIDs,
		Past:       past,
		PageTokens: r.cachePageTokens,
	}})
	if err != nil {
		return reference.Value{}, nil, err
	}
	return reference.Value{}, next[0], nil
}

type deviceBatchAppend struct {
	Tokens     []tokenizer.TokenID
	Past       *deviceKVCache
	PageTokens uint32
}

type deviceBatchGraph struct {
	logits       *tensor.Tensor
	selection    *tensor.Tensor
	candidates   *tensor.Tensor
	keys         []*tensor.Tensor
	values       []*tensor.Tensor
	states       []deviceGraphStates
	pastTokens   uint32
	nextPosition uint32
	tokenCount   uint32
	sequences    uint32
}

type deviceGraphState = model.CacheState[*tensor.Tensor]
type deviceGraphStates = model.CacheStates[*tensor.Tensor]

func retainedDeviceGreedySelections(
	ctx context.Context,
	retained *executor.RetainedOutputs,
	node *tensor.Tensor,
	count, vocabulary int,
) ([]tokenizer.TokenID, executor.DeviceValue, error) {
	if retained == nil || node == nil || count <= 0 || vocabulary <= 0 {
		return nil, executor.DeviceValue{}, errors.New("inference: device greedy selection is invalid")
	}
	device, ok := retained.Value(node)
	if !ok || device.Pointer == 0 {
		return nil, executor.DeviceValue{}, errors.New("inference: retained device greedy selection is missing")
	}
	host, err := retained.CopyToHost(ctx, node)
	if err != nil {
		return nil, executor.DeviceValue{}, err
	}
	if len(host.Data) != count {
		return nil, executor.DeviceValue{}, fmt.Errorf(
			"inference: device greedy selection count %d differs from %d", len(host.Data), count,
		)
	}
	tokens := make([]tokenizer.TokenID, count)
	for index, raw := range host.Data {
		token := int(raw)
		if raw != float32(token) || token < 0 || token >= vocabulary {
			return nil, executor.DeviceValue{}, fmt.Errorf(
				"inference: device greedy selection %d is invalid: %g", index, raw,
			)
		}
		tokens[index] = tokenizer.TokenID(token)
	}
	return tokens, executor.DeviceValue{
		Pointer: device.Pointer,
		Shape:   tensor.MustShape(uint64(count)),
	}, nil
}

func (r *Runner) decodeCandidatePairs(
	data []float32,
	sequences int,
	topK uint32,
) ([][]LogitCandidate, error) {
	const pairValues = 2
	count := int(topK)
	if sequences <= 0 || count <= 0 || len(data) != sequences*count*pairValues {
		return nil, errors.New("inference: device top-K pair shape is invalid")
	}
	vocabulary := int(r.spec.VocabularySize)
	result := make([][]LogitCandidate, sequences)
	for sequence := range sequences {
		items := make([]LogitCandidate, count)
		logits := make([]float32, count)
		base := sequence * count * pairValues
		for index := range count {
			raw := data[base+index*pairValues]
			id := int(raw)
			if raw != float32(id) || id < 0 || id >= vocabulary {
				return nil, fmt.Errorf("inference: device top-K token %d is invalid: %g", index, raw)
			}
			items[index].ID = tokenizer.TokenID(id)
			logits[index] = data[base+index*pairValues+1]
		}
		logits = r.finalizeLogits(logits)
		for index := range items {
			items[index].Logit = logits[index]
		}
		result[sequence] = items
	}
	return result, nil
}

// forwardDeviceCachedBatchLocked: one graph, variable independent branches.
func (r *Runner) forwardDeviceCachedBatchLocked(
	ctx context.Context,
	appends []deviceBatchAppend,
) ([]*deviceKVCache, error) {
	return r.forwardDeviceCachedBatchModeLocked(ctx, appends, deviceOutputPlan{})
}

func (r *Runner) forwardDeviceCachedGreedyBatchLocked(
	ctx context.Context,
	appends []deviceBatchAppend,
) ([]*deviceKVCache, error) {
	if r.profile().Attention != model.AttentionQwenGDN {
		return nil, errors.New("inference: device greedy feedback requires Qwen GDN")
	}
	return r.forwardDeviceCachedBatchModeLocked(ctx, appends, deviceOutputPlan{mode: deviceOutputGreedy})
}

func (r *Runner) forwardDeviceCachedTopKBatchLocked(
	ctx context.Context,
	appends []deviceBatchAppend,
	topK uint32,
) ([]*deviceKVCache, error) {
	if topK == 0 || topK > r.spec.VocabularySize {
		return nil, errors.New("inference: device top-K count is invalid")
	}
	return r.forwardDeviceCachedBatchModeLocked(
		ctx, appends, deviceOutputPlan{mode: deviceOutputTopK, topK: topK},
	)
}

func (r *Runner) forwardDeviceCachedBatchModeLocked(
	ctx context.Context,
	appends []deviceBatchAppend,
	plan deviceOutputPlan,
) ([]*deviceKVCache, error) {
	if len(appends) == 0 {
		return nil, errors.New("inference: device batch is empty")
	}
	if next, handled, err := r.forwardPackedQwen35CohortsLocked(ctx, appends, plan); handled {
		return next, err
	}
	return r.forwardDeviceCachedBranchedBatchLocked(ctx, appends, plan)
}

func (r *Runner) forwardDeviceCachedBranchedBatchLocked(
	ctx context.Context,
	appends []deviceBatchAppend,
	plan deviceOutputPlan,
) ([]*deviceKVCache, error) {
	builder := r.newGraphBuilder()
	hostFeeds := make(map[*tensor.Tensor]reference.Value)
	deviceFeeds := make(map[*tensor.Tensor]driver.DevicePtr)
	graphs := make([]deviceBatchGraph, len(appends))
	outputs := make([]*tensor.Tensor, 0, len(appends)*(1+2*len(r.weights.Layers)))
	for index, appendInput := range appends {
		graph, err := r.buildDeviceCachedBatchBranch(
			builder, index, appendInput.Tokens, appendInput.Past, 1, plan, hostFeeds, deviceFeeds,
		)
		if err != nil {
			return nil, fmt.Errorf("inference: device batch branch %d: %w", index, err)
		}
		graphs[index] = graph
		switch plan.mode {
		case deviceOutputGreedy:
			outputs = append(outputs, graph.selection)
		case deviceOutputTopK:
			outputs = append(outputs, graph.candidates)
		default:
			outputs = append(outputs, graph.logits)
		}
		for layer := range graph.keys {
			outputs = append(outputs, graph.keys[layer], graph.values[layer])
			outputs = graph.states[layer].AppendValues(outputs)
		}
	}
	if err := builder.Err(); err != nil {
		return nil, err
	}
	compiled, err := executor.Compile(outputs...)
	if err != nil {
		return nil, err
	}
	targets, storages, err := r.prepareDeviceCacheTargets(ctx, graphs, appends)
	if err != nil {
		return nil, err
	}
	releaseStorages := func() error {
		var errs []error
		for _, storage := range storages {
			errs = append(errs, storage.release(context.Background()))
		}
		return errors.Join(errs...)
	}
	retained, err := r.cuda.ExecuteRetainedCompiledWithTargets(
		ctx, compiled, hostFeeds, deviceFeeds, targets,
	)
	if err != nil {
		return nil, errors.Join(err, releaseStorages())
	}
	fail := func(cause error) ([]*deviceKVCache, error) {
		return nil, errors.Join(
			cause,
			retained.Release(context.Background()),
			releaseStorages(),
		)
	}
	next := make([]*deviceKVCache, len(graphs))
	for index, graph := range graphs {
		cache := &deviceKVCache{
			storage:    storages[index],
			Keys:       make([]executor.DeviceValue, len(graph.keys)),
			Values:     make([]executor.DeviceValue, len(graph.values)),
			States:     make([]deviceLayerStates, len(graph.states)),
			Tokens:     graph.pastTokens + graph.tokenCount,
			Position:   graph.nextPosition + graph.tokenCount,
			PageTokens: resolveCachePageTokens(appends[index].PageTokens),
		}
		switch plan.mode {
		case deviceOutputGreedy:
			selected, device, selectionErr := retainedDeviceGreedySelections(
				ctx, retained, graph.selection, 1, int(r.spec.VocabularySize),
			)
			if selectionErr != nil {
				return fail(selectionErr)
			}
			cache.Selection, cache.Selected = device, selected[0]
		case deviceOutputTopK:
			value, copyErr := retained.CopyToHost(ctx, graph.candidates)
			if copyErr != nil {
				return fail(copyErr)
			}
			items, candidateErr := r.decodeCandidatePairs(value.Data, 1, plan.topK)
			if candidateErr != nil {
				return fail(candidateErr)
			}
			cache.Candidates = items[0]
		default:
			logits, copyErr := retained.CopyToHost(ctx, graph.logits)
			if copyErr != nil {
				return fail(copyErr)
			}
			cache.Logits = r.finalizeLogits(logits.Data)
		}
		var ok bool
		for layer := range graph.keys {
			cache.Keys[layer], ok = retained.Value(graph.keys[layer])
			if !ok {
				return fail(fmt.Errorf("inference: missing retained key for branch %d layer %d", index, layer))
			}
			cache.Values[layer], ok = retained.Value(graph.values[layer])
			if !ok {
				return fail(fmt.Errorf("inference: missing retained value for branch %d layer %d", index, layer))
			}
			if len(graph.states[layer]) != 0 {
				cache.States[layer] = make(deviceLayerStates, len(graph.states[layer]))
				for name, state := range graph.states[layer] {
					value, present := retained.Value(state.Value)
					if !present {
						return fail(fmt.Errorf(
							"inference: missing retained state %q for branch %d layer %d",
							name, index, layer,
						))
					}
					cache.States[layer][name] = deviceLayerState{Mode: state.Mode, Value: value}
				}
			}
		}
		if pageErr := rebuildDeviceCachePages(cache, cache.PageTokens); pageErr != nil {
			return fail(pageErr)
		}
		next[index] = cache
	}
	owner := newDeviceCacheOwner(retained, len(next))
	for _, cache := range next {
		cache.owner = owner
	}
	return next, nil
}

func (r *Runner) prepareDeviceCacheTargets(
	ctx context.Context,
	graphs []deviceBatchGraph,
	appends []deviceBatchAppend,
) (map[*tensor.Tensor]executor.DeviceValue, []*deviceCacheStorage, error) {
	targets := make(map[*tensor.Tensor]executor.DeviceValue)
	storages := make([]*deviceCacheStorage, len(graphs))
	used := make(map[*deviceCacheStorage]struct{})
	fail := func(cause error) (map[*tensor.Tensor]executor.DeviceValue, []*deviceCacheStorage, error) {
		var errs []error
		errs = append(errs, cause)
		for _, storage := range storages {
			errs = append(errs, storage.release(context.Background()))
		}
		return nil, nil, errors.Join(errs...)
	}
	for branch, graph := range graphs {
		past := appends[branch].Past
		var storage *deviceCacheStorage
		if past != nil && past.storage != nil && past.storage.exclusive() &&
			deviceCacheStorageFits(past.storage, graph) {
			if _, duplicate := used[past.storage]; !duplicate && past.storage.retain() {
				storage = past.storage
				used[storage] = struct{}{}
			}
		}
		if storage == nil {
			storage = &deviceCacheStorage{
				refs:   1,
				keys:   make([]*executor.DeviceBuffer, len(graph.keys)),
				values: make([]*executor.DeviceBuffer, len(graph.values)),
			}
		}
		storages[branch] = storage
		for layer := range graph.keys {
			schema, schemaErr := model.CacheSchema(
				r.spec,
				layer,
				r.weights.Layers[layer],
				graph.pastTokens+graph.tokenCount,
			)
			if schemaErr != nil {
				return fail(schemaErr)
			}
			if !schema.Primary.Key.Mode.TokenAligned() || !schema.Primary.Value.Mode.TokenAligned() ||
				graph.keys[layer].Shape.Rank != 3 || graph.values[layer].Shape.Rank != 3 {
				continue
			}
			allocate := func(
				buffers []*executor.DeviceBuffer,
				node *tensor.Tensor,
			) (executor.DeviceValue, error) {
				buffer := buffers[layer]
				capacityShape := node.Shape
				capacityShape.Dims[2] = uint64(cachePageCapacity(
					graph.pastTokens+graph.tokenCount,
					appends[branch].PageTokens,
					r.spec.ContextLength,
				))
				capacityBytes, sizeErr := capacityShape.Bytes(dtype.F32)
				if sizeErr != nil {
					return executor.DeviceValue{}, sizeErr
				}
				if buffer == nil {
					buffer, sizeErr = r.cuda.AllocateDeviceBuffer(ctx, capacityBytes)
					if sizeErr != nil {
						return executor.DeviceValue{}, sizeErr
					}
					buffers[layer] = buffer
				}
				return buffer.Value(node.Shape)
			}
			key, keyErr := allocate(storage.keys, graph.keys[layer])
			if keyErr != nil {
				return fail(fmt.Errorf("inference: allocate key cache layer %d: %w", layer, keyErr))
			}
			value, valueErr := allocate(storage.values, graph.values[layer])
			if valueErr != nil {
				return fail(fmt.Errorf("inference: allocate value cache layer %d: %w", layer, valueErr))
			}
			targets[graph.keys[layer]] = key
			targets[graph.values[layer]] = value
		}
	}
	return targets, storages, nil
}

func deviceCacheStorageFits(storage *deviceCacheStorage, graph deviceBatchGraph) bool {
	if storage == nil || len(storage.keys) != len(graph.keys) || len(storage.values) != len(graph.values) {
		return false
	}
	for layer := range graph.keys {
		if storage.keys[layer] != nil {
			if _, err := storage.keys[layer].Value(graph.keys[layer].Shape); err != nil {
				return false
			}
		}
		if storage.values[layer] != nil {
			if _, err := storage.values[layer].Value(graph.values[layer].Shape); err != nil {
				return false
			}
		}
	}
	return true
}

func (r *Runner) buildDeviceCachedBatchBranch(
	builder *tensor.Builder,
	branch int,
	tokenIDs []tokenizer.TokenID,
	past *deviceKVCache,
	sequences uint64,
	plan deviceOutputPlan,
	hostFeeds map[*tensor.Tensor]reference.Value,
	deviceFeeds map[*tensor.Tensor]driver.DevicePtr,
) (deviceBatchGraph, error) {
	fail := func(err error) (deviceBatchGraph, error) {
		return deviceBatchGraph{}, err
	}
	if r.spec.NonCausalAttention {
		return fail(errors.New("non-causal models do not support a device KV cache"))
	}
	if len(tokenIDs) == 0 {
		return fail(errors.New("token sequence is empty"))
	}
	var pastTokens, nextPosition uint32
	if past != nil {
		pastTokens = past.Tokens
		nextPosition = past.Position
		if len(past.Keys) != len(r.weights.Layers) || len(past.Values) != len(r.weights.Layers) {
			return fail(errors.New("device cache layer count differs"))
		}
	}
	if sequences == 0 || uint64(len(tokenIDs))%sequences != 0 {
		return fail(errors.New("packed sequence token count is invalid"))
	}
	tokensPerSequence := len(tokenIDs) / int(sequences)
	sequence, err := r.planForwardSequence(
		tokenIDs[:tokensPerSequence], pastTokens, nextPosition,
	)
	if err != nil {
		return fail(err)
	}
	rows, err := r.tokenRows(tokenIDs)
	if err != nil {
		return fail(err)
	}
	positions := sequence.positions
	prefix := fmt.Sprintf("seq.%d.", branch)
	embeddingTable, embeddingPointer, err := r.deviceInput(builder, r.weights.TokenEmbedding)
	if err != nil {
		return fail(err)
	}
	current := builder.GetRows(embeddingTable, rows)
	deviceFeeds[embeddingTable] = embeddingPointer
	dynamicEmbedding := embeddingTable.Type == dtype.F32 || embeddingTable.Type == dtype.Q8_0
	if plan.mode == deviceOutputGreedy && dynamicEmbedding && past != nil && past.Selection.Pointer != 0 && tokensPerSequence == 1 {
		selection := builder.Input(prefix+"selected_token", dtype.F32, tensor.MustShape(sequences))
		deviceFeeds[selection] = past.Selection.Pointer
		current = builder.GatherLast(embeddingTable, selection)
	}
	if r.weights.PositionEmbedding != nil {
		repeatedPositions := make([]uint32, 0, len(positions)*int(sequences))
		for range sequences {
			repeatedPositions = append(repeatedPositions, positions...)
		}
		if positionErr := validateLearnedPositions(repeatedPositions, r.spec.ContextLength); positionErr != nil {
			return fail(positionErr)
		}
		positionTable, pointer, positionErr := r.deviceInput(builder, *r.weights.PositionEmbedding)
		if positionErr != nil {
			return fail(positionErr)
		}
		deviceFeeds[positionTable] = pointer
		current = builder.Add(current, builder.GetRows(positionTable, repeatedPositions))
	}
	if scale := r.spec.InputEmbeddingScale(); scale != 1 {
		current = builder.Scale(current, scale)
	}
	if r.weights.TokenEmbeddingNorm != nil {
		normWeight, pointer, normErr := r.deviceInput(builder, *r.weights.TokenEmbeddingNorm)
		if normErr != nil {
			return fail(normErr)
		}
		deviceFeeds[normWeight] = pointer
		var normBias *tensor.Tensor
		if r.weights.TokenEmbeddingNormBias != nil {
			normBias, pointer, normErr = r.deviceInput(builder, *r.weights.TokenEmbeddingNormBias)
			if normErr != nil {
				return fail(normErr)
			}
			deviceFeeds[normBias] = pointer
		}
		current = model.ApplyNormalization(builder, current, normWeight, normBias, r.spec)
	}
	embeddingSkip := current
	if r.spec.UsesUnweightedRMSNorm() {
		current = builder.RMSNorm(current, r.spec.RMSNormEpsilon)
		embeddingSkip = current
	}
	var perLayerInputs []*tensor.Tensor
	if r.profile().Has(model.ArchitecturePerLayerEmbeddings) && r.spec.EmbeddingPerLayer > 0 {
		if r.weights.PerLayerTokenEmbedding == nil || r.weights.PerLayerModelProjection == nil ||
			r.weights.PerLayerProjectionNorm == nil {
			return fail(errors.New("Gemma 4 per-layer weights are incomplete"))
		}
		perLayerTable, pointer, inputErr := r.deviceInput(builder, *r.weights.PerLayerTokenEmbedding)
		if inputErr != nil {
			return fail(inputErr)
		}
		deviceFeeds[perLayerTable] = pointer
		projection, pointer, inputErr := r.deviceInput(builder, *r.weights.PerLayerModelProjection)
		if inputErr != nil {
			return fail(inputErr)
		}
		deviceFeeds[projection] = pointer
		norm, pointer, inputErr := r.deviceInput(builder, *r.weights.PerLayerProjectionNorm)
		if inputErr != nil {
			return fail(inputErr)
		}
		deviceFeeds[norm] = pointer
		perLayerInputs, inputErr = model.BuildGemma4PerLayerInputs(
			builder, current, builder.GetRows(perLayerTable, rows), projection, norm, r.spec,
		)
		if inputErr != nil {
			return fail(inputErr)
		}
	}
	keys := make([]*tensor.Tensor, len(r.weights.Layers))
	values := make([]*tensor.Tensor, len(r.weights.Layers))
	states := make([]deviceGraphStates, len(r.weights.Layers))
	decodeCatalog := plan.mode == deviceOutputGreedy && tokensPerSequence == 1 && r.decodeWeights != nil
	for layerIndex, info := range r.weights.Layers {
		plan := r.layerPlan(layerIndex, info.Recurrent)
		var graphWeights model.LayerGraphWeights
		var layerFeeds map[*tensor.Tensor]driver.DevicePtr
		var layerErr error
		if decodeCatalog {
			graphWeights, layerFeeds, layerErr = r.layerDecodeDeviceInputs(builder, info)
		} else {
			graphWeights, layerFeeds, layerErr = r.layerDeviceInputs(builder, info)
		}
		if layerErr != nil {
			return fail(layerErr)
		}
		for node, pointer := range layerFeeds {
			deviceFeeds[node] = pointer
		}
		sideInputs := layerSideInputs{embeddingSkip: embeddingSkip}
		if len(perLayerInputs) > 0 {
			sideInputs.perLayerInput = perLayerInputs[layerIndex]
		}
		boundSideInputs, sideErr := bindLayerSideInputs(
			builder, r.spec, positions, plan, hostFeeds, &graphWeights, sideInputs,
		)
		if sideErr != nil {
			return fail(sideErr)
		}
		if plan.Attention == model.AttentionQwenGDN {
			cacheInputs, inputErr := r.deviceBatchLayerCacheInputs(
				builder, prefix, layerIndex, info, past, hostFeeds, deviceFeeds,
			)
			if inputErr != nil {
				return fail(inputErr)
			}
			pastKey, pastValue := cacheInputs.key, cacheInputs.value
			convState := cacheInputs.states[model.CacheStateConvolution].Value
			ssmState := cacheInputs.states[model.CacheStateSSM].Value
			if plan.Recurrent {
				convState, ssmState = pastKey, pastValue
				pastKey, pastValue = nil, nil
			}
			var result model.Qwen35BlockResult
			var buildErr error
			if sequences == 1 {
				result, buildErr = model.BuildQwen35BlockCached(
					builder, current, r.spec, graphWeights, positions, plan.Recurrent,
					pastKey, pastValue, convState, ssmState,
				)
			} else {
				result, buildErr = model.BuildQwen35BlockCachedBatch(
					builder, current, r.spec, graphWeights, positions, sequences, plan.Recurrent,
					pastKey, pastValue, convState, ssmState,
				)
			}
			if buildErr != nil {
				return fail(buildErr)
			}
			current = result.Output
			if plan.Recurrent {
				keys[layerIndex], values[layerIndex] = result.ConvState, result.SSMState
			} else {
				keys[layerIndex], values[layerIndex] = result.Key, result.Value
			}
			continue
		}
		if plan.Attention == model.AttentionLFM2 {
			cacheInputs, inputErr := r.deviceBatchLayerCacheInputs(
				builder, prefix, layerIndex, info, past, hostFeeds, deviceFeeds,
			)
			if inputErr != nil {
				return fail(inputErr)
			}
			result, buildErr := model.BuildLFM2BlockCached(
				builder, current, r.spec, graphWeights, positions, plan.Recurrent,
				cacheInputs.key, cacheInputs.value, uint32(layerIndex),
			)
			if buildErr != nil {
				return fail(buildErr)
			}
			current = result.Output
			keys[layerIndex], values[layerIndex] = result.Key, result.Value
			continue
		}
		cacheInputs := layerGraphCacheInputs{states: make(model.CacheStates[*tensor.Tensor])}
		if plan.SharedKV {
			cacheInputs.key, cacheInputs.value = keys[plan.KVSource], values[plan.KVSource]
		} else {
			cacheInputs, layerErr = r.deviceBatchLayerCacheInputs(
				builder, prefix, layerIndex, info, past, hostFeeds, deviceFeeds,
			)
			if layerErr != nil {
				return fail(layerErr)
			}
		}
		result, buildErr := model.BuildArchitectureBlockCached(model.BlockDispatchOptions{
			Context: model.CachedBlockContext{
				Builder: builder, Input: current, Positions: positions, TokenRows: rows,
				PastKey: cacheInputs.key, PastValue: cacheInputs.value,
				PastStates:       cacheInputs.states,
				CurrentPositions: boundSideInputs.currentPositions,
				PerLayerInput:    graphWeights.PerLayerInput,
				Layer:            uint32(layerIndex), Recurrent: plan.Recurrent,
			},
			Spec: r.spec, Weights: graphWeights, Plan: &plan,
		})
		if buildErr != nil {
			return fail(buildErr)
		}
		current = result.Output
		keys[layerIndex], values[layerIndex] = result.Key, result.Value
		states[layerIndex] = result.States
	}
	current, err = r.applyDeviceOutputNorm(builder, current, deviceFeeds)
	if err != nil {
		return fail(err)
	}
	outputInfo := r.outputTensor()
	var outputTable *tensor.Tensor
	var outputPointer driver.DevicePtr
	if decodeCatalog {
		outputTable, outputPointer, err = r.decodeDeviceInput(builder, outputInfo)
	} else {
		outputTable, outputPointer, err = r.deviceInput(builder, outputInfo)
	}
	if err != nil {
		return fail(err)
	}
	deviceFeeds[outputTable] = outputPointer
	width := uint64(r.spec.EmbeddingLength)
	hiddenElements, err := current.Shape.Elements()
	if err != nil {
		return fail(err)
	}
	lastHidden := current
	if sequences == 1 {
		lastHidden = builder.FlatSlice(current, hiddenElements-width, width, 1)
	} else if tokensPerSequence != 1 {
		return fail(errors.New("packed output selection requires one token per sequence"))
	}
	logits := builder.MulMat(outputTable, lastHidden)
	if r.weights.OutputBias != nil {
		bias, pointer, biasErr := r.deviceInput(builder, *r.weights.OutputBias)
		if biasErr != nil {
			return fail(biasErr)
		}
		deviceFeeds[bias] = pointer
		logits = builder.Add(logits, bias)
	}
	if scale := r.spec.OutputLogitMultiplier(); scale != 1 {
		logits = builder.Scale(logits, scale)
	}
	var selection, candidates *tensor.Tensor
	switch plan.mode {
	case deviceOutputGreedy:
		selection = builder.TopK(logits, 1)
	case deviceOutputTopK:
		candidates = builder.TopKPairs(logits, plan.topK)
	}
	return deviceBatchGraph{
		logits: logits, selection: selection, candidates: candidates, keys: keys, values: values, states: states,
		pastTokens: pastTokens, nextPosition: nextPosition,
		tokenCount: uint32(tokensPerSequence), sequences: uint32(sequences),
	}, nil
}

func (r *Runner) deviceBatchLayerCacheInputs(
	builder *tensor.Builder,
	prefix string,
	layerIndex int,
	info model.LayerWeights,
	past *deviceKVCache,
	hostFeeds map[*tensor.Tensor]reference.Value,
	deviceFeeds map[*tensor.Tensor]driver.DevicePtr,
) (layerGraphCacheInputs, error) {
	plan := r.layerPlan(layerIndex, info.Recurrent)
	name := func(suffix string) string {
		return prefix + fmt.Sprintf("blk.%d.%s", layerIndex, suffix)
	}
	inputDevice := func(suffix string, value executor.DeviceValue) *tensor.Tensor {
		input := builder.Input(name(suffix), dtype.F32, value.Shape)
		deviceFeeds[input] = value.Pointer
		return input
	}
	inputZero := func(suffix string, shape tensor.Shape) *tensor.Tensor {
		input := builder.Input(name(suffix), dtype.F32, shape)
		hostFeeds[input] = reference.ZeroValue(shape)
		return input
	}
	schema, err := model.CacheSchemaForPlan(r.spec, plan, info, 0)
	if err != nil {
		return layerGraphCacheInputs{}, err
	}
	var source *layerCacheSource[executor.DeviceValue]
	if past != nil {
		var states deviceLayerStates
		if layerIndex < len(past.States) {
			states = past.States[layerIndex]
		}
		source = &layerCacheSource[executor.DeviceValue]{
			primary: model.NewCachePair(past.Keys[layerIndex], past.Values[layerIndex]),
			states:  states,
		}
	}
	return bindLayerGraphCacheInputs(schema, source, inputDevice, inputZero), nil
}

func rebuildDeviceCachePages(
	cache *deviceKVCache,
	pageTokens uint32,
) error {
	if cache == nil {
		return errors.New("inference: device cache is nil")
	}
	pageTokens = resolveCachePageTokens(pageTokens)
	cache.PageTokens = pageTokens
	pageCount := int(cache.Tokens / pageTokens)
	if cache.Tokens%pageTokens != 0 {
		pageCount++
	}
	if cap(cache.Pages) < pageCount {
		cache.Pages = make([]deviceKVPage, pageCount)
	} else {
		cache.Pages = cache.Pages[:pageCount]
	}
	pageIndex := 0
	for start := uint32(0); start < cache.Tokens; start += pageTokens {
		count := pageTokens
		if remaining := cache.Tokens - start; remaining < count {
			count = remaining
		}
		page := &cache.Pages[pageIndex]
		page.Start, page.Tokens = start, count
		page.Keys = resizeDeviceValues(page.Keys, len(cache.Keys))
		page.Values = resizeDeviceValues(page.Values, len(cache.Values))
		for layer := range cache.Keys {
			var err error
			page.Keys[layer], err = deviceCachePageValue(
				cache.Keys[layer], start, count, cache.Tokens, layer,
			)
			if err != nil {
				return err
			}
			page.Values[layer], err = deviceCachePageValue(
				cache.Values[layer], start, count, cache.Tokens, layer,
			)
			if err != nil {
				return err
			}
		}
		pageIndex++
	}
	return nil
}

func resizeDeviceValues(values []executor.DeviceValue, count int) []executor.DeviceValue {
	if cap(values) < count {
		return make([]executor.DeviceValue, count)
	}
	return values[:count]
}

func deviceCachePageValue(
	source executor.DeviceValue,
	start, count, tokens uint32,
	layer int,
) (executor.DeviceValue, error) {
	if source.Shape.Rank != 3 || source.Shape.Dims[2] != uint64(tokens) {
		return source, nil
	}
	stride := source.Shape.Dims[0] * source.Shape.Dims[1]
	if stride == 0 || stride > math.MaxUint64/4 ||
		uint64(start) > math.MaxUint64/(stride*4) {
		return executor.DeviceValue{}, fmt.Errorf(
			"inference: device cache layer %d page offset overflows", layer,
		)
	}
	offset := uint64(start) * stride * 4
	if uint64(source.Pointer) > math.MaxUint64-offset {
		return executor.DeviceValue{}, fmt.Errorf(
			"inference: device cache layer %d page pointer overflows", layer,
		)
	}
	source.Pointer += driver.DevicePtr(offset)
	source.Shape.Dims[2] = uint64(count)
	return source, nil
}
