package inference

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sync"

	"llamacpp2go/internal/cuda/driver"
	"llamacpp2go/internal/cuda/executor"
	"llamacpp2go/internal/model"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/dtype"
	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
)

type deviceKVCache struct {
	owner      *deviceCacheOwner
	Keys       []executor.DeviceValue
	Values     []executor.DeviceValue
	States     []deviceLayerStates
	Pages      []deviceKVPage
	PageTokens uint32
	Tokens     uint32
	Position   uint32
	Logits     []float32
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
	o.outputs = nil
	return err
}

type deviceKVPage struct {
	Start  uint32
	Tokens uint32
	Keys   []executor.DeviceValue
	Values []executor.DeviceValue
}

func (c *deviceKVCache) Release(ctx context.Context) error {
	if c == nil || c.owner == nil {
		return nil
	}
	err := c.owner.release(ctx)
	c.owner = nil
	return err
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
		Tokens: tokenIDs,
		Past:   past,
	}})
	if err != nil {
		return reference.Value{}, nil, err
	}
	return reference.Value{}, next[0], nil
}

type deviceBatchAppend struct {
	Tokens []tokenizer.TokenID
	Past   *deviceKVCache
}

type deviceBatchGraph struct {
	logits       *tensor.Tensor
	keys         []*tensor.Tensor
	values       []*tensor.Tensor
	states       []deviceGraphStates
	pastTokens   uint32
	nextPosition uint32
	tokenCount   uint32
}

type deviceGraphState = model.CacheState[*tensor.Tensor]
type deviceGraphStates = model.CacheStates[*tensor.Tensor]

// forwardDeviceCachedBatchLocked: one graph, variable independent branches.
func (r *Runner) forwardDeviceCachedBatchLocked(
	ctx context.Context,
	appends []deviceBatchAppend,
) ([]*deviceKVCache, error) {
	if len(appends) == 0 {
		return nil, errors.New("inference: device batch is empty")
	}
	builder := r.newGraphBuilder()
	hostFeeds := make(map[*tensor.Tensor]reference.Value)
	deviceFeeds := make(map[*tensor.Tensor]driver.DevicePtr)
	graphs := make([]deviceBatchGraph, len(appends))
	outputs := make([]*tensor.Tensor, 0, len(appends)*(1+2*len(r.weights.Layers)))
	for index, appendInput := range appends {
		graph, err := r.buildDeviceCachedBatchBranch(
			builder, index, appendInput.Tokens, appendInput.Past, hostFeeds, deviceFeeds,
		)
		if err != nil {
			return nil, fmt.Errorf("inference: device batch branch %d: %w", index, err)
		}
		graphs[index] = graph
		outputs = append(outputs, graph.logits)
		for layer := range graph.keys {
			outputs = append(outputs, graph.keys[layer], graph.values[layer])
			outputs = graph.states[layer].AppendValues(outputs)
		}
	}
	if err := builder.Err(); err != nil {
		return nil, err
	}
	retained, err := r.cuda.ExecuteRetainedWithDeviceFeeds(
		ctx, outputs, hostFeeds, deviceFeeds,
	)
	if err != nil {
		return nil, err
	}
	fail := func(cause error) ([]*deviceKVCache, error) {
		_ = retained.Release(context.Background())
		return nil, cause
	}
	next := make([]*deviceKVCache, len(graphs))
	for index, graph := range graphs {
		logits, copyErr := retained.CopyToHost(ctx, graph.logits)
		if copyErr != nil {
			return fail(copyErr)
		}
		cache := &deviceKVCache{
			Keys:       make([]executor.DeviceValue, len(graph.keys)),
			Values:     make([]executor.DeviceValue, len(graph.values)),
			States:     make([]deviceLayerStates, len(graph.states)),
			Tokens:     graph.pastTokens + graph.tokenCount,
			Position:   graph.nextPosition + graph.tokenCount,
			PageTokens: r.cachePageTokens,
			Logits:     r.finalizeLogits(logits.Data),
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

func (r *Runner) buildDeviceCachedBatchBranch(
	builder *tensor.Builder,
	branch int,
	tokenIDs []tokenizer.TokenID,
	past *deviceKVCache,
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
	sequence, err := r.planForwardSequence(tokenIDs, pastTokens, nextPosition)
	if err != nil {
		return fail(err)
	}
	rows, positions := sequence.rows, sequence.positions
	prefix := fmt.Sprintf("seq.%d.", branch)
	embeddingTable, embeddingPointer, err := r.deviceInput(builder, r.weights.TokenEmbedding)
	if err != nil {
		return fail(err)
	}
	current := builder.GetRows(embeddingTable, rows)
	deviceFeeds[embeddingTable] = embeddingPointer
	if r.weights.PositionEmbedding != nil {
		if positionErr := validateLearnedPositions(positions, r.spec.ContextLength); positionErr != nil {
			return fail(positionErr)
		}
		positionTable, pointer, positionErr := r.deviceInput(builder, *r.weights.PositionEmbedding)
		if positionErr != nil {
			return fail(positionErr)
		}
		deviceFeeds[positionTable] = pointer
		current = builder.Add(current, builder.GetRows(positionTable, positions))
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
	for layerIndex, info := range r.weights.Layers {
		plan := r.layerPlan(layerIndex, info.Recurrent)
		graphWeights, layerFeeds, layerErr := r.layerDeviceInputs(builder, info)
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
			result, buildErr := model.BuildQwen35BlockCached(
				builder, current, r.spec, graphWeights, positions, plan.Recurrent,
				pastKey, pastValue, convState, ssmState,
			)
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
	outputTable, outputPointer, err := r.deviceInput(builder, outputInfo)
	if err != nil {
		return fail(err)
	}
	deviceFeeds[outputTable] = outputPointer
	width := uint64(r.spec.EmbeddingLength)
	hiddenElements, err := current.Shape.Elements()
	if err != nil {
		return fail(err)
	}
	lastHidden := builder.FlatSlice(current, hiddenElements-width, width, 1)
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
	return deviceBatchGraph{
		logits: logits, keys: keys, values: values, states: states,
		pastTokens: pastTokens, nextPosition: nextPosition,
		tokenCount: uint32(len(tokenIDs)),
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
		elements, _ := shape.Elements()
		hostFeeds[input] = reference.Value{Shape: shape, Data: make([]float32, int(elements))}
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
	cache.Pages = cache.Pages[:0]
	for start := uint32(0); start < cache.Tokens; start += pageTokens {
		count := pageTokens
		if remaining := cache.Tokens - start; remaining < count {
			count = remaining
		}
		page := deviceKVPage{
			Start: start, Tokens: count,
			Keys:   make([]executor.DeviceValue, len(cache.Keys)),
			Values: make([]executor.DeviceValue, len(cache.Values)),
		}
		for layer := range cache.Keys {
			for source, destination := range map[*executor.DeviceValue]*executor.DeviceValue{
				&cache.Keys[layer]:   &page.Keys[layer],
				&cache.Values[layer]: &page.Values[layer],
			} {
				*destination = *source
				if source.Shape.Rank != 3 || source.Shape.Dims[2] != uint64(cache.Tokens) {
					continue
				}
				stride := source.Shape.Dims[0] * source.Shape.Dims[1]
				if stride == 0 || stride > math.MaxUint64/4 ||
					uint64(start) > math.MaxUint64/(stride*4) {
					return fmt.Errorf("inference: device cache layer %d page offset overflows", layer)
				}
				offset := uint64(start) * stride * 4
				if uint64(source.Pointer) > math.MaxUint64-offset {
					return fmt.Errorf("inference: device cache layer %d page pointer overflows", layer)
				}
				destination.Pointer += driver.DevicePtr(offset)
				destination.Shape.Dims[2] = uint64(count)
			}
		}
		cache.Pages = append(cache.Pages, page)
	}
	return nil
}
