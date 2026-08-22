package inference

import (
	"context"
	"fmt"

	"overgo/internal/checked"
	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func (r *Runner) forwardDenseLayersPreloaded(
	ctx context.Context,
	activation reference.Value,
	embeddingSkip reference.Value,
	perLayerInputs []reference.Value,
	positions []uint32,
	multiPositions *MultiAxisPositions,
	deepstackBase reference.Value,
	deepstackInputs []reference.Value,
	attentionBlockIDs []float32,
	cache *KVCache,
	nextCache *KVCache,
	visualMode bool,
	applyOutputNorm bool,
	capture *layerInputCapture,
) (reference.Value, *KVCache, error) {
	runtime := r.newInferenceGraphRuntime(ctx)
	builder := runtime.builder
	input := runtime.input("model.input", activation)
	current := input
	hostFeeds, deviceFeeds := runtime.feeds.Host, runtime.feeds.Device
	var attentionBlockInput *tensor.Tensor
	if checked.Nonzero(len(attentionBlockIDs)) {
		shape := tensor.MustShape(uint64(len(attentionBlockIDs)))
		attentionBlockInput = builder.Input("model.attention_block_ids", dtype.F32, shape)
		hostFeeds[attentionBlockInput] = reference.Value{Shape: shape, Data: attentionBlockIDs}
	}
	keys := make([]*tensor.Tensor, len(r.weights.Layers))
	values := make([]*tensor.Tensor, len(r.weights.Layers))
	captured := make(map[int32]*tensor.Tensor)
	var attnQueryNode *tensor.Tensor
	for layerIndex, info := range r.weights.Layers {
		program := r.layerProgram(layerIndex)
		plan := program.Layer()
		if stream := deepstackInputForLayer(plan.DeepstackBefore, deepstackBase, deepstackInputs); stream != nil {
			deepstack := builder.Input(
				fmt.Sprintf("blk.%d.deepstack_input", layerIndex), dtype.F32, stream.Shape,
			)
			hostFeeds[deepstack] = *stream
			current = builder.Add(current, deepstack)
		}
		if capture.wants(layerIndex) {
			captured[int32(layerIndex)] = current
		}
		graphWeights, err := runtime.layer(info, fmt.Sprintf("blk.%d.", layerIndex))
		if err != nil {
			return reference.Value{}, nil, err
		}
		if visualMode {
			if err := r.applyCogVLMVisualWeights(
				ctx, builder, info, &graphWeights, nil, deviceFeeds,
			); err != nil {
				return reference.Value{}, nil, err
			}
		}
		sideInputs := layerSideInputs{embeddingSkip: input, attentionBlock: attentionBlockInput}
		if checked.Nonzero(len(perLayerInputs)) {
			perLayer := builder.Input(
				fmt.Sprintf("blk.%d.per_layer_input", layerIndex),
				dtype.F32,
				perLayerInputs[layerIndex].Shape,
			)
			hostFeeds[perLayer] = perLayerInputs[layerIndex]
			sideInputs.perLayerInput = perLayer
		}
		if _, err := bindLayerSideInputs(
			builder, r.spec, positions, plan, hostFeeds, &graphWeights, sideInputs,
		); err != nil {
			return reference.Value{}, nil, err
		}
		var pastKey, pastValue *tensor.Tensor
		if plan.SharedKV {
			pastKey, pastValue = keys[plan.KVSource], values[plan.KVSource]
		} else if cache != nil {
			past := cache.Layers[layerIndex]
			pastKey = builder.Input(
				fmt.Sprintf("blk.%d.cache_key", layerIndex),
				dtype.F32,
				past.Key.Shape,
			)
			pastValue = builder.Input(
				fmt.Sprintf("blk.%d.cache_value", layerIndex),
				dtype.F32,
				past.Value.Shape,
			)
			hostFeeds[pastKey] = past.Key
			hostFeeds[pastValue] = past.Value
		}
		var axes *[tensor.MaxDimensions][]uint32
		if multiPositions != nil {
			converted := [tensor.MaxDimensions][]uint32(*multiPositions)
			axes = &converted
		}
		result, err := program.Build(model.CachedBlockContext{
			Builder: builder, Input: current, Positions: positions, MultiPositions: axes,
			PastKey: pastKey, PastValue: pastValue, Layer: plan.Layer,
		}, graphWeights)
		if err != nil {
			return reference.Value{}, nil, err
		}
		current = result.Output
		if stream := deepstackInputForLayer(plan.DeepstackAfter, deepstackBase, deepstackInputs); stream != nil {
			deepstack := builder.Input(
				fmt.Sprintf("blk.%d.deepstack_output", layerIndex), dtype.F32, stream.Shape,
			)
			hostFeeds[deepstack] = *stream
			current = builder.Add(current, deepstack)
		}
		keys[layerIndex] = result.Key
		values[layerIndex] = result.Value
		if capture != nil && capture.attention && capture.attnLayer == int32(layerIndex) && result.Query != nil {
			attnQueryNode = result.Query
			capture.attnScale = result.AttentionScale
			capture.attnHeads = int(r.spec.LayerHeadCount(plan.Layer))
			capture.attnKV = int(r.spec.LayerKVHeadCount(plan.Layer))
			capture.attnDim = int(r.spec.KeyLength)
		}
	}
	if applyOutputNorm {
		normalized, normErr := r.applyDeviceOutputNorm(builder, current, deviceFeeds)
		if normErr != nil {
			return reference.Value{}, nil, normErr
		}
		current = normalized
	}
	outputs := []*tensor.Tensor{current}
	seenOutputs := map[*tensor.Tensor]struct{}{current: {}}
	for layerIndex := range keys {
		outputs = appendUniqueGraphOutputs(outputs, seenOutputs, keys[layerIndex], values[layerIndex])
	}
	if capture != nil {
		for _, layer := range capture.order {
			if node := captured[layer]; node != nil {
				outputs = appendUniqueGraphOutputs(outputs, seenOutputs, node)
			}
		}
	}
	if attnQueryNode != nil {
		outputs = appendUniqueGraphOutputs(outputs, seenOutputs, attnQueryNode)
	}
	results, err := runtime.execute(outputs...)
	if err != nil {
		return reference.Value{}, nil, err
	}
	for layerIndex := range keys {
		nextCache.Layers[layerIndex] = LayerCache{
			Key:   results[keys[layerIndex]],
			Value: results[values[layerIndex]],
		}
	}
	for layer, node := range captured {
		capture.set(int(layer), results[node])
	}
	if attnQueryNode != nil {
		capture.attnQuery = results[attnQueryNode].Clone()
		capture.attnKey = results[keys[capture.attnLayer]].Clone()
	}
	return results[current], nextCache, nil
}

func (r *Runner) forwardDenseLayersNoCachePreloaded(
	ctx context.Context,
	activation reference.Value,
	positions []uint32,
) (reference.Value, error) {
	runtime := r.newInferenceGraphRuntime(ctx)
	builder := runtime.builder
	input := runtime.input("model.input", activation)
	current := input
	hostFeeds, deviceFeeds := runtime.feeds.Host, runtime.feeds.Device
	for layerIndex, info := range r.weights.Layers {
		program := r.layerProgram(layerIndex)
		plan := program.Layer()
		graphWeights, err := runtime.layer(info, fmt.Sprintf("blk.%d.", layerIndex))
		if err != nil {
			return reference.Value{}, err
		}
		if _, err := bindLayerSideInputs(
			builder, r.spec, positions, plan, hostFeeds, &graphWeights, layerSideInputs{},
		); err != nil {
			return reference.Value{}, err
		}
		result, err := program.Build(model.CachedBlockContext{
			Builder: builder, Input: current, Positions: positions, Layer: plan.Layer,
		}, graphWeights)
		if err != nil {
			return reference.Value{}, err
		}
		current = result.Output
	}
	var err error
	current, err = r.applyDeviceOutputNorm(builder, current, deviceFeeds)
	if err != nil {
		return reference.Value{}, err
	}
	results, err := runtime.execute(current)
	if err != nil {
		return reference.Value{}, err
	}
	return results[current], nil
}

func (r *Runner) runLayerCached(
	ctx context.Context,
	activation reference.Value,
	info model.LayerWeights,
	layerIndex int,
	positions []uint32,
	tokenRows []uint32,
	past *LayerCache,
	embeddingSkip reference.Value,
	perLayerInput *reference.Value,
	multiPositions *MultiAxisPositions,
	visualMode bool,
	attentionBlockIDs []float32,
) (reference.Value, LayerCache, error) {
	program := r.layerProgram(layerIndex)
	plan := program.Layer()
	runtime := r.newInferenceGraphRuntime(ctx)
	builder := runtime.builder
	input := runtime.input("input", activation)
	hostFeeds, deviceFeeds := runtime.feeds.Host, runtime.feeds.Device
	graphWeights, err := runtime.layer(info, fmt.Sprintf("blk.%d.", layerIndex))
	if err != nil {
		return reference.Value{}, LayerCache{}, err
	}
	if visualMode {
		if err := r.applyCogVLMVisualWeights(
			ctx, builder, info, &graphWeights, hostFeeds, deviceFeeds,
		); err != nil {
			return reference.Value{}, LayerCache{}, err
		}
	}
	sideInputs := layerSideInputs{}
	if plan.EmbeddingSkip {
		skip := builder.Input("embedding_skip", dtype.F32, embeddingSkip.Shape)
		hostFeeds[skip] = embeddingSkip
		sideInputs.embeddingSkip = skip
	}
	if perLayerInput != nil {
		perLayer := builder.Input("per_layer_input", dtype.F32, perLayerInput.Shape)
		hostFeeds[perLayer] = *perLayerInput
		sideInputs.perLayerInput = perLayer
	}
	if checked.Nonzero(len(attentionBlockIDs)) {
		shape := tensor.MustShape(uint64(len(attentionBlockIDs)))
		blockInput := builder.Input("attention_block_ids", dtype.F32, shape)
		hostFeeds[blockInput] = reference.Value{Shape: shape, Data: attentionBlockIDs}
		sideInputs.attentionBlock = blockInput
	}
	boundSideInputs, sideErr := bindLayerSideInputs(
		builder, r.spec, positions, plan, hostFeeds, &graphWeights, sideInputs,
	)
	if sideErr != nil {
		return reference.Value{}, LayerCache{}, sideErr
	}
	cacheInputs, cacheErr := r.hostLayerCacheInputs(builder, layerIndex, past, hostFeeds)
	if cacheErr != nil {
		return reference.Value{}, LayerCache{}, cacheErr
	}
	var result model.DenseBlockResult
	var dispatchMultiPositions *[tensor.MaxDimensions][]uint32
	if multiPositions != nil {
		converted := [tensor.MaxDimensions][]uint32(*multiPositions)
		dispatchMultiPositions = &converted
	}
	result, err = program.Build(model.CachedBlockContext{
		Builder:          builder,
		Input:            input,
		Positions:        positions,
		MultiPositions:   dispatchMultiPositions,
		TokenRows:        tokenRows,
		PastKey:          cacheInputs.key,
		PastValue:        cacheInputs.value,
		PastStates:       cacheInputs.states,
		CurrentPositions: boundSideInputs.currentPositions,
		PerLayerInput:    graphWeights.PerLayerInput,
		Layer:            uint32(layerIndex),
		Recurrent:        info.Recurrent,
	}, graphWeights)
	if err != nil {
		return reference.Value{}, LayerCache{}, err
	}
	outputTensor := result.Output
	if r.hasPreloadedWeights() && layerIndex == len(r.weights.Layers)-tensor.SingletonExtent {
		outputTensor, err = r.applyDeviceOutputNorm(builder, result.Output, deviceFeeds)
		if err != nil {
			return reference.Value{}, LayerCache{}, err
		}
	}
	outputs := []*tensor.Tensor{outputTensor, result.Key, result.Value}
	if result.Auxiliary != nil {
		outputs = append(outputs, result.Auxiliary)
	}
	outputs = result.States.AppendValues(outputs)
	results, err := runtime.execute(outputs...)
	if err != nil {
		return reference.Value{}, LayerCache{}, err
	}
	layerCache := LayerCache{
		Key:   results[result.Key],
		Value: results[result.Value],
	}
	if result.Auxiliary != nil {
		auxiliary := results[result.Auxiliary]
		layerCache.Auxiliary = &auxiliary
	}
	if checked.Nonzero(len(result.States)) {
		layerCache.States = model.MapCacheStateValues(
			result.States, func(value *tensor.Tensor) reference.Value { return results[value] },
		)
	}
	return results[outputTensor], layerCache, nil
}

func (r *Runner) runLFM2LayerNonCausal(
	ctx context.Context,
	activation reference.Value,
	info model.LayerWeights,
	layerIndex int,
	positions []uint32,
) (reference.Value, error) {
	runtime := r.newInferenceGraphRuntime(ctx)
	builder := runtime.builder
	input := runtime.input("input", activation)
	hostFeeds := runtime.feeds.Host
	graphWeights, err := runtime.layer(info, fmt.Sprintf("blk.%d.", layerIndex))
	if err != nil {
		return reference.Value{}, err
	}
	var state, reserved *tensor.Tensor
	if info.Recurrent {
		stateShape := tensor.MustShape(
			uint64(r.spec.ShortConvCacheLength-uint32(tensor.SingletonExtent)), uint64(r.spec.EmbeddingLength),
		)
		stateElements, _ := stateShape.Elements()
		stateValue := reference.Value{
			Shape: stateShape, Data: make([]float32, int(stateElements)),
		}
		reservedShape := tensor.MustShape(tensor.SingletonExtent)
		reservedValue := reference.Value{Shape: reservedShape, Data: make([]float32, tensor.SingletonExtent)}
		state = builder.Input(fmt.Sprintf("blk.%d.%s", layerIndex, model.CacheStateConvolution), dtype.F32, stateShape)
		reserved = builder.Input(
			fmt.Sprintf("blk.%d.reserved_state", layerIndex), dtype.F32, reservedValue.Shape,
		)
		hostFeeds[state], hostFeeds[reserved] = stateValue, reservedValue
	}
	spec := r.spec
	spec.NonCausalAttention = true
	program, err := r.program.Model.LayerProgram(uint32(layerIndex))
	if err != nil {
		return reference.Value{}, err
	}
	plan := program.Layer()
	result, err := program.Build(model.CachedBlockContext{
		Builder: builder, Input: input, Positions: positions,
		PastKey: state, PastValue: reserved,
		Layer: plan.Layer, Recurrent: info.Recurrent,
	}, graphWeights)
	if err != nil {
		return reference.Value{}, err
	}
	results, err := runtime.execute(result.Output)
	if err != nil {
		return reference.Value{}, err
	}
	return results[result.Output], nil
}

func (r *Runner) runDenseLayerNoCache(
	ctx context.Context,
	activation reference.Value,
	info model.LayerWeights,
	layerIndex int,
	positions []uint32,
) (reference.Value, error) {
	program := r.layerProgram(layerIndex)
	plan := program.Layer()
	runtime := r.newInferenceGraphRuntime(ctx)
	builder := runtime.builder
	input := runtime.input("input", activation)
	hostFeeds := runtime.feeds.Host
	graphWeights, err := runtime.layer(info, fmt.Sprintf("blk.%d.", layerIndex))
	if err != nil {
		return reference.Value{}, err
	}
	if _, err := bindLayerSideInputs(
		builder, r.spec, positions, plan, hostFeeds, &graphWeights, layerSideInputs{},
	); err != nil {
		return reference.Value{}, err
	}
	result, err := program.Build(model.CachedBlockContext{
		Builder: builder, Input: input, Positions: positions, Layer: plan.Layer,
	}, graphWeights)
	if err != nil {
		return reference.Value{}, err
	}
	results, err := runtime.execute(result.Output)
	if err != nil {
		return reference.Value{}, err
	}
	return results[result.Output], nil
}

func (r *Runner) runOutputNorm(ctx context.Context, activation reference.Value) (reference.Value, error) {
	if r.program.Model.Terminal().Normalization == model.OutputNormAbsent {
		return activation, nil
	}
	runtime := r.newInferenceGraphRuntime(ctx)
	input := runtime.input("output_norm.input", activation)
	output, err := r.buildOutputNorm(runtime.builder, input, runtime.weight)
	if err != nil {
		return reference.Value{}, err
	}
	results, err := runtime.execute(output)
	if err != nil {
		return reference.Value{}, err
	}
	return results[output], nil
}

func (r *Runner) runUnweightedRMSNorm(
	ctx context.Context,
	activation reference.Value,
) (reference.Value, error) {
	runtime := r.newInferenceGraphRuntime(ctx)
	input := runtime.input("rms_norm.input", activation)
	output := runtime.builder.RMSNorm(input, r.spec.RMSNormEpsilon)
	results, err := runtime.execute(output)
	if err != nil {
		return reference.Value{}, err
	}
	return results[output], nil
}

// Greedy: tokenizes prompt and appends up to maxNewTokens argmax tokens
