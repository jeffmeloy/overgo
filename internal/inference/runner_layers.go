package inference

import (
	"context"
	"fmt"

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
	if len(attentionBlockIDs) > 0 {
		shape := tensor.MustShape(uint64(len(attentionBlockIDs)))
		attentionBlockInput = builder.Input("model.attention_block_ids", dtype.F32, shape)
		hostFeeds[attentionBlockInput] = reference.Value{Shape: shape, Data: attentionBlockIDs}
	}
	keys := make([]*tensor.Tensor, len(r.weights.Layers))
	values := make([]*tensor.Tensor, len(r.weights.Layers))
	captured := make(map[int32]*tensor.Tensor)
	for layerIndex, info := range r.weights.Layers {
		plan := r.layerPlan(layerIndex, info.Recurrent)
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
		if len(perLayerInputs) > 0 {
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
		var result model.DenseBlockResult
		if multiPositions != nil {
			result, err = model.BuildDenseBlockCachedForLayerWithMultiPositions(
				builder, current, r.spec, graphWeights, [4][]uint32(*multiPositions),
				pastKey, pastValue, uint32(layerIndex),
			)
		} else {
			result, err = model.BuildDenseBlockCachedForLayer(
				builder, current, r.spec, graphWeights, positions,
				pastKey, pastValue, uint32(layerIndex),
			)
		}
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
	}
	if applyOutputNorm {
		normalized, normErr := r.applyDeviceOutputNorm(builder, current, deviceFeeds)
		if normErr != nil {
			return reference.Value{}, nil, normErr
		}
		current = normalized
	}
	outputs := make([]*tensor.Tensor, 1, 1+2*len(keys))
	outputs[0] = current
	for layerIndex := range keys {
		outputs = append(outputs, keys[layerIndex], values[layerIndex])
	}
	if capture != nil {
		for _, layer := range capture.order {
			if node := captured[layer]; node != nil {
				outputs = append(outputs, node)
			}
		}
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
		plan := r.layerPlan(layerIndex, info.Recurrent)
		graphWeights, err := runtime.layer(info, fmt.Sprintf("blk.%d.", layerIndex))
		if err != nil {
			return reference.Value{}, err
		}
		if _, err := bindLayerSideInputs(
			builder, r.spec, positions, plan, hostFeeds, &graphWeights, layerSideInputs{},
		); err != nil {
			return reference.Value{}, err
		}
		result, err := model.BuildDenseBlockCachedForLayer(
			builder,
			current,
			r.spec,
			graphWeights,
			positions,
			nil,
			nil,
			uint32(layerIndex),
		)
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
	plan := r.layerPlan(layerIndex, info.Recurrent)
	if plan.Attention == model.AttentionQwenGDN {
		return r.runQwen35LayerCached(
			ctx,
			activation,
			info,
			layerIndex,
			positions,
			past,
			multiPositions,
		)
	}
	if plan.Attention == model.AttentionLFM2 {
		return r.runLFM2LayerCached(ctx, activation, info, layerIndex, positions, past)
	}
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
	if len(attentionBlockIDs) > 0 {
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
	cacheInputs, cacheErr := r.hostLayerCacheInputs(builder, layerIndex, info, plan, past, hostFeeds)
	if cacheErr != nil {
		return reference.Value{}, LayerCache{}, cacheErr
	}
	var result model.DenseBlockResult
	var dispatchMultiPositions *[4][]uint32
	if multiPositions != nil {
		converted := [4][]uint32(*multiPositions)
		dispatchMultiPositions = &converted
	}
	result, err = model.BuildArchitectureBlockCached(model.BlockDispatchOptions{
		Context: model.CachedBlockContext{
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
		},
		Spec: r.spec, Weights: graphWeights, Plan: &plan,
	})
	if err != nil {
		return reference.Value{}, LayerCache{}, err
	}
	outputTensor := result.Output
	if r.hasPreloadedWeights() && layerIndex == len(r.weights.Layers)-1 {
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
	if len(result.States) > 0 {
		layerCache.States = model.MapCacheStateValues(
			result.States, func(value *tensor.Tensor) reference.Value { return results[value] },
		)
	}
	return results[outputTensor], layerCache, nil
}

func (r *Runner) runLFM2LayerCached(
	ctx context.Context,
	activation reference.Value,
	info model.LayerWeights,
	layerIndex int,
	positions []uint32,
	past *LayerCache,
) (reference.Value, LayerCache, error) {
	runtime := r.newInferenceGraphRuntime(ctx)
	builder := runtime.builder
	input := runtime.input("input", activation)
	hostFeeds, deviceFeeds := runtime.feeds.Host, runtime.feeds.Device
	graphWeights, err := runtime.layer(info, fmt.Sprintf("blk.%d.", layerIndex))
	if err != nil {
		return reference.Value{}, LayerCache{}, err
	}
	var pastKey, pastValue *tensor.Tensor
	if info.Recurrent {
		keyValue := reference.Value{}
		valueValue := reference.Value{}
		if past == nil {
			shape := tensor.MustShape(
				uint64(r.spec.ShortConvCacheLength-1), uint64(r.spec.EmbeddingLength),
			)
			elements, _ := shape.Elements()
			keyValue = reference.Value{Shape: shape, Data: make([]float32, int(elements))}
			valueValue = reference.Value{Shape: tensor.MustShape(1), Data: []float32{0}}
		} else {
			keyValue, valueValue = past.Key, past.Value
		}
		pastKey = builder.Input(fmt.Sprintf("blk.%d.%s", layerIndex, model.CacheStateConvolution), dtype.F32, keyValue.Shape)
		pastValue = builder.Input(fmt.Sprintf("blk.%d.reserved_state", layerIndex), dtype.F32, valueValue.Shape)
		hostFeeds[pastKey], hostFeeds[pastValue] = keyValue, valueValue
	} else if past != nil {
		pastKey = builder.Input(fmt.Sprintf("blk.%d.cache_key", layerIndex), dtype.F32, past.Key.Shape)
		pastValue = builder.Input(fmt.Sprintf("blk.%d.cache_value", layerIndex), dtype.F32, past.Value.Shape)
		hostFeeds[pastKey], hostFeeds[pastValue] = past.Key, past.Value
	}
	result, err := model.BuildLFM2BlockCached(
		builder, input, r.spec, graphWeights, positions, info.Recurrent, pastKey, pastValue,
		uint32(layerIndex),
	)
	if err != nil {
		return reference.Value{}, LayerCache{}, err
	}
	output := result.Output
	if r.hasPreloadedWeights() && layerIndex == len(r.weights.Layers)-1 {
		output, err = r.applyDeviceOutputNorm(builder, output, deviceFeeds)
		if err != nil {
			return reference.Value{}, LayerCache{}, err
		}
	}
	outputs := []*tensor.Tensor{output, result.Key, result.Value}
	results, err := runtime.execute(outputs...)
	if err != nil {
		return reference.Value{}, LayerCache{}, err
	}
	return results[output], LayerCache{Key: results[result.Key], Value: results[result.Value]}, nil
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
			uint64(r.spec.ShortConvCacheLength-1), uint64(r.spec.EmbeddingLength),
		)
		stateElements, _ := stateShape.Elements()
		stateValue := reference.Value{
			Shape: stateShape, Data: make([]float32, int(stateElements)),
		}
		reservedValue := reference.Value{
			Shape: tensor.MustShape(1), Data: []float32{0},
		}
		state = builder.Input(fmt.Sprintf("blk.%d.%s", layerIndex, model.CacheStateConvolution), dtype.F32, stateShape)
		reserved = builder.Input(
			fmt.Sprintf("blk.%d.reserved_state", layerIndex), dtype.F32, reservedValue.Shape,
		)
		hostFeeds[state], hostFeeds[reserved] = stateValue, reservedValue
	}
	spec := r.spec
	spec.NonCausalAttention = true
	result, err := model.BuildLFM2BlockCached(
		builder, input, spec, graphWeights, positions, info.Recurrent, state, reserved,
		uint32(layerIndex),
	)
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
	plan := r.layerPlan(layerIndex, info.Recurrent)
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
	result, err := model.BuildDenseBlockCachedForLayer(
		builder,
		input,
		r.spec,
		graphWeights,
		positions,
		nil,
		nil,
		uint32(layerIndex),
	)
	if err != nil {
		return reference.Value{}, err
	}
	results, err := runtime.execute(result.Output)
	if err != nil {
		return reference.Value{}, err
	}
	return results[result.Output], nil
}

func (r *Runner) runQwen35LayerCached(
	ctx context.Context,
	activation reference.Value,
	info model.LayerWeights,
	layerIndex int,
	positions []uint32,
	past *LayerCache,
	multiPositions *MultiAxisPositions,
) (reference.Value, LayerCache, error) {
	runtime := r.newInferenceGraphRuntime(ctx)
	builder := runtime.builder
	input := runtime.input("input", activation)
	hostFeeds, deviceFeeds := runtime.feeds.Host, runtime.feeds.Device
	graphWeights, err := runtime.layer(info, fmt.Sprintf("blk.%d.", layerIndex))
	if err != nil {
		return reference.Value{}, LayerCache{}, err
	}

	var pastKey, pastValue, convState, ssmState *tensor.Tensor
	if info.Recurrent {
		var convValue, ssmValue reference.Value
		if past == nil {
			convChannels := uint64(r.spec.SSMInnerSize) +
				2*uint64(r.spec.SSMStateSize)*uint64(r.spec.SSMGroupCount)
			convShape := tensor.MustShape(uint64(r.spec.SSMConvKernel-1), convChannels)
			ssmShape := tensor.MustShape(
				uint64(r.spec.SSMStateSize),
				uint64(r.spec.SSMStateSize),
				uint64(r.spec.SSMTimeStepRank),
				1,
			)
			convElements, _ := convShape.Elements()
			ssmElements, _ := ssmShape.Elements()
			convValue = reference.Value{Shape: convShape, Data: make([]float32, int(convElements))}
			ssmValue = reference.Value{Shape: ssmShape, Data: make([]float32, int(ssmElements))}
		} else {
			convValue = past.Key
			ssmValue = past.Value
		}
		convState = builder.Input(
			fmt.Sprintf("blk.%d.%s", layerIndex, model.CacheStateConvolution),
			dtype.F32,
			convValue.Shape,
		)
		ssmState = builder.Input(
			fmt.Sprintf("blk.%d.%s", layerIndex, model.CacheStateSSM),
			dtype.F32,
			ssmValue.Shape,
		)
		hostFeeds[convState] = convValue
		hostFeeds[ssmState] = ssmValue
	} else if past != nil {
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
	var result model.Qwen35BlockResult
	if multiPositions != nil {
		result, err = model.BuildQwen35BlockCachedWithMultiPositions(
			builder, input, r.spec, graphWeights, [4][]uint32(*multiPositions),
			info.Recurrent,
			pastKey, pastValue, convState, ssmState,
		)
	} else {
		result, err = model.BuildQwen35BlockCached(
			builder, input, r.spec, graphWeights, positions, info.Recurrent,
			pastKey, pastValue, convState, ssmState,
		)
	}
	if err != nil {
		return reference.Value{}, LayerCache{}, err
	}
	outputTensor := result.Output
	if r.hasPreloadedWeights() && layerIndex == len(r.weights.Layers)-1 {
		outputTensor, err = r.applyDeviceOutputNorm(builder, result.Output, deviceFeeds)
		if err != nil {
			return reference.Value{}, LayerCache{}, err
		}
	}
	outputs := []*tensor.Tensor{outputTensor}
	if info.Recurrent {
		outputs = append(outputs, result.ConvState, result.SSMState)
	} else {
		outputs = append(outputs, result.Key, result.Value)
	}
	results, err := runtime.execute(outputs...)
	if err != nil {
		return reference.Value{}, LayerCache{}, err
	}
	if info.Recurrent {
		return results[outputTensor], LayerCache{
			Key:   results[result.ConvState],
			Value: results[result.SSMState],
		}, nil
	}
	return results[outputTensor], LayerCache{
		Key:   results[result.Key],
		Value: results[result.Value],
	}, nil
}

func (r *Runner) runOutputNorm(ctx context.Context, activation reference.Value) (reference.Value, error) {
	if r.profile().OutputNorm == model.OutputNormAbsent {
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
