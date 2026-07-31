package inference

import (
	"context"
	"errors"
	"fmt"
	"math"

	"llamacpp2go/internal/cuda/driver"
	"llamacpp2go/internal/cuda/executor"
	"llamacpp2go/internal/model"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/dtype"
	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
)

type deviceKVCache struct {
	Outputs  *executor.RetainedOutputs
	Keys     []executor.DeviceValue
	Values   []executor.DeviceValue
	Tokens   uint32
	Position uint32
	Logits   []float32
}

func (c *deviceKVCache) Release(ctx context.Context) error {
	if c == nil || c.Outputs == nil {
		return nil
	}
	err := c.Outputs.Release(ctx)
	c.Outputs = nil
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
	total := uint64(cache.Tokens) + uint64(incoming)
	if total <= uint64(r.spec.ContextLength) {
		return nil
	}
	if uint64(incoming) > uint64(r.spec.ContextLength) {
		return fmt.Errorf(
			"inference: new token count %d exceeds context length %d",
			incoming,
			r.spec.ContextLength,
		)
	}
	discardCount, err := contextDiscardCount(
		cache.Tokens,
		incoming,
		r.spec.ContextLength,
		0,
		requestedDiscard,
	)
	if err != nil {
		return err
	}
	discard := uint64(discardCount)
	remaining := uint64(cache.Tokens) - discard
	for layerIndex := range cache.Keys {
		recurrent := r.spec.Architecture == "qwen35" &&
			r.weights.Layers[layerIndex].Recurrent
		if recurrent {
			continue
		}
		for _, value := range []*executor.DeviceValue{
			&cache.Keys[layerIndex],
			&cache.Values[layerIndex],
		} {
			if value.Shape.Rank != 3 ||
				value.Shape.Dims[2] != uint64(cache.Tokens) {
				return fmt.Errorf(
					"inference: layer %d device cache shape %v does not contain %d tokens",
					layerIndex,
					value.Shape.Slice(),
					cache.Tokens,
				)
			}
			stride := value.Shape.Dims[0] * value.Shape.Dims[1]
			offset := discard * stride * 4
			if uint64(value.Pointer) > math.MaxUint64-offset {
				return errors.New("inference: shifted device cache pointer overflows")
			}
			value.Pointer += driver.DevicePtr(offset)
			value.Shape.Dims[2] = remaining
		}
	}
	cache.Tokens = uint32(remaining)
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
	total := uint64(cache.Tokens) + uint64(incoming)
	if total <= uint64(r.spec.ContextLength) {
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
	discard, err := contextDiscardCount(
		cache.Tokens,
		incoming,
		r.spec.ContextLength,
		keep,
		requestedDiscard,
	)
	if err != nil {
		return nil, err
	}
	if len(cache.Keys) != len(r.weights.Layers) ||
		len(cache.Values) != len(r.weights.Layers) {
		return nil, errors.New("inference: device cache layer count differs")
	}
	copies := make([]executor.DeviceCopy, 0, len(cache.Keys)*2)
	for layerIndex := range cache.Keys {
		recurrent := r.spec.Architecture == "qwen35" &&
			r.weights.Layers[layerIndex].Recurrent
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
	}
	outputs, values, err := r.cuda.CopyDeviceValues(ctx, copies)
	if err != nil {
		return nil, err
	}
	next := &deviceKVCache{
		Outputs:  outputs,
		Keys:     make([]executor.DeviceValue, len(cache.Keys)),
		Values:   make([]executor.DeviceValue, len(cache.Values)),
		Tokens:   cache.Tokens - discard,
		Position: cache.Position,
	}
	for index := range next.Keys {
		next.Keys[index] = values[2*index]
		next.Values[index] = values[2*index+1]
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
	if len(tokenIDs) == 0 {
		return reference.Value{}, nil, errors.New("inference: token sequence is empty")
	}
	var pastTokens, nextPosition uint32
	if past != nil {
		pastTokens = past.Tokens
		nextPosition = past.Position
		if len(past.Keys) != len(r.weights.Layers) || len(past.Values) != len(r.weights.Layers) {
			return reference.Value{}, nil, errors.New("inference: device cache layer count differs")
		}
	}
	if uint64(pastTokens)+uint64(len(tokenIDs)) > uint64(r.spec.ContextLength) {
		return reference.Value{}, nil, fmt.Errorf(
			"inference: cached plus new token count %d exceeds context length %d",
			uint64(pastTokens)+uint64(len(tokenIDs)),
			r.spec.ContextLength,
		)
	}
	if uint64(nextPosition)+uint64(len(tokenIDs)) > math.MaxUint32 {
		return reference.Value{}, nil, errors.New("inference: absolute token position exceeds uint32")
	}
	rows := make([]uint32, len(tokenIDs))
	positions := make([]uint32, len(tokenIDs))
	for index, id := range tokenIDs {
		if id < 0 || int(id) >= r.vocab.Len() {
			return reference.Value{}, nil, fmt.Errorf("inference: token ID %d is out of range", id)
		}
		rows[index] = uint32(id)
		positions[index] = nextPosition + uint32(index)
	}
	builder := tensor.NewBuilder()
	embeddingTable, embeddingPointer, err := r.deviceInput(builder, r.weights.TokenEmbedding)
	if err != nil {
		return reference.Value{}, nil, err
	}
	current := builder.GetRows(embeddingTable, rows)
	hostFeeds := map[*tensor.Tensor]reference.Value{}
	deviceFeeds := map[*tensor.Tensor]driver.DevicePtr{embeddingTable: embeddingPointer}
	if r.weights.PositionEmbedding != nil {
		for _, position := range positions {
			if position >= r.spec.ContextLength {
				return reference.Value{}, nil, fmt.Errorf(
					"inference: learned position %d exceeds context length %d",
					position,
					r.spec.ContextLength,
				)
			}
		}
		positionTable, positionPointer, positionErr := r.deviceInput(
			builder, *r.weights.PositionEmbedding,
		)
		if positionErr != nil {
			return reference.Value{}, nil, positionErr
		}
		deviceFeeds[positionTable] = positionPointer
		current = builder.Add(current, builder.GetRows(positionTable, positions))
	}
	if scale := r.spec.InputEmbeddingScale(); scale != 1 {
		current = builder.Scale(current, scale)
	}
	if r.weights.TokenEmbeddingNorm != nil {
		normWeight, normPointer, normErr := r.deviceInput(
			builder, *r.weights.TokenEmbeddingNorm,
		)
		if normErr != nil {
			return reference.Value{}, nil, normErr
		}
		deviceFeeds[normWeight] = normPointer
		var normBias *tensor.Tensor
		if r.weights.TokenEmbeddingNormBias != nil {
			normBias, normPointer, normErr = r.deviceInput(
				builder, *r.weights.TokenEmbeddingNormBias,
			)
			if normErr != nil {
				return reference.Value{}, nil, normErr
			}
			deviceFeeds[normBias] = normPointer
		}
		current = model.ApplyNormalization(builder, current, normWeight, normBias, r.spec)
	}
	keys := make([]*tensor.Tensor, len(r.weights.Layers))
	values := make([]*tensor.Tensor, len(r.weights.Layers))
	for layerIndex, info := range r.weights.Layers {
		graphWeights, layerFeeds, layerErr := r.layerDeviceInputs(builder, info)
		if layerErr != nil {
			return reference.Value{}, nil, layerErr
		}
		for node, pointer := range layerFeeds {
			deviceFeeds[node] = pointer
		}
		if r.spec.Architecture == "qwen35" {
			var pastKey, pastValue, convState, ssmState *tensor.Tensor
			if past != nil {
				first := past.Keys[layerIndex]
				second := past.Values[layerIndex]
				firstInput := builder.Input(
					fmt.Sprintf("blk.%d.state_0", layerIndex),
					dtype.F32,
					first.Shape,
				)
				secondInput := builder.Input(
					fmt.Sprintf("blk.%d.state_1", layerIndex),
					dtype.F32,
					second.Shape,
				)
				deviceFeeds[firstInput] = first.Pointer
				deviceFeeds[secondInput] = second.Pointer
				if info.Recurrent {
					convState, ssmState = firstInput, secondInput
				} else {
					pastKey, pastValue = firstInput, secondInput
				}
			} else if info.Recurrent {
				convChannels := uint64(r.spec.SSMInnerSize) +
					2*uint64(r.spec.SSMStateSize)*uint64(r.spec.SSMGroupCount)
				convShape := tensor.MustShape(
					uint64(r.spec.SSMConvKernel-1),
					convChannels,
				)
				ssmShape := tensor.MustShape(
					uint64(r.spec.SSMStateSize),
					uint64(r.spec.SSMStateSize),
					uint64(r.spec.SSMTimeStepRank),
					1,
				)
				convState = builder.Input(
					fmt.Sprintf("blk.%d.conv_state", layerIndex),
					dtype.F32,
					convShape,
				)
				ssmState = builder.Input(
					fmt.Sprintf("blk.%d.ssm_state", layerIndex),
					dtype.F32,
					ssmShape,
				)
				convElements, _ := convShape.Elements()
				ssmElements, _ := ssmShape.Elements()
				hostFeeds[convState] = reference.Value{
					Shape: convShape,
					Data:  make([]float32, int(convElements)),
				}
				hostFeeds[ssmState] = reference.Value{
					Shape: ssmShape,
					Data:  make([]float32, int(ssmElements)),
				}
			}
			result, layerErr := model.BuildQwen35BlockCached(
				builder,
				current,
				r.spec,
				graphWeights,
				positions,
				info.Recurrent,
				pastKey,
				pastValue,
				convState,
				ssmState,
			)
			if layerErr != nil {
				return reference.Value{}, nil, layerErr
			}
			current = result.Output
			if info.Recurrent {
				keys[layerIndex], values[layerIndex] = result.ConvState, result.SSMState
			} else {
				keys[layerIndex], values[layerIndex] = result.Key, result.Value
			}
		} else {
			var pastKey, pastValue *tensor.Tensor
			if past != nil {
				pastKey = builder.Input(
					fmt.Sprintf("blk.%d.cache_key", layerIndex),
					dtype.F32,
					past.Keys[layerIndex].Shape,
				)
				pastValue = builder.Input(
					fmt.Sprintf("blk.%d.cache_value", layerIndex),
					dtype.F32,
					past.Values[layerIndex].Shape,
				)
				deviceFeeds[pastKey] = past.Keys[layerIndex].Pointer
				deviceFeeds[pastValue] = past.Values[layerIndex].Pointer
			}
			result, layerErr := model.BuildDenseBlockCachedForLayer(
				builder,
				current,
				r.spec,
				graphWeights,
				positions,
				pastKey,
				pastValue,
				uint32(layerIndex),
			)
			if layerErr != nil {
				return reference.Value{}, nil, layerErr
			}
			current = result.Output
			keys[layerIndex] = result.Key
			values[layerIndex] = result.Value
		}
	}
	current, err = r.applyDeviceOutputNorm(builder, current, deviceFeeds)
	if err != nil {
		return reference.Value{}, nil, err
	}
	outputInfo := r.weights.TokenEmbedding
	if r.weights.Output != nil {
		outputInfo = *r.weights.Output
	}
	outputTable, outputPointer, err := r.deviceInput(builder, outputInfo)
	if err != nil {
		return reference.Value{}, nil, err
	}
	deviceFeeds[outputTable] = outputPointer
	width := uint64(r.spec.EmbeddingLength)
	hiddenElements, err := current.Shape.Elements()
	if err != nil {
		return reference.Value{}, nil, err
	}
	lastHidden := builder.FlatSlice(current, hiddenElements-width, width, 1)
	logitsTensor := builder.MulMat(outputTable, lastHidden)
	if r.weights.OutputBias != nil {
		bias, biasPointer, biasErr := r.deviceInput(builder, *r.weights.OutputBias)
		if biasErr != nil {
			return reference.Value{}, nil, biasErr
		}
		deviceFeeds[bias] = biasPointer
		logitsTensor = builder.Add(logitsTensor, bias)
	}
	if scale := r.spec.OutputLogitMultiplier(); scale != 1 {
		logitsTensor = builder.Scale(logitsTensor, scale)
	}
	if err := builder.Err(); err != nil {
		return reference.Value{}, nil, err
	}
	outputs := make([]*tensor.Tensor, 1, 1+2*len(keys))
	outputs[0] = logitsTensor
	for layerIndex := range keys {
		outputs = append(outputs, keys[layerIndex], values[layerIndex])
	}
	retained, err := r.cuda.ExecuteRetainedWithDeviceFeeds(
		ctx,
		outputs,
		hostFeeds,
		deviceFeeds,
	)
	if err != nil {
		return reference.Value{}, nil, err
	}
	fail := func(cause error) (reference.Value, *deviceKVCache, error) {
		_ = retained.Release(context.Background())
		return reference.Value{}, nil, cause
	}
	next := &deviceKVCache{
		Outputs:  retained,
		Keys:     make([]executor.DeviceValue, len(keys)),
		Values:   make([]executor.DeviceValue, len(values)),
		Tokens:   pastTokens + uint32(len(tokenIDs)),
		Position: nextPosition + uint32(len(tokenIDs)),
	}
	logits, err := retained.CopyToHost(ctx, logitsTensor)
	if err != nil {
		return fail(err)
	}
	next.Logits = applyLogitSoftcap(
		logits.Data,
		r.spec.FinalLogitSoftcap,
	)
	var ok bool
	for layerIndex := range keys {
		next.Keys[layerIndex], ok = retained.Value(keys[layerIndex])
		if !ok {
			return fail(fmt.Errorf("inference: missing retained key for layer %d", layerIndex))
		}
		next.Values[layerIndex], ok = retained.Value(values[layerIndex])
		if !ok {
			return fail(fmt.Errorf("inference: missing retained value for layer %d", layerIndex))
		}
	}
	return reference.Value{}, next, nil
}
