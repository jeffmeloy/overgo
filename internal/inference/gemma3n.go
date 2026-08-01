package inference

import (
	"context"
	"errors"
	"fmt"
	"math"

	"llamacpp2go/internal/cuda/driver"
	"llamacpp2go/internal/model"
	"llamacpp2go/internal/quant"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/dtype"
	"llamacpp2go/internal/tensor/reference"
)

func (r *Runner) forwardGemma3nCachedLocked(
	ctx context.Context,
	activation reference.Value,
	perLayerInputs []reference.Value,
	positions []uint32,
	cache *KVCache,
	pastTokens, nextPosition uint32,
) (reference.Value, *KVCache, error) {
	if r.weights.AltUpProjection == nil || r.weights.AltUpUnembedding == nil ||
		len(perLayerInputs) != len(r.weights.Layers) {
		return reference.Value{}, nil, errors.New("inference: Gemma 3n top-level weights are incomplete")
	}
	projection, err := model.LoadHostTensor(ctx, r.file, *r.weights.AltUpProjection)
	if err != nil {
		return reference.Value{}, nil, err
	}
	unembedding, err := model.LoadHostTensor(ctx, r.file, *r.weights.AltUpUnembedding)
	if err != nil {
		return reference.Value{}, nil, err
	}
	states, err := gemma3nInitializeAltUp(activation, projection, int(r.spec.AltUpCount))
	if err != nil {
		return reference.Value{}, nil, err
	}
	nextCache := &KVCache{
		Layers:   make([]LayerCache, len(r.weights.Layers)),
		Tokens:   pastTokens + uint32(len(positions)),
		Position: nextPosition + uint32(len(positions)),
	}
	for layerIndex, info := range r.weights.Layers {
		hostLayer, loadErr := model.LoadHostLayer(ctx, r.file, info)
		if loadErr != nil {
			return reference.Value{}, nil, fmt.Errorf("inference layer %d: %w", layerIndex, loadErr)
		}
		predictions, predictErr := gemma3nPredict(states, hostLayer, r.spec)
		if predictErr != nil {
			return reference.Value{}, nil, fmt.Errorf("inference layer %d: %w", layerIndex, predictErr)
		}
		var past *LayerCache
		if !r.spec.LayerHasKV(uint32(layerIndex)) {
			source := r.spec.LayerSharedKVSource(uint32(layerIndex))
			past = &nextCache.Layers[source]
		} else if cache != nil {
			past = &cache.Layers[layerIndex]
		}
		activated, layerCache, stageErr := r.runGemma3nActiveLayer(
			ctx, predictions[r.spec.AltUpActive], hostLayer, info,
			layerIndex, positions, past,
		)
		if stageErr != nil {
			return reference.Value{}, nil, fmt.Errorf("inference layer %d: %w", layerIndex, stageErr)
		}
		states, err = gemma3nCorrectAndInject(
			predictions, activated, perLayerInputs[layerIndex], hostLayer, r.spec,
		)
		if err != nil {
			return reference.Value{}, nil, fmt.Errorf("inference layer %d: %w", layerIndex, err)
		}
		nextCache.Layers[layerIndex] = layerCache
	}
	activation, err = gemma3nMergeAltUp(states, unembedding, int(r.spec.AltUpActive))
	if err != nil {
		return reference.Value{}, nil, err
	}
	activation, err = r.runOutputNorm(ctx, activation)
	if err != nil {
		return reference.Value{}, nil, err
	}
	return activation, nextCache, nil
}

func (r *Runner) runGemma3nActiveLayer(
	ctx context.Context,
	input reference.Value,
	hostLayer model.HostLayer,
	info model.LayerWeights,
	layerIndex int,
	positions []uint32,
	past *LayerCache,
) (reference.Value, LayerCache, error) {
	builder := tensor.NewBuilder()
	inputNode := builder.Input("gemma3n.input", dtype.F32, input.Shape)
	hostFeeds := map[*tensor.Tensor]reference.Value{inputNode: input}
	deviceFeeds := make(map[*tensor.Tensor]driver.DevicePtr)
	graphWeights, err := r.gemma3nGraphWeights(builder, hostLayer, info, hostFeeds, deviceFeeds)
	if err != nil {
		return reference.Value{}, LayerCache{}, err
	}
	var pastKey, pastValue *tensor.Tensor
	if past != nil {
		pastKey = builder.Input("gemma3n.cache_key", dtype.F32, past.Key.Shape)
		pastValue = builder.Input("gemma3n.cache_value", dtype.F32, past.Value.Shape)
		hostFeeds[pastKey], hostFeeds[pastValue] = past.Key, past.Value
	}
	stage, err := model.BuildGemma3nAttentionStage(
		builder, inputNode, r.spec, graphWeights, positions,
		pastKey, pastValue, uint32(layerIndex),
	)
	if err != nil {
		return reference.Value{}, LayerCache{}, err
	}
	outputs := []*tensor.Tensor{stage.Residual, stage.Gate, stage.Up, stage.Key, stage.Value}
	results, err := r.executeGemma3n(ctx, outputs, hostFeeds, deviceFeeds)
	if err != nil {
		return reference.Value{}, LayerCache{}, err
	}
	activated, err := gemma3nActivateFFN(
		results[stage.Gate], results[stage.Up],
		layerIndex < int(r.spec.SparseLayerCount), r.spec.SparsityStdMultiplier,
	)
	if err != nil {
		return reference.Value{}, LayerCache{}, err
	}
	builder = tensor.NewBuilder()
	residualNode := builder.Input("gemma3n.residual", dtype.F32, results[stage.Residual].Shape)
	activatedNode := builder.Input("gemma3n.ffn_activated", dtype.F32, activated.Shape)
	hostFeeds = map[*tensor.Tensor]reference.Value{
		residualNode:  results[stage.Residual],
		activatedNode: activated,
	}
	deviceFeeds = make(map[*tensor.Tensor]driver.DevicePtr)
	graphWeights, err = r.gemma3nGraphWeights(builder, hostLayer, info, hostFeeds, deviceFeeds)
	if err != nil {
		return reference.Value{}, LayerCache{}, err
	}
	output, err := model.BuildGemma3nFeedForwardOutput(
		builder, residualNode, activatedNode, r.spec, graphWeights,
	)
	if err != nil {
		return reference.Value{}, LayerCache{}, err
	}
	results, err = r.executeGemma3n(ctx, []*tensor.Tensor{output}, hostFeeds, deviceFeeds)
	if err != nil {
		return reference.Value{}, LayerCache{}, err
	}
	return results[output], LayerCache{Key: results[stage.Key], Value: results[stage.Value]}, nil
}

func (r *Runner) gemma3nGraphWeights(
	builder *tensor.Builder,
	hostLayer model.HostLayer,
	info model.LayerWeights,
	hostFeeds map[*tensor.Tensor]reference.Value,
	deviceFeeds map[*tensor.Tensor]driver.DevicePtr,
) (model.LayerGraphWeights, error) {
	if r.hasPreloadedWeights() {
		weights, feeds, err := r.layerDeviceInputs(builder, info)
		if err != nil {
			return model.LayerGraphWeights{}, err
		}
		for node, pointer := range feeds {
			deviceFeeds[node] = pointer
		}
		return weights, nil
	}
	weights, feeds, err := hostLayer.GraphInputs(builder, "gemma3n.")
	if err != nil {
		return model.LayerGraphWeights{}, err
	}
	for node, value := range feeds {
		hostFeeds[node] = value
	}
	return weights, nil
}

func (r *Runner) executeGemma3n(
	ctx context.Context,
	outputs []*tensor.Tensor,
	hostFeeds map[*tensor.Tensor]reference.Value,
	deviceFeeds map[*tensor.Tensor]driver.DevicePtr,
) (map[*tensor.Tensor]reference.Value, error) {
	if r.hasPreloadedWeights() {
		return r.cuda.ExecuteWithDeviceFeeds(ctx, outputs, hostFeeds, deviceFeeds)
	}
	return r.cuda.Execute(ctx, outputs, hostFeeds)
}

func gemma3nInitializeAltUp(
	input, projection reference.Value,
	count int,
) ([]reference.Value, error) {
	if count < 2 || projection.Shape.Rank != 3 || projection.Shape.Dims[2] != uint64(count-1) {
		return nil, errors.New("inference: Gemma 3n AltUp projection shape is invalid")
	}
	states := make([]reference.Value, count)
	states[0] = cloneReferenceValue(input)
	for index := 1; index < count; index++ {
		projected, err := gemma3nMatMulSlice(projection, index-1, input)
		if err != nil {
			return nil, err
		}
		states[index] = gemma3nMatchMagnitude(projected, input)
	}
	return states, nil
}

func gemma3nPredict(
	states []reference.Value,
	layer model.HostLayer,
	spec model.Spec,
) ([]reference.Value, error) {
	modalities, err := gemma3nModalities(states[spec.AltUpActive], layer, spec)
	if err != nil {
		return nil, err
	}
	coefficients, err := gemma3nMatMul(*layer.AltUpPredictCoefficient, modalities)
	if err != nil {
		return nil, err
	}
	count := len(states)
	if coefficients.Shape.Dims[0] != uint64(count*count) {
		return nil, errors.New("inference: Gemma 3n prediction coefficient shape is invalid")
	}
	result := make([]reference.Value, count)
	for output := range count {
		result[output] = cloneReferenceValue(states[output])
		for token := 0; token < int(states[output].Shape.Dims[1]); token++ {
			for feature := 0; feature < int(states[output].Shape.Dims[0]); feature++ {
				position := token*int(states[output].Shape.Dims[0]) + feature
				var delta float64
				for source := range count {
					coefficient := coefficients.Data[token*count*count+output*count+source]
					delta += float64(states[source].Data[position]) * float64(coefficient)
				}
				result[output].Data[position] += float32(delta)
			}
		}
	}
	return result, nil
}

func gemma3nCorrectAndInject(
	predictions []reference.Value,
	activated, perLayer reference.Value,
	layer model.HostLayer,
	spec model.Spec,
) ([]reference.Value, error) {
	modalities, err := gemma3nModalities(activated, layer, spec)
	if err != nil {
		return nil, err
	}
	coefficients, err := gemma3nMatMul(*layer.AltUpCorrectCoefficient, modalities)
	if err != nil {
		return nil, err
	}
	active := int(spec.AltUpActive)
	result := make([]reference.Value, len(predictions))
	for index := range predictions {
		result[index] = cloneReferenceValue(predictions[index])
		for token := 0; token < int(activated.Shape.Dims[1]); token++ {
			coefficient := coefficients.Data[token*len(predictions)+index] + 1
			for feature := 0; feature < int(activated.Shape.Dims[0]); feature++ {
				position := token*int(activated.Shape.Dims[0]) + feature
				innovation := activated.Data[position] - predictions[active].Data[position]
				result[index].Data[position] += innovation * coefficient
			}
		}
	}
	scaled := cloneReferenceValue(result[active])
	for index := range scaled.Data {
		scaled.Data[index] *= layer.AltUpCorrectScale.Data[index%int(scaled.Shape.Dims[0])]
	}
	gate, err := gemma3nMatMul(*layer.PerLayerInputGate, scaled)
	if err != nil {
		return nil, err
	}
	if !gate.Shape.Equal(perLayer.Shape) {
		return nil, errors.New("inference: Gemma 3n per-layer input shape differs")
	}
	for index := range gate.Data {
		gate.Data[index] = gemma3nGELU(gate.Data[index]) * perLayer.Data[index]
	}
	injection, err := gemma3nMatMul(*layer.PerLayerProjection, gate)
	if err != nil {
		return nil, err
	}
	injection, err = gemma3nWeightedRMS(injection, *layer.PerLayerPostNorm, spec.RMSNormEpsilon)
	if err != nil {
		return nil, err
	}
	for index := 1; index < len(result); index++ {
		for valueIndex := range result[index].Data {
			result[index].Data[valueIndex] += injection.Data[valueIndex]
		}
	}
	return result, nil
}

func gemma3nModalities(
	input reference.Value,
	layer model.HostLayer,
	spec model.Spec,
) (reference.Value, error) {
	if layer.AltUpRouterNorm == nil || layer.AltUpRouter == nil {
		return reference.Value{}, errors.New("inference: Gemma 3n router weights are incomplete")
	}
	normalized, err := gemma3nWeightedRMS(input, *layer.AltUpRouterNorm, spec.RMSNormEpsilon)
	if err != nil {
		return reference.Value{}, err
	}
	for index := range normalized.Data {
		normalized.Data[index] /= float32(spec.EmbeddingLength)
	}
	result, err := gemma3nMatMul(*layer.AltUpRouter, normalized)
	if err != nil {
		return reference.Value{}, err
	}
	for index := range result.Data {
		result.Data[index] = float32(math.Tanh(float64(result.Data[index])))
	}
	return result, nil
}

func gemma3nActivateFFN(
	gate, up reference.Value,
	sparse bool,
	standardDeviationMultiplier float32,
) (reference.Value, error) {
	if !gate.Shape.Equal(up.Shape) || gate.Shape.Rank != 2 || gate.Shape.Dims[0] < 2 {
		return reference.Value{}, errors.New("inference: Gemma 3n FFN projection shape is invalid")
	}
	result := reference.Value{Shape: gate.Shape, Data: make([]float32, len(gate.Data))}
	width := int(gate.Shape.Dims[0])
	for token := 0; token < int(gate.Shape.Dims[1]); token++ {
		base := token * width
		cutoff := float32(-math.MaxFloat32)
		if sparse {
			var sum float64
			for index := range width {
				sum += float64(gate.Data[base+index])
			}
			mean := sum / float64(width)
			var squared float64
			for index := range width {
				delta := float64(gate.Data[base+index]) - mean
				squared += delta * delta
			}
			cutoff = float32(mean + float64(standardDeviationMultiplier)*math.Sqrt(squared/float64(width-1)))
		}
		for index := range width {
			value := gate.Data[base+index]
			if sparse {
				value = max(value-cutoff, 0)
			}
			result.Data[base+index] = gemma3nGELU(value) * up.Data[base+index]
		}
	}
	return result, nil
}

func gemma3nMergeAltUp(
	states []reference.Value,
	unembedding reference.Value,
	active int,
) (reference.Value, error) {
	if len(states) < 2 || active < 0 || active >= len(states) ||
		unembedding.Shape.Rank != 3 || unembedding.Shape.Dims[2] != uint64(len(states)-1) {
		return reference.Value{}, errors.New("inference: Gemma 3n unembedding shape is invalid")
	}
	result := cloneReferenceValue(states[active])
	for index := 1; index < len(states); index++ {
		projected, err := gemma3nMatMulSlice(unembedding, index-1, states[index])
		if err != nil {
			return reference.Value{}, err
		}
		projected = gemma3nMatchMagnitude(projected, states[active])
		for valueIndex := range result.Data {
			result.Data[valueIndex] += projected.Data[valueIndex]
		}
	}
	for index := range result.Data {
		result.Data[index] /= float32(len(states))
	}
	return result, nil
}

func gemma3nMatMul(weight, input reference.Value) (reference.Value, error) {
	if weight.Shape.Rank != 2 || input.Shape.Rank != 2 || weight.Shape.Dims[0] != input.Shape.Dims[0] {
		return reference.Value{}, errors.New("inference: Gemma 3n matrix dimensions differ")
	}
	inner := int(weight.Shape.Dims[0])
	outputWidth := int(weight.Shape.Dims[1])
	tokens := int(input.Shape.Dims[1])
	output := reference.Value{
		Shape: tensor.MustShape(uint64(outputWidth), uint64(tokens)),
		Data:  make([]float32, outputWidth*tokens),
	}
	for token := range tokens {
		for row := range outputWidth {
			var sum float64
			for column := range inner {
				sum += float64(weight.Data[row*inner+column]) * float64(input.Data[token*inner+column])
			}
			output.Data[token*outputWidth+row] = float32(sum)
		}
	}
	return output, nil
}

func gemma3nMatMulSlice(weight reference.Value, slice int, input reference.Value) (reference.Value, error) {
	if weight.Shape.Rank != 3 || slice < 0 || uint64(slice) >= weight.Shape.Dims[2] {
		return reference.Value{}, errors.New("inference: Gemma 3n grouped projection slice is invalid")
	}
	size := int(weight.Shape.Dims[0] * weight.Shape.Dims[1])
	start := slice * size
	return gemma3nMatMul(reference.Value{
		Shape: tensor.MustShape(weight.Shape.Dims[0], weight.Shape.Dims[1]),
		Data:  weight.Data[start : start+size],
	}, input)
}

func gemma3nWeightedRMS(
	input, weight reference.Value,
	epsilon float32,
) (reference.Value, error) {
	if input.Shape.Rank != 2 || weight.Shape.Rank != 1 || input.Shape.Dims[0] != weight.Shape.Dims[0] {
		return reference.Value{}, errors.New("inference: Gemma 3n RMS norm shape is invalid")
	}
	result := cloneReferenceValue(input)
	width := int(input.Shape.Dims[0])
	for token := 0; token < int(input.Shape.Dims[1]); token++ {
		base := token * width
		var squared float64
		for index := range width {
			value := float64(input.Data[base+index])
			squared += value * value
		}
		scale := 1 / float32(math.Sqrt(squared/float64(width)+float64(epsilon)))
		for index := range width {
			result.Data[base+index] *= scale * weight.Data[index]
		}
	}
	return result, nil
}

func gemma3nMatchMagnitude(input, target reference.Value) reference.Value {
	result := cloneReferenceValue(input)
	width := int(input.Shape.Dims[0])
	for token := 0; token < int(input.Shape.Dims[1]); token++ {
		base := token * width
		var inputSquared, targetSquared float64
		for index := range width {
			inputValue := float64(input.Data[base+index])
			targetValue := float64(target.Data[base+index])
			inputSquared += inputValue * inputValue
			targetSquared += targetValue * targetValue
		}
		if inputSquared == 0 {
			continue
		}
		scale := float32(math.Sqrt(targetSquared / inputSquared))
		for index := range width {
			result.Data[base+index] *= scale
		}
	}
	return result
}

func gemma3nGELU(value float32) float32 {
	if value <= -10 {
		return 0
	}
	if value >= 10 {
		return value
	}
	x := float64(quant.Float16ToFloat32(quant.Float32ToFloat16(value)))
	result := float32(0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*x*(1+0.044715*x*x))))
	return quant.Float16ToFloat32(quant.Float32ToFloat16(result))
}

func cloneReferenceValue(value reference.Value) reference.Value {
	return reference.Value{Shape: value.Shape, Data: append([]float32(nil), value.Data...)}
}
