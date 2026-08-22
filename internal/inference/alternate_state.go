package inference

import (
	"context"
	"errors"
	"fmt"
	"math"

	"overgo/internal/checked"
	"overgo/internal/hostmath"
	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

func (r *Runner) forwardAlternatePredictionsCachedLocked(
	ctx context.Context,
	program model.AlternateStateProgram,
	activation reference.Value,
	perLayerInputs []reference.Value,
	positions []uint32,
	cache *KVCache,
	pastTokens, nextPosition uint32,
) (reference.Value, *KVCache, error) {
	if r.weights.AltUpProjection == nil || r.weights.AltUpUnembedding == nil ||
		len(perLayerInputs) != len(r.weights.Layers) {
		return reference.Value{}, nil, errors.New("inference: alternate-state top-level weights are incomplete")
	}
	projection, err := r.hostTensor(ctx, *r.weights.AltUpProjection)
	if err != nil {
		return reference.Value{}, nil, err
	}
	unembedding, err := r.hostTensor(ctx, *r.weights.AltUpUnembedding)
	if err != nil {
		return reference.Value{}, nil, err
	}
	states, err := initializeAlternateStates(activation, projection, int(program.StateCount))
	if err != nil {
		return reference.Value{}, nil, err
	}
	nextCache := &KVCache{
		Layers:   make([]LayerCache, len(r.weights.Layers)),
		Tokens:   pastTokens + uint32(len(positions)),
		Position: nextPosition + uint32(len(positions)),
	}
	for layerIndex, info := range r.weights.Layers {
		plan := r.layerProgram(layerIndex).Layer()
		hostLayer, loadErr := r.hostLayer(ctx, fmt.Sprintf("blk.%d.", layerIndex), info)
		if loadErr != nil {
			return reference.Value{}, nil, fmt.Errorf("inference layer %d: %w", layerIndex, loadErr)
		}
		predictions, predictErr := predictAlternateStates(
			states, hostLayer, program.ActiveState, program.EmbeddingLength, program.NormalizationEpsilon,
		)
		if predictErr != nil {
			return reference.Value{}, nil, fmt.Errorf("inference layer %d: %w", layerIndex, predictErr)
		}
		var past *LayerCache
		if plan.SharedKV {
			past = &nextCache.Layers[plan.KVSource]
		} else if cache != nil {
			past = &cache.Layers[layerIndex]
		}
		activated, layerCache, stageErr := r.runAlternatePredictionLayer(
			ctx, program, predictions[program.ActiveState], hostLayer, info,
			layerIndex, positions, past,
		)
		if stageErr != nil {
			return reference.Value{}, nil, fmt.Errorf("inference layer %d: %w", layerIndex, stageErr)
		}
		states, err = correctAndInjectAlternateStates(
			predictions, activated, perLayerInputs[layerIndex], hostLayer,
			program.ActiveState, program.EmbeddingLength, program.NormalizationEpsilon,
		)
		if err != nil {
			return reference.Value{}, nil, fmt.Errorf("inference layer %d: %w", layerIndex, err)
		}
		nextCache.Layers[layerIndex] = layerCache
	}
	activation, err = mergeAlternateStates(states, unembedding, int(program.ActiveState))
	if err != nil {
		return reference.Value{}, nil, err
	}
	activation, err = r.runOutputNorm(ctx, activation)
	if err != nil {
		return reference.Value{}, nil, err
	}
	return activation, nextCache, nil
}

func (r *Runner) runAlternatePredictionLayer(
	ctx context.Context,
	alternate model.AlternateStateProgram,
	input reference.Value,
	hostLayer model.HostLayer,
	info model.LayerWeights,
	layerIndex int,
	positions []uint32,
	past *LayerCache,
) (reference.Value, LayerCache, error) {
	runtime := r.newInferenceGraphRuntime(ctx)
	inputNode := runtime.input("alternate_state.input", input)
	graphWeights, err := runtime.layerWithHost(info, "alternate_state.", &hostLayer)
	if err != nil {
		return reference.Value{}, LayerCache{}, err
	}
	var pastKey, pastValue *tensor.Tensor
	if past != nil {
		pastKey = runtime.input("alternate_state.cache_key", past.Key)
		pastValue = runtime.input("alternate_state.cache_value", past.Value)
	}
	program := r.layerProgram(layerIndex)
	stage, err := program.BuildActivationProjection(model.CachedBlockContext{
		Builder: runtime.builder, Input: inputNode, Positions: positions,
		PastKey: pastKey, PastValue: pastValue, Layer: uint32(layerIndex),
	}, graphWeights)
	if err != nil {
		return reference.Value{}, LayerCache{}, err
	}
	outputs := []*tensor.Tensor{stage.Residual, stage.Gate, stage.Up, stage.Key, stage.Value}
	results, err := runtime.execute(outputs...)
	if err != nil {
		return reference.Value{}, LayerCache{}, err
	}
	activated, err := activateAlternateFFN(
		results[stage.Gate], results[stage.Up],
		alternate.SparseLayer(uint32(layerIndex)), alternate.SparsityStdMultiplier,
	)
	if err != nil {
		return reference.Value{}, LayerCache{}, err
	}
	runtime = r.newInferenceGraphRuntime(ctx)
	residualNode := runtime.input("alternate_state.residual", results[stage.Residual])
	activatedNode := runtime.input("alternate_state.ffn_activated", activated)
	graphWeights, err = runtime.layerWithHost(info, "alternate_state.", &hostLayer)
	if err != nil {
		return reference.Value{}, LayerCache{}, err
	}
	output, err := program.BuildActivatedOutput(
		runtime.builder, residualNode, activatedNode, graphWeights,
	)
	if err != nil {
		return reference.Value{}, LayerCache{}, err
	}
	results, err = runtime.execute(output)
	if err != nil {
		return reference.Value{}, LayerCache{}, err
	}
	return results[output], LayerCache{Key: results[stage.Key], Value: results[stage.Value]}, nil
}

func initializeAlternateStates(
	input, projection reference.Value,
	count int,
) ([]reference.Value, error) {
	states := make([]reference.Value, count)
	states[0] = input.Clone()
	for index := tensor.SingletonExtent; index < count; index++ {
		projected, err := alternateLinearSlice(projection, index-tensor.SingletonExtent, input)
		if err != nil {
			return nil, err
		}
		rescaleMagnitudeInPlace(&projected, input)
		states[index] = projected
	}
	return states, nil
}

func predictAlternateStates(
	states []reference.Value,
	layer model.HostLayer,
	active uint32,
	embeddingLength uint32,
	normalizationEpsilon float32,
) ([]reference.Value, error) {
	if active >= uint32(len(states)) {
		return nil, errors.New("inference: alternate prediction active state is invalid")
	}
	modalities, err := alternateStateModalities(
		states[active], layer, embeddingLength, normalizationEpsilon,
	)
	if err != nil {
		return nil, err
	}
	coefficients, err := alternateLinear(*layer.AltUpPredictCoefficient, modalities)
	if err != nil {
		return nil, err
	}
	count := len(states)
	widthExtent, tokenExtent, _ := tensor.MatrixExtents(states[active].Shape)
	width, tokens := int(widthExtent), int(tokenExtent)
	result := make([]reference.Value, count)
	for output := range count {
		result[output] = states[output].Clone()
		for token := range tokens {
			for feature := range width {
				position := token*width + feature
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

func correctAndInjectAlternateStates(
	predictions []reference.Value,
	activated, perLayer reference.Value,
	layer model.HostLayer,
	activeState uint32,
	embeddingLength uint32,
	normalizationEpsilon float32,
) ([]reference.Value, error) {
	modalities, err := alternateStateModalities(
		activated, layer, embeddingLength, normalizationEpsilon,
	)
	if err != nil {
		return nil, err
	}
	coefficients, err := alternateLinear(*layer.AltUpCorrectCoefficient, modalities)
	if err != nil {
		return nil, err
	}
	active := int(activeState)
	if active >= len(predictions) {
		return nil, errors.New("inference: alternate correction active state is invalid")
	}
	widthExtent, tokenExtent, _ := tensor.MatrixExtents(activated.Shape)
	width, tokens := int(widthExtent), int(tokenExtent)
	result := make([]reference.Value, len(predictions))
	for index := range predictions {
		result[index] = predictions[index].Clone()
		for token := range tokens {
			coefficient := coefficients.Data[token*len(predictions)+index] + float32(tensor.SingletonExtent)
			for feature := range width {
				position := token*width + feature
				innovation := activated.Data[position] - predictions[active].Data[position]
				result[index].Data[position] += innovation * coefficient
			}
		}
	}
	scaled := result[active].Clone()
	for index := range scaled.Data {
		scaled.Data[index] *= layer.AltUpCorrectScale.Data[index%width]
	}
	gate, err := alternateLinear(*layer.PerLayerInputGate, scaled)
	if err != nil {
		return nil, err
	}
	if !gate.Shape.Equal(perLayer.Shape) {
		return nil, errors.New("inference: alternate-state per-layer input shape differs")
	}
	for index := range gate.Data {
		gate.Data[index] = reference.RoundedGELUTanh(gate.Data[index]) * perLayer.Data[index]
	}
	injection, err := alternateLinear(*layer.PerLayerProjection, gate)
	if err != nil {
		return nil, err
	}
	injection, err = alternateRMSNorm(injection, *layer.PerLayerPostNorm, normalizationEpsilon)
	if err != nil {
		return nil, err
	}
	for index := tensor.SingletonExtent; index < len(result); index++ {
		for valueIndex := range result[index].Data {
			result[index].Data[valueIndex] += injection.Data[valueIndex]
		}
	}
	return result, nil
}

func alternateStateModalities(
	input reference.Value,
	layer model.HostLayer,
	embeddingLength uint32,
	normalizationEpsilon float32,
) (reference.Value, error) {
	if layer.AltUpRouterNorm == nil || layer.AltUpRouter == nil {
		return reference.Value{}, errors.New("inference: alternate-state router weights are incomplete")
	}
	if !checked.Nonzero(embeddingLength) {
		return reference.Value{}, errors.New("inference: alternate router embedding width is invalid")
	}
	normalized, err := alternateRMSNorm(input, *layer.AltUpRouterNorm, normalizationEpsilon)
	if err != nil {
		return reference.Value{}, err
	}
	for index := range normalized.Data {
		normalized.Data[index] /= float32(embeddingLength)
	}
	result, err := alternateLinear(*layer.AltUpRouter, normalized)
	if err != nil {
		return reference.Value{}, err
	}
	for index := range result.Data {
		result.Data[index] = float32(math.Tanh(float64(result.Data[index])))
	}
	return result, nil
}

func activateAlternateFFN(
	gate, up reference.Value,
	sparse bool,
	standardDeviationMultiplier float32,
) (reference.Value, error) {
	widthExtent, tokenExtent, valid := tensor.MatrixExtents(gate.Shape)
	if !gate.Shape.Equal(up.Shape) || !valid {
		return reference.Value{}, errors.New("inference: alternate-state FFN projection shape is invalid")
	}
	result := reference.Value{Shape: gate.Shape, Data: make([]float32, len(gate.Data))}
	width := int(widthExtent)
	for token := 0; token < int(tokenExtent); token++ {
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
			degreesOfFreedom := max(width-tensor.SingletonExtent, tensor.SingletonExtent)
			cutoff = float32(mean + float64(standardDeviationMultiplier)*math.Sqrt(squared/float64(degreesOfFreedom)))
		}
		for index := range width {
			value := gate.Data[base+index]
			if sparse {
				value = max(value-cutoff, float32(tensor.FirstOffset))
			}
			result.Data[base+index] = reference.RoundedGELUTanh(value) * up.Data[base+index]
		}
	}
	return result, nil
}

func mergeAlternateStates(
	states []reference.Value,
	unembedding reference.Value,
	active int,
) (reference.Value, error) {
	_, _, projectedStates, valid := tensor.Extents3(unembedding.Shape)
	if len(states) <= tensor.SingletonExtent || !checked.NonNegativeInts(active) || active >= len(states) ||
		!valid || projectedStates != uint64(len(states)-tensor.SingletonExtent) {
		return reference.Value{}, errors.New("inference: alternate-state unembedding shape is invalid")
	}
	result := states[active].Clone()
	for index := tensor.SingletonExtent; index < len(states); index++ {
		projected, err := alternateLinearSlice(unembedding, index-tensor.SingletonExtent, states[index])
		if err != nil {
			return reference.Value{}, err
		}
		rescaleMagnitudeInPlace(&projected, states[active])
		for valueIndex := range result.Data {
			result.Data[valueIndex] += projected.Data[valueIndex]
		}
	}
	for index := range result.Data {
		result.Data[index] /= float32(len(states))
	}
	return result, nil
}

func alternateLinear(weight, input reference.Value) (reference.Value, error) {
	innerExtent, outputExtent, validWeight := tensor.MatrixExtents(weight.Shape)
	inputExtent, tokenExtent, validInput := tensor.MatrixExtents(input.Shape)
	if !validWeight || !validInput || innerExtent != inputExtent {
		return reference.Value{}, errors.New("inference: alternate-state matrix dimensions differ")
	}
	inner, outputWidth, tokens := int(innerExtent), int(outputExtent), int(tokenExtent)
	output := reference.Value{
		Shape: tensor.MustShape(uint64(outputWidth), uint64(tokens)),
		Data:  make([]float32, outputWidth*tokens),
	}
	hostmath.LinearF64(output.Data, input.Data, weight.Data, nil, tokens, inner, outputWidth)
	return output, nil
}

func alternateLinearSlice(weight reference.Value, slice int, input reference.Value) (reference.Value, error) {
	inner, output, slices, valid := tensor.Extents3(weight.Shape)
	if !valid || !checked.NonNegativeInts(slice) || uint64(slice) >= slices {
		return reference.Value{}, errors.New("inference: alternate-state grouped projection slice is invalid")
	}
	size := int(inner * output)
	start := slice * size
	return alternateLinear(reference.Value{
		Shape: tensor.MustShape(inner, output),
		Data:  weight.Data[start : start+size],
	}, input)
}

func alternateRMSNorm(
	input, weight reference.Value,
	epsilon float32,
) (reference.Value, error) {
	width, validWeight := tensor.VectorWidth(weight.Shape)
	rows, validInput := tensor.MatrixRows(input.Shape, width)
	if !validInput || !validWeight {
		return reference.Value{}, errors.New("inference: alternate-state RMS norm shape is invalid")
	}
	result := input.Clone()
	hostmath.RMSNormInto(
		result.Data, input.Data, weight.Data, int(rows), int(width), float64(epsilon),
	)
	return result, nil
}

func rescaleMagnitudeInPlace(input *reference.Value, target reference.Value) {
	widthExtent, tokenExtent, _ := tensor.MatrixExtents(input.Shape)
	width := int(widthExtent)
	for token := 0; token < int(tokenExtent); token++ {
		base := token * width
		var inputSquared, targetSquared float64
		for index := range width {
			inputValue := float64(input.Data[base+index])
			targetValue := float64(target.Data[base+index])
			inputSquared += inputValue * inputValue
			targetSquared += targetValue * targetValue
		}
		if !checked.Nonzero(inputSquared) {
			continue
		}
		scale := float32(math.Sqrt(targetSquared / inputSquared))
		for index := range width {
			input.Data[base+index] *= scale
		}
	}
}
