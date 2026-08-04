package reference

import (
	"errors"
	"math"

	"llamacpp2go/internal/tensor"
)

func ssmConv(shape tensor.Shape, input, weights Value) (Value, error) {
	window := int(input.Shape.Dims[0])
	channels := int(input.Shape.Dims[1])
	sequences := int(input.Shape.Dims[2])
	kernelSize := int(weights.Shape.Dims[0])
	tokens := window - kernelSize + 1
	if window < kernelSize || int(weights.Shape.Dims[1]) != channels {
		return Value{}, errors.New("invalid SSMConv shapes")
	}
	output := make([]float32, channels*tokens*sequences)
	for sequence := range sequences {
		for token := range tokens {
			for channel := range channels {
				var sum float32
				inputBase := sequence*window*channels + channel*window + token
				weightBase := channel * kernelSize
				for tap := range kernelSize {
					sum += input.Data[inputBase+tap] * weights.Data[weightBase+tap]
				}
				output[sequence*tokens*channels+token*channels+channel] = sum
			}
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

func ssmScan(shape tensor.Shape, inputs []Value) (Value, error) {
	if len(inputs) != 6 {
		return Value{}, errors.New("SSMScan requires six inputs")
	}
	state, x, dt, a, beta, c := inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5]
	stateWidth := int(state.Shape.Dims[0])
	dimension := int(state.Shape.Dims[1])
	heads := int(state.Shape.Dims[2])
	sequences := int(state.Shape.Dims[3])
	tokens := int(x.Shape.Dims[2])
	groups := int(beta.Shape.Dims[1])
	aWidth := int(a.Shape.Dims[0])
	if stateWidth <= 0 || dimension <= 0 || heads <= 0 || sequences <= 0 || tokens <= 0 || groups <= 0 || heads%groups != 0 {
		return Value{}, errors.New("invalid SSMScan dimensions")
	}
	attentionElements := dimension * heads * tokens * sequences
	stateElements := stateWidth * dimension * heads * sequences
	output := make([]float32, attentionElements+stateElements)
	copy(output[attentionElements:], state.Data)
	headsPerGroup := heads / groups
	for sequence := range sequences {
		for head := range heads {
			group := head / headsPerGroup
			for token := range tokens {
				delta := dt.Data[head+heads*(token+tokens*sequence)]
				absolute := math.Abs(float64(delta))
				delta = float32(math.Max(float64(delta), 0) + math.Log1p(math.Exp(-absolute)))
				for inner := range dimension {
					xIndex := inner + dimension*(head+heads*(token+tokens*sequence))
					xDelta := x.Data[xIndex] * delta
					var sum float32
					stateBase := attentionElements + stateWidth*(inner+dimension*(head+heads*sequence))
					for column := range stateWidth {
						stateIndex := stateBase + column
						factor := float32(math.Exp(float64(delta * a.Data[column%aWidth+aWidth*head])))
						bcIndex := column + stateWidth*(group+groups*(token+tokens*sequence))
						next := output[stateIndex]*factor + beta.Data[bcIndex]*xDelta
						output[stateIndex] = next
						sum += next * c.Data[bcIndex]
					}
					output[xIndex] = sum
				}
			}
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

func gatedDeltaNet(
	shape tensor.Shape,
	inputs []Value,
	attributes tensor.GatedDeltaNetAttributes,
) (Value, error) {
	if len(inputs) != 6 {
		return Value{}, errors.New("GatedDeltaNet requires six inputs")
	}
	q, k, v := inputs[0], inputs[1], inputs[2]
	gate, beta, state := inputs[3], inputs[4], inputs[5]
	size := int(v.Shape.Dims[0])
	heads := int(v.Shape.Dims[1])
	tokens := int(v.Shape.Dims[2])
	sequences := int(v.Shape.Dims[3])
	qHeads := int(q.Shape.Dims[1])
	kHeads := int(k.Shape.Dims[1])
	gateWidth := int(gate.Shape.Dims[0])
	if size <= 0 || heads <= 0 || tokens <= 0 || sequences <= 0 ||
		qHeads <= 0 || kHeads <= 0 ||
		(gateWidth != 1 && gateWidth != size) {
		return Value{}, errors.New("invalid GatedDeltaNet dimensions")
	}

	attentionElements := size * heads * tokens * sequences
	stateElements := size * size * heads * sequences
	output := make([]float32, attentionElements+stateElements)
	scale := float32(1 / math.Sqrt(float64(size)))
	for sequence := range sequences {
		for head := range heads {
			stateBase := attentionElements + (sequence*heads+head)*size*size
			inputStateBase := (sequence*heads + head) * size * size
			copy(
				output[stateBase:stateBase+size*size],
				state.Data[inputStateBase:inputStateBase+size*size],
			)
			for token := range tokens {
				valueBase := ((sequence*tokens+token)*heads + head) * size
				queryHead := head % qHeads
				keyHead := head % kHeads
				if attributes.RepeatInterleave {
					queryHead = head / (heads / qHeads)
					keyHead = head / (heads / kHeads)
				}
				queryBase := ((sequence*tokens+token)*qHeads + queryHead) * size
				keyBase := ((sequence*tokens+token)*kHeads + keyHead) * size
				gateBase := ((sequence*tokens+token)*heads + head) * gateWidth
				betaValue := beta.Data[(sequence*tokens+token)*heads+head]

				if gateWidth == 1 {
					gateScale := float32(math.Exp(float64(gate.Data[gateBase])))
					for index := range size * size {
						output[stateBase+index] *= gateScale
					}
				} else {
					for row := range size {
						for column := range size {
							gateScale := float32(math.Exp(float64(gate.Data[gateBase+column])))
							output[stateBase+row*size+column] *= gateScale
						}
					}
				}

				attentionBase := valueBase
				for row := range size {
					stateRow := stateBase + row*size
					var dot float32
					for column := range size {
						dot += output[stateRow+column] * k.Data[keyBase+column]
					}
					delta := (v.Data[valueBase+row] - dot) * betaValue
					for column := range size {
						output[stateRow+column] += delta * k.Data[keyBase+column]
					}
					dot = 0
					for column := range size {
						dot += output[stateRow+column] * q.Data[queryBase+column]
					}
					output[attentionBase+row] = dot * scale
				}
			}
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

func gatedLinearAttention(
	shape tensor.Shape,
	inputs []Value,
	attributes tensor.GatedLinearAttentionAttributes,
) (Value, error) {
	if len(inputs) != 5 {
		return Value{}, errors.New("GatedLinearAttention requires five inputs")
	}
	key, value, receptance, decay, inputState := inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
	width := int(key.Shape.Dims[0])
	keyHeads := int(key.Shape.Dims[1])
	heads := int(receptance.Shape.Dims[1])
	tokens := int(key.Shape.Dims[2])
	sequences := int(key.Shape.Dims[3])
	if width <= 0 || heads <= 0 || tokens <= 0 || sequences <= 0 {
		return Value{}, errors.New("invalid GatedLinearAttention dimensions")
	}
	attentionElements := width * heads * tokens * sequences
	stateElements := width * width
	output := make([]float32, attentionElements+stateElements*heads*sequences)
	for sequence := range sequences {
		for head := range heads {
			keyHead := head / (heads / keyHeads)
			stateBase := attentionElements + (sequence*heads+head)*stateElements
			inputStateBase := (sequence*heads + head) * stateElements
			copy(output[stateBase:stateBase+stateElements], inputState.Data[inputStateBase:inputStateBase+stateElements])
			for token := range tokens {
				vectorBase := ((sequence*tokens+token)*heads + head) * width
				keyBase := ((sequence*tokens+token)*keyHeads + keyHead) * width
				for row := range width {
					stateRow := stateBase + row*width
					var sum float32
					for column := range width {
						vectorIndex := vectorBase + column
						effectiveKey := key.Data[keyBase+column] * (1 - decay.Data[vectorIndex])
						next := output[stateRow+column]*decay.Data[vectorIndex] +
							effectiveKey*value.Data[keyBase+row]
						output[stateRow+column] = next
						sum += receptance.Data[vectorIndex] * next
					}
					output[vectorBase+row] = sum * attributes.Scale
				}
			}
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

func rwkv6(shape tensor.Shape, inputs []Value) (Value, error) {
	if len(inputs) != 6 {
		return Value{}, errors.New("RWKV6 requires six inputs")
	}
	key, value, receptance := inputs[0], inputs[1], inputs[2]
	first, decay, inputState := inputs[3], inputs[4], inputs[5]
	width := int(key.Shape.Dims[0])
	heads := int(key.Shape.Dims[1])
	tokens := int(key.Shape.Dims[2])
	sequences := int(key.Shape.Dims[3])
	if width <= 0 || heads <= 0 || tokens <= 0 || sequences <= 0 {
		return Value{}, errors.New("invalid RWKV6 dimensions")
	}
	attentionElements := width * heads * tokens * sequences
	stateElements := width * width
	output := make([]float32, attentionElements+stateElements*heads*sequences)
	for sequence := range sequences {
		for head := range heads {
			stateBase := attentionElements + (sequence*heads+head)*stateElements
			inputStateBase := (sequence*heads + head) * stateElements
			copy(output[stateBase:stateBase+stateElements], inputState.Data[inputStateBase:inputStateBase+stateElements])
			for token := range tokens {
				vectorBase := ((sequence*tokens+token)*heads + head) * width
				for row := range width {
					stateRow := stateBase + row*width
					keyValue := key.Data[vectorBase+row]
					receptanceValue := receptance.Data[vectorBase+row]
					firstValue := first.Data[head*width+row]
					decayValue := decay.Data[vectorBase+row]
					for column := range width {
						kv := keyValue * value.Data[vectorBase+column]
						previous := output[stateRow+column]
						output[vectorBase+column] += (previous + kv*firstValue) * receptanceValue
						output[stateRow+column] = previous*decayValue + kv
					}
				}
			}
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

func sumRows(shape tensor.Shape, input Value) (Value, error) {
	width := int(input.Shape.Dims[0])
	if width <= 0 || len(input.Data)%width != 0 {
		return Value{}, errors.New("invalid SumRows dimensions")
	}
	output := make([]float32, len(input.Data)/width)
	for row := range output {
		for column := range width {
			output[row] += input.Data[row*width+column]
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

func rwkv7(shape tensor.Shape, inputs []Value) (Value, error) {
	if len(inputs) != 7 {
		return Value{}, errors.New("RWKV7 requires seven inputs")
	}
	receptance, decay, key, value := inputs[0], inputs[1], inputs[2], inputs[3]
	a, bVector, inputState := inputs[4], inputs[5], inputs[6]
	width := int(key.Shape.Dims[0])
	heads := int(key.Shape.Dims[1])
	tokens := int(key.Shape.Dims[2])
	sequences := int(key.Shape.Dims[3])
	if width <= 0 || heads <= 0 || tokens <= 0 || sequences <= 0 {
		return Value{}, errors.New("invalid RWKV7 dimensions")
	}
	attentionElements := width * heads * tokens * sequences
	stateElements := width * width
	output := make([]float32, attentionElements+stateElements*heads*sequences)
	for sequence := range sequences {
		for head := range heads {
			stateBase := attentionElements + (sequence*heads+head)*stateElements
			inputStateBase := (sequence*heads + head) * stateElements
			copy(output[stateBase:stateBase+stateElements], inputState.Data[inputStateBase:inputStateBase+stateElements])
			for token := range tokens {
				vectorBase := ((sequence*tokens+token)*heads + head) * width
				for row := range width {
					stateRow := stateBase + row*width
					var stateA float32
					for column := range width {
						stateA += a.Data[vectorBase+column] * output[stateRow+column]
					}
					var result float32
					for column := range width {
						previous := output[stateRow+column]
						next := previous*decay.Data[vectorBase+column] +
							value.Data[vectorBase+row]*key.Data[vectorBase+column] +
							stateA*bVector.Data[vectorBase+column]
						output[stateRow+column] = next
						result += next * receptance.Data[vectorBase+column]
					}
					output[vectorBase+row] = result
				}
			}
		}
	}
	return Value{Shape: shape, Data: output}, nil
}
