package reference

import (
	"errors"
	"fmt"
	"math"

	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/dtype"
)

// Value: contiguous F32 reference tensor
type Value struct {
	Shape tensor.Shape
	Data  []float32
}

func NewValue(shape tensor.Shape, data []float32) (Value, error) {
	elements, err := shape.Elements()
	if err != nil {
		return Value{}, err
	}
	if elements != uint64(len(data)) {
		return Value{}, fmt.Errorf("reference data has %d elements, need %d", len(data), elements)
	}
	copied := append([]float32(nil), data...)
	return Value{Shape: shape, Data: copied}, nil
}

// Execute: evaluates outputs using feeds for input nodes
func Execute(outputs []*tensor.Tensor, feeds map[*tensor.Tensor]Value) (map[*tensor.Tensor]Value, error) {
	order, err := tensor.Topological(outputs...)
	if err != nil {
		return nil, err
	}
	values := make(map[*tensor.Tensor]Value, len(order))
	for _, node := range order {
		if node.Type != dtype.F32 {
			return nil, fmt.Errorf("reference executor does not support %s for tensor %d", node.Type, node.ID)
		}
		if node.Op == tensor.OpInput {
			value, ok := feeds[node]
			if !ok {
				return nil, fmt.Errorf("missing feed for input %q", node.Name)
			}
			if !value.Shape.Equal(node.Shape) {
				return nil, fmt.Errorf("feed shape for %q does not match graph", node.Name)
			}
			values[node] = value
			continue
		}
		inputs := make([]Value, len(node.Inputs))
		for index, input := range node.Inputs {
			value, ok := values[input]
			if !ok {
				return nil, fmt.Errorf("value for tensor %d is unavailable", input.ID)
			}
			inputs[index] = value
		}
		value, err := executeNode(node, inputs)
		if err != nil {
			return nil, fmt.Errorf("execute tensor %d (%s): %w", node.ID, node.Op, err)
		}
		values[node] = value
	}
	results := make(map[*tensor.Tensor]Value, len(outputs))
	for _, output := range outputs {
		results[output] = values[output]
	}
	return results, nil
}

func executeNode(node *tensor.Tensor, inputs []Value) (Value, error) {
	switch node.Op {
	case tensor.OpAdd:
		return elementwiseBroadcast(node.Shape, inputs[0], inputs[1], func(a, b float32) float32 { return a + b })
	case tensor.OpMultiply:
		return elementwiseBroadcast(node.Shape, inputs[0], inputs[1], func(a, b float32) float32 { return a * b })
	case tensor.OpScale:
		attributes, ok := node.Attrs.(tensor.ScaleAttributes)
		if !ok {
			return Value{}, errors.New("invalid scale attributes")
		}
		output := make([]float32, len(inputs[0].Data))
		for i, value := range inputs[0].Data {
			output[i] = value * attributes.Value
		}
		return Value{Shape: node.Shape, Data: output}, nil
	case tensor.OpClamp:
		attributes, ok := node.Attrs.(tensor.ClampAttributes)
		if !ok {
			return Value{}, errors.New("invalid clamp attributes")
		}
		output := make([]float32, len(inputs[0].Data))
		for i, value := range inputs[0].Data {
			output[i] = min(max(value, attributes.Minimum), attributes.Maximum)
		}
		return Value{Shape: node.Shape, Data: output}, nil
	case tensor.OpRMSNorm:
		attributes, ok := node.Attrs.(tensor.RMSNormAttributes)
		if !ok {
			return Value{}, errors.New("invalid RMSNorm attributes")
		}
		return rmsNorm(node.Shape, inputs[0], attributes.Epsilon)
	case tensor.OpLayerNorm:
		attributes, ok := node.Attrs.(tensor.LayerNormAttributes)
		if !ok {
			return Value{}, errors.New("invalid LayerNorm attributes")
		}
		return layerNorm(node.Shape, inputs[0], attributes.Epsilon)
	case tensor.OpSoftmax:
		return softmax(node.Shape, inputs[0])
	case tensor.OpSiLU:
		output := make([]float32, len(inputs[0].Data))
		for i, value := range inputs[0].Data {
			output[i] = value / (1 + float32(math.Exp(float64(-value))))
		}
		return Value{Shape: node.Shape, Data: output}, nil
	case tensor.OpGELU:
		output := make([]float32, len(inputs[0].Data))
		const coefficient = 0.044715
		factor := math.Sqrt(2 / math.Pi)
		for i, value := range inputs[0].Data {
			if value <= -10 {
				output[i] = 0
				continue
			}
			if value >= 10 {
				output[i] = value
				continue
			}
			x := float64(float16Round(value))
			gelu := float32(0.5 * x * (1 + math.Tanh(factor*x*(1+coefficient*x*x))))
			output[i] = float16Round(gelu)
		}
		return Value{Shape: node.Shape, Data: output}, nil
	case tensor.OpXIELU:
		attributes, ok := node.Attrs.(tensor.XIELUAttributes)
		if !ok {
			return Value{}, errors.New("invalid xIELU attributes")
		}
		output := make([]float32, len(inputs[0].Data))
		for i, value := range inputs[0].Data {
			if value > 0 {
				output[i] = attributes.AlphaP*value*value + attributes.Beta*value
				continue
			}
			minimum := float32(math.Min(float64(value), float64(attributes.Epsilon)))
			output[i] = (float32(math.Expm1(float64(minimum)))-value)*attributes.AlphaN +
				attributes.Beta*value
		}
		return Value{Shape: node.Shape, Data: output}, nil
	case tensor.OpReLUSquared:
		output := make([]float32, len(inputs[0].Data))
		for i, value := range inputs[0].Data {
			if value > 0 {
				output[i] = value * value
			}
		}
		return Value{Shape: node.Shape, Data: output}, nil
	case tensor.OpSigmoid:
		output := make([]float32, len(inputs[0].Data))
		for i, value := range inputs[0].Data {
			output[i] = 1 / (1 + float32(math.Exp(float64(-value))))
		}
		return Value{Shape: node.Shape, Data: output}, nil
	case tensor.OpSoftplus:
		output := make([]float32, len(inputs[0].Data))
		for i, value := range inputs[0].Data {
			absolute := math.Abs(float64(value))
			output[i] = float32(math.Max(float64(value), 0) + math.Log1p(math.Exp(-absolute)))
		}
		return Value{Shape: node.Shape, Data: output}, nil
	case tensor.OpTanh:
		output := make([]float32, len(inputs[0].Data))
		for i, value := range inputs[0].Data {
			output[i] = float32(math.Tanh(float64(value)))
		}
		return Value{Shape: node.Shape, Data: output}, nil
	case tensor.OpExp:
		output := make([]float32, len(inputs[0].Data))
		for i, value := range inputs[0].Data {
			output[i] = float32(math.Exp(float64(value)))
		}
		return Value{Shape: node.Shape, Data: output}, nil
	case tensor.OpL2Norm:
		attributes, ok := node.Attrs.(tensor.L2NormAttributes)
		if !ok {
			return Value{}, errors.New("invalid L2Norm attributes")
		}
		return l2Norm(node.Shape, inputs[0], attributes.Epsilon)
	case tensor.OpSSMConv:
		return ssmConv(node.Shape, inputs[0], inputs[1])
	case tensor.OpSSMScan:
		return ssmScan(node.Shape, inputs)
	case tensor.OpGatedDeltaNet:
		return gatedDeltaNet(node.Shape, inputs, node.Attrs.(tensor.GatedDeltaNetAttributes))
	case tensor.OpGatedLinearAttention:
		return gatedLinearAttention(node.Shape, inputs, node.Attrs.(tensor.GatedLinearAttentionAttributes))
	case tensor.OpRWKV6:
		return rwkv6(node.Shape, inputs)
	case tensor.OpSumRows:
		return sumRows(node.Shape, inputs[0])
	case tensor.OpRWKV7:
		return rwkv7(node.Shape, inputs)
	case tensor.OpFWHT:
		return fwht(node.Shape, inputs[0])
	case tensor.OpTopK:
		return topK(node.Shape, inputs[0], node.Attrs.(tensor.TopKAttributes))
	case tensor.OpGatherLast:
		return gatherLast(node.Shape, inputs[0], inputs[1])
	case tensor.OpSparseAttention:
		return sparseAttention(node.Shape, inputs, node.Attrs.(tensor.SparseAttentionAttributes))
	case tensor.OpIndexerScore:
		return indexerScore(node.Shape, inputs, node.Attrs.(tensor.IndexerScoreAttributes))
	case tensor.OpMoE:
		attributes, ok := node.Attrs.(tensor.MoEAttributes)
		if !ok {
			return Value{}, errors.New("invalid MoE attributes")
		}
		return moe(node.Shape, inputs, attributes)
	case tensor.OpRepeatHeads:
		attributes, ok := node.Attrs.(tensor.RepeatHeadsAttributes)
		if !ok || attributes.Heads == 0 {
			return Value{}, errors.New("invalid RepeatHeads attributes")
		}
		return repeatHeads(node.Shape, inputs[0]), nil
	case tensor.OpTranspose2D:
		return transpose2D(node.Shape, inputs[0])
	case tensor.OpGroupSlice:
		attributes, ok := node.Attrs.(tensor.GroupSliceAttributes)
		if !ok {
			return Value{}, errors.New("invalid GroupSlice attributes")
		}
		return groupSlice(node.Shape, inputs[0], attributes)
	case tensor.OpFlatSlice:
		attributes, ok := node.Attrs.(tensor.FlatSliceAttributes)
		if !ok {
			return Value{}, errors.New("invalid FlatSlice attributes")
		}
		return flatSlice(node.Shape, inputs[0], attributes)
	case tensor.OpMulMat:
		return mulMat(node.Shape, inputs[0], inputs[1])
	case tensor.OpGroupedMulMat:
		return groupedMulMat(node.Shape, inputs[0], inputs[1])
	case tensor.OpGetRows:
		attributes, ok := node.Attrs.(tensor.GetRowsAttributes)
		if !ok {
			return Value{}, errors.New("invalid get_rows attributes")
		}
		return getRows(node.Shape, inputs[0], attributes.Rows)
	case tensor.OpRoPENeoX:
		attributes, ok := node.Attrs.(tensor.RoPEAttributes)
		if !ok {
			return Value{}, errors.New("invalid rope_neox attributes")
		}
		return ropeNeoX(node.Shape, inputs, attributes)
	case tensor.OpRoPENormal:
		attributes, ok := node.Attrs.(tensor.RoPEAttributes)
		if !ok {
			return Value{}, errors.New("invalid rope_normal attributes")
		}
		return ropeNormal(node.Shape, inputs, attributes)
	case tensor.OpRoPEMulti:
		attributes, ok := node.Attrs.(tensor.RoPEMultiAttributes)
		if !ok {
			return Value{}, errors.New("invalid rope_multi attributes")
		}
		return ropeMulti(node.Shape, inputs[0], attributes)
	case tensor.OpReshape:
		return Value{Shape: node.Shape, Data: append([]float32(nil), inputs[0].Data...)}, nil
	case tensor.OpAttention:
		attributes, ok := node.Attrs.(tensor.AttentionAttributes)
		if !ok {
			return Value{}, errors.New("invalid attention attributes")
		}
		var bias, sinks *Value
		if len(inputs) == 4 {
			if attributes.HasSinks {
				sinks = &inputs[3]
			} else {
				bias = &inputs[3]
			}
		}
		return attention(node.Shape, inputs[0], inputs[1], inputs[2], bias, sinks, attributes)
	case tensor.OpConcat:
		attributes, ok := node.Attrs.(tensor.ConcatAttributes)
		if !ok {
			return Value{}, errors.New("invalid concat attributes")
		}
		return concat(node.Shape, inputs[0], inputs[1], attributes.Axis)
	default:
		return Value{}, fmt.Errorf("unsupported operation %s", node.Op)
	}
}

func float16Round(value float32) float32 {
	bits := math.Float32bits(value)
	sign := uint16(bits >> 16 & 0x8000)
	exponent := int32(bits>>23&0xff) - 127 + 15
	mantissa := bits & 0x7fffff
	var half uint16
	switch {
	case exponent <= 0:
		if exponent < -10 {
			half = sign
			break
		}
		mantissa |= 0x800000
		shift := uint32(14 - exponent)
		rounded := mantissa + (1 << (shift - 1)) - 1 + ((mantissa >> shift) & 1)
		half = sign | uint16(rounded>>shift)
	case exponent >= 31:
		if mantissa == 0 {
			half = sign | 0x7c00
		} else {
			half = sign | 0x7e00
		}
	default:
		rounded := mantissa + 0xfff + ((mantissa >> 13) & 1)
		if rounded&0x800000 != 0 {
			rounded = 0
			exponent++
		}
		if exponent >= 31 {
			half = sign | 0x7c00
		} else {
			half = sign | uint16(exponent<<10) | uint16(rounded>>13)
		}
	}
	sign32 := uint32(half&0x8000) << 16
	exp16 := uint32(half>>10) & 0x1f
	frac16 := uint32(half & 0x03ff)
	if exp16 == 0 {
		if frac16 == 0 {
			return math.Float32frombits(sign32)
		}
		exp := int32(-14)
		for frac16&0x0400 == 0 {
			frac16 <<= 1
			exp--
		}
		frac16 &= 0x03ff
		return math.Float32frombits(sign32 | uint32(exp+127)<<23 | frac16<<13)
	}
	if exp16 == 31 {
		return math.Float32frombits(sign32 | 0x7f800000 | frac16<<13)
	}
	return math.Float32frombits(sign32 | (exp16-15+127)<<23 | frac16<<13)
}

func elementwiseBroadcast(
	shape tensor.Shape,
	left, right Value,
	operation func(float32, float32) float32,
) (Value, error) {
	elements, err := shape.Elements()
	if err != nil {
		return Value{}, err
	}
	if elements > uint64(maxInt()) {
		return Value{}, errors.New("elementwise output is too large")
	}
	output := make([]float32, int(elements))
	for i := range output {
		leftIndex := broadcastIndex(uint64(i), shape, left.Shape)
		rightIndex := broadcastIndex(uint64(i), shape, right.Shape)
		if leftIndex >= uint64(len(left.Data)) || rightIndex >= uint64(len(right.Data)) {
			return Value{}, errors.New("elementwise broadcast index is out of range")
		}
		output[i] = operation(left.Data[leftIndex], right.Data[rightIndex])
	}
	return Value{Shape: shape, Data: output}, nil
}

func broadcastIndex(index uint64, outputShape, inputShape tensor.Shape) uint64 {
	var result uint64
	var stride uint64 = 1
	for axis := range tensor.MaxDimensions {
		coordinate := index % outputShape.Dims[axis]
		index /= outputShape.Dims[axis]
		if inputShape.Dims[axis] != 1 {
			result += coordinate * stride
		}
		stride *= inputShape.Dims[axis]
	}
	return result
}

func rmsNorm(shape tensor.Shape, input Value, epsilon float32) (Value, error) {
	width := int(shape.Dims[0])
	if width == 0 || len(input.Data)%width != 0 {
		return Value{}, errors.New("invalid RMSNorm row width")
	}
	output := make([]float32, len(input.Data))
	for row := 0; row < len(input.Data); row += width {
		var sumSquares float64
		for _, value := range input.Data[row : row+width] {
			sumSquares += float64(value) * float64(value)
		}
		inverse := float32(1 / math.Sqrt(sumSquares/float64(width)+float64(epsilon)))
		for column, value := range input.Data[row : row+width] {
			output[row+column] = value * inverse
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

func layerNorm(shape tensor.Shape, input Value, epsilon float32) (Value, error) {
	width := int(shape.Dims[0])
	if width == 0 || len(input.Data)%width != 0 {
		return Value{}, errors.New("invalid LayerNorm row width")
	}
	output := make([]float32, len(input.Data))
	for row := 0; row < len(input.Data); row += width {
		var sum float64
		for _, value := range input.Data[row : row+width] {
			sum += float64(value)
		}
		mean := sum / float64(width)
		var sumSquares float64
		for _, value := range input.Data[row : row+width] {
			centered := float64(value) - mean
			sumSquares += centered * centered
		}
		inverse := float32(1 / math.Sqrt(sumSquares/float64(width)+float64(epsilon)))
		for column, value := range input.Data[row : row+width] {
			output[row+column] = (value - float32(mean)) * inverse
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

func l2Norm(shape tensor.Shape, input Value, epsilon float32) (Value, error) {
	width := int(shape.Dims[0])
	if width == 0 || len(input.Data)%width != 0 {
		return Value{}, errors.New("invalid L2Norm row width")
	}
	output := make([]float32, len(input.Data))
	for row := 0; row < len(input.Data); row += width {
		var sumSquares float64
		for _, value := range input.Data[row : row+width] {
			sumSquares += float64(value) * float64(value)
		}
		denominator := math.Max(math.Sqrt(sumSquares), float64(epsilon))
		inverse := float32(1 / denominator)
		for column, value := range input.Data[row : row+width] {
			output[row+column] = value * inverse
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

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

func transpose2D(shape tensor.Shape, input Value) (Value, error) {
	width := int(input.Shape.Dims[0])
	rows := int(input.Shape.Dims[1])
	output := make([]float32, len(input.Data))
	for row := range rows {
		for column := range width {
			output[row+rows*column] = input.Data[column+width*row]
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

func groupSlice(
	shape tensor.Shape,
	input Value,
	attributes tensor.GroupSliceAttributes,
) (Value, error) {
	elements, err := shape.Elements()
	if err != nil || elements > uint64(maxInt()) {
		return Value{}, errors.New("invalid GroupSlice output size")
	}
	width := int(attributes.Width)
	groups := int(attributes.Groups)
	stride := int(attributes.Stride)
	offset := int(attributes.Offset)
	inputWidth := int(input.Shape.Dims[0])
	output := make([]float32, int(elements))
	for index := range output {
		column := index % width
		remainder := index / width
		group := remainder % groups
		outer := remainder / groups
		output[index] = input.Data[outer*inputWidth+offset+group*stride+column]
	}
	return Value{Shape: shape, Data: output}, nil
}

func flatSlice(
	shape tensor.Shape,
	input Value,
	attributes tensor.FlatSliceAttributes,
) (Value, error) {
	elements, err := shape.Elements()
	if err != nil || elements > uint64(maxInt()) ||
		attributes.Offset > uint64(len(input.Data)) ||
		elements > uint64(len(input.Data))-attributes.Offset {
		return Value{}, errors.New("invalid FlatSlice storage range")
	}
	start := int(attributes.Offset)
	return Value{
		Shape: shape,
		Data:  append([]float32(nil), input.Data[start:start+int(elements)]...),
	}, nil
}

func softmax(shape tensor.Shape, input Value) (Value, error) {
	width := int(shape.Dims[0])
	if width == 0 || len(input.Data)%width != 0 {
		return Value{}, errors.New("invalid softmax row width")
	}
	output := make([]float32, len(input.Data))
	for row := 0; row < len(input.Data); row += width {
		maximum := input.Data[row]
		for _, value := range input.Data[row+1 : row+width] {
			if value > maximum {
				maximum = value
			}
		}
		var sum float64
		for column, value := range input.Data[row : row+width] {
			exponent := math.Exp(float64(value - maximum))
			output[row+column] = float32(exponent)
			sum += exponent
		}
		for column := range width {
			output[row+column] /= float32(sum)
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

func mulMat(shape tensor.Shape, left, right Value) (Value, error) {
	k := int(left.Shape.Dims[0])
	m := int(left.Shape.Dims[1])
	n := int(right.Shape.Dims[1])
	if int(right.Shape.Dims[0]) != k {
		return Value{}, errors.New("mul_mat inner dimensions differ")
	}
	output := make([]float32, m*n)
	for row := 0; row < n; row++ {
		for column := 0; column < m; column++ {
			var sum float64
			for inner := 0; inner < k; inner++ {
				sum += float64(left.Data[column*k+inner]) * float64(right.Data[row*k+inner])
			}
			output[row*m+column] = float32(sum)
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

func groupedMulMat(shape tensor.Shape, left, right Value) (Value, error) {
	k := int(left.Shape.Dims[0])
	m := int(left.Shape.Dims[1])
	groups := int(left.Shape.Dims[2])
	n := int(right.Shape.Dims[2])
	if int(right.Shape.Dims[0]) != k || int(right.Shape.Dims[1]) != groups {
		return Value{}, errors.New("grouped_mul_mat dimensions differ")
	}
	output := make([]float32, m*groups*n)
	for token := range n {
		for group := range groups {
			for row := range m {
				var sum float64
				for inner := range k {
					leftIndex := (group*m+row)*k + inner
					rightIndex := (token*groups+group)*k + inner
					sum += float64(left.Data[leftIndex]) * float64(right.Data[rightIndex])
				}
				output[(token*groups+group)*m+row] = float32(sum)
			}
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

func getRows(shape tensor.Shape, table Value, rows []uint32) (Value, error) {
	width := int(table.Shape.Dims[0])
	if width <= 0 || len(table.Data)%width != 0 {
		return Value{}, errors.New("invalid get_rows table width")
	}
	output := make([]float32, width*len(rows))
	for outputRow, tableRow := range rows {
		if uint64(tableRow) >= table.Shape.Dims[1] {
			return Value{}, fmt.Errorf("get_rows row %d exceeds table", tableRow)
		}
		copy(
			output[outputRow*width:(outputRow+1)*width],
			table.Data[int(tableRow)*width:(int(tableRow)+1)*width],
		)
	}
	return Value{Shape: shape, Data: output}, nil
}

func ropeNeoX(shape tensor.Shape, inputs []Value, attributes tensor.RoPENeoXAttributes) (Value, error) {
	input := inputs[0]
	width := int(shape.Dims[0])
	heads := int(shape.Dims[1])
	tokens := int(shape.Dims[2])
	batches := int(shape.Dims[3])
	rotary := int(attributes.RotaryDimensions)
	if rotary <= 0 || rotary%2 != 0 || rotary > width {
		return Value{}, errors.New("invalid rope_neox rotary width")
	}
	if len(attributes.Positions) != tokens {
		return Value{}, errors.New("rope_neox position count differs from token count")
	}
	factors, err := ropeFrequencyFactors(inputs, rotary)
	if err != nil {
		return Value{}, err
	}
	output := append([]float32(nil), input.Data...)
	half := rotary / 2
	for batch := 0; batch < batches; batch++ {
		for tokenIndex, position := range attributes.Positions {
			for head := 0; head < heads; head++ {
				offset := ((batch*tokens+tokenIndex)*heads + head) * width
				for pairIndex := 0; pairIndex < half; pairIndex++ {
					cosine, sine := ropeCosSin(attributes, pairIndex, rotary, position, factors[pairIndex])
					x0 := input.Data[offset+pairIndex]
					x1 := input.Data[offset+pairIndex+half]
					output[offset+pairIndex] = x0*cosine - x1*sine
					output[offset+pairIndex+half] = x0*sine + x1*cosine
				}
			}
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

func ropeNormal(shape tensor.Shape, inputs []Value, attributes tensor.RoPEAttributes) (Value, error) {
	input := inputs[0]
	width := int(shape.Dims[0])
	heads := int(shape.Dims[1])
	tokens := int(shape.Dims[2])
	batches := int(shape.Dims[3])
	rotary := int(attributes.RotaryDimensions)
	if rotary <= 0 || rotary%2 != 0 || rotary > width {
		return Value{}, errors.New("invalid rope_normal rotary width")
	}
	if len(attributes.Positions) != tokens {
		return Value{}, errors.New("rope_normal position count differs from token count")
	}
	factors, err := ropeFrequencyFactors(inputs, rotary)
	if err != nil {
		return Value{}, err
	}
	output := append([]float32(nil), input.Data...)
	for batch := 0; batch < batches; batch++ {
		for tokenIndex, position := range attributes.Positions {
			for head := 0; head < heads; head++ {
				offset := ((batch*tokens+tokenIndex)*heads + head) * width
				for pairIndex := 0; pairIndex < rotary/2; pairIndex++ {
					cosine, sine := ropeCosSin(attributes, pairIndex, rotary, position, factors[pairIndex])
					first := offset + pairIndex*2
					x0 := input.Data[first]
					x1 := input.Data[first+1]
					output[first] = x0*cosine - x1*sine
					output[first+1] = x0*sine + x1*cosine
				}
			}
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

func ropeCosSin(
	attributes tensor.RoPEAttributes,
	pairIndex, rotary int,
	position uint32,
	factor float32,
) (float32, float32) {
	thetaExtrapolated := float64(position) * math.Pow(
		float64(attributes.FrequencyBase), -2*float64(pairIndex)/float64(rotary),
	) / float64(factor)
	theta := float64(attributes.FrequencyScale) * thetaExtrapolated
	magnitude := float64(attributes.AttentionFactor)
	if magnitude == 0 {
		magnitude = 1
	}
	if attributes.ExtFactor != 0 && attributes.OriginalContext > 0 {
		correction := func(rotations float32) float64 {
			return float64(rotary) * math.Log(
				float64(attributes.OriginalContext)/(float64(rotations)*2*math.Pi),
			) / (2 * math.Log(float64(attributes.FrequencyBase)))
		}
		low := math.Floor(correction(attributes.BetaFast))
		high := math.Ceil(correction(attributes.BetaSlow))
		low = math.Max(0, math.Min(float64(rotary-1), low))
		high = math.Max(0, math.Min(float64(rotary-1), high))
		ramp := 1 - math.Min(1, math.Max(0, (float64(pairIndex)-low)/math.Max(0.001, high-low)))
		mix := ramp * float64(attributes.ExtFactor)
		theta = theta*(1-mix) + thetaExtrapolated*mix
		magnitude *= 1 + 0.1*math.Log(1/float64(attributes.FrequencyScale))
	}
	return float32(math.Cos(theta) * magnitude), float32(math.Sin(theta) * magnitude)
}

func ropeFrequencyFactors(inputs []Value, rotary int) ([]float32, error) {
	factors := make([]float32, rotary/2)
	for index := range factors {
		factors[index] = 1
	}
	if len(inputs) == 1 {
		return factors, nil
	}
	if len(inputs) != 2 ||
		inputs[1].Shape.Rank != 1 ||
		inputs[1].Shape.Dims[0] != uint64(len(factors)) ||
		len(inputs[1].Data) != len(factors) {
		return nil, errors.New("RoPE frequency factors have invalid shape")
	}
	for index, factor := range inputs[1].Data {
		if factor <= 0 || math.IsNaN(float64(factor)) || math.IsInf(float64(factor), 0) {
			return nil, errors.New("RoPE frequency factor must be finite and positive")
		}
		factors[index] = factor
	}
	return factors, nil
}

func ropeMulti(
	shape tensor.Shape,
	input Value,
	attributes tensor.RoPEMultiAttributes,
) (Value, error) {
	width := int(shape.Dims[0])
	heads := int(shape.Dims[1])
	tokens := int(shape.Dims[2])
	batches := int(shape.Dims[3])
	rotary := int(attributes.RotaryDimensions)
	half := rotary / 2
	var sectionPairs int
	for _, section := range attributes.Sections {
		sectionPairs += int(section)
	}
	if rotary <= 0 || rotary%2 != 0 || rotary > width ||
		sectionPairs <= 0 || sectionPairs > half {
		return Value{}, errors.New("invalid rope_multi dimensions")
	}
	output := append([]float32(nil), input.Data...)
	for batch := range batches {
		for token := range tokens {
			for head := range heads {
				offset := ((batch*tokens+token)*heads + head) * width
				for pair := range half {
					sector := pair % sectionPairs
					axis := 0
					boundary := int(attributes.Sections[0])
					for axis < 3 && sector >= boundary {
						axis++
						boundary += int(attributes.Sections[axis])
					}
					theta := float64(attributes.Positions[axis][token]) * float64(attributes.FrequencyScale) * math.Pow(
						float64(attributes.FrequencyBase),
						-2*float64(pair)/float64(rotary),
					)
					cosine := float32(math.Cos(theta))
					sine := float32(math.Sin(theta))
					x0 := input.Data[offset+pair]
					x1 := input.Data[offset+pair+half]
					output[offset+pair] = x0*cosine - x1*sine
					output[offset+pair+half] = x0*sine + x1*cosine
				}
			}
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

func repeatHeads(shape tensor.Shape, input Value) Value {
	width := int(input.Shape.Dims[0])
	tokens := int(input.Shape.Dims[2])
	heads := int(shape.Dims[1])
	output := make([]float32, width*heads*tokens)
	for token := 0; token < tokens; token++ {
		source := input.Data[token*width : (token+1)*width]
		for head := 0; head < heads; head++ {
			destination := (token*heads + head) * width
			copy(output[destination:destination+width], source)
		}
	}
	return Value{Shape: shape, Data: output}
}

func moe(shape tensor.Shape, inputs []Value, attributes tensor.MoEAttributes) (Value, error) {
	wantInputs := 5
	if attributes.Gated && !attributes.FusedGateUp {
		wantInputs = 6
	}
	expectedInputs := wantInputs
	if attributes.HasSelectionBias {
		expectedInputs++
	}
	if attributes.HasExpertScale {
		expectedInputs++
	}
	if attributes.HasRouterBias {
		expectedInputs++
	}
	if attributes.HasExpertBiases {
		expectedInputs += 3
	}
	if len(inputs) != expectedInputs {
		return Value{}, errors.New("MoE input count is invalid")
	}
	input, routerInput, router := inputs[0], inputs[1], inputs[2]
	var gate Value
	next := 3
	if attributes.Gated && !attributes.FusedGateUp {
		gate = inputs[next]
		next++
	}
	up, down := inputs[next], inputs[next+1]
	if attributes.FusedGateUp {
		gate = up
	}
	var selectionBias, expertScale, routerBias, gateBias, upBias, downBias []float32
	optionalIndex := next + 2
	if attributes.HasSelectionBias {
		selectionBias = inputs[optionalIndex].Data
		optionalIndex++
	}
	if attributes.HasExpertScale {
		expertScale = inputs[optionalIndex].Data
		optionalIndex++
	}
	if attributes.HasRouterBias {
		routerBias = inputs[optionalIndex].Data
		optionalIndex++
	}
	if attributes.HasExpertBiases {
		gateBias = inputs[optionalIndex].Data
		upBias = inputs[optionalIndex+1].Data
		downBias = inputs[optionalIndex+2].Data
	}
	hidden := int(input.Shape.Dims[0])
	routerHidden := int(routerInput.Shape.Dims[0])
	tokens := int(input.Shape.Dims[1])
	experts := int(attributes.Experts)
	expertIndexDivisor := int(attributes.ExpertIndexDivisor)
	if expertIndexDivisor == 0 {
		expertIndexDivisor = 1
	}
	topK := int(attributes.TopK)
	intermediate := int(up.Shape.Dims[1])
	if attributes.FusedGateUp {
		intermediate /= 2
	}
	bankExperts := int(up.Shape.Dims[2])
	if hidden <= 0 || tokens <= 0 || experts <= 0 || topK <= 0 || topK > experts ||
		experts%expertIndexDivisor != 0 || experts/expertIndexDivisor != bankExperts ||
		topK > bankExperts || int(down.Shape.Dims[2]) != bankExperts {
		return Value{}, errors.New("invalid MoE dimensions")
	}
	output := make([]float32, hidden*tokens)
	logits := make([]float64, experts)
	probabilities := make([]float64, experts)
	selectionScores := make([]float64, experts)
	selected := make([]int, topK)
	weights := make([]float64, topK)
	used := make([]bool, experts)
	for token := 0; token < tokens; token++ {
		x := input.Data[token*hidden : (token+1)*hidden]
		routerX := routerInput.Data[token*routerHidden : (token+1)*routerHidden]
		maximum := math.Inf(-1)
		for expert := 0; expert < experts; expert++ {
			var dot float64
			for channel := 0; channel < routerHidden; channel++ {
				dot += float64(routerX[channel]) * float64(router.Data[expert*routerHidden+channel])
			}
			if routerBias != nil {
				dot += float64(routerBias[expert])
			}
			logits[expert] = dot
			maximum = math.Max(maximum, dot)
		}
		var denominator float64
		if attributes.Routing == tensor.MoERoutingSoftmax {
			for _, logit := range logits {
				denominator += math.Exp(logit - maximum)
			}
		}
		for expert, logit := range logits {
			switch attributes.Routing {
			case tensor.MoERoutingSoftmax:
				probabilities[expert] = math.Exp(logit-maximum) / denominator
			case tensor.MoERoutingSigmoid:
				probabilities[expert] = 1 / (1 + math.Exp(-logit))
			case tensor.MoERoutingSelectedSoftmax:
				probabilities[expert] = logit
			default:
				return Value{}, errors.New("invalid MoE routing function")
			}
			selectionScores[expert] = probabilities[expert]
			if selectionBias != nil {
				selectionScores[expert] += float64(selectionBias[expert])
			}
		}
		clear(used)
		var selectedSum float64
		for slot := 0; slot < topK; slot++ {
			best := -1
			bestScore := math.Inf(-1)
			for expert, score := range selectionScores {
				if !used[expert] && (best < 0 || score > bestScore) {
					best, bestScore = expert, score
				}
			}
			used[best] = true
			selected[slot] = best
			weights[slot] = probabilities[best]
			selectedSum += weights[slot]
		}
		if attributes.Routing == tensor.MoERoutingSelectedSoftmax {
			selectedMaximum := math.Inf(-1)
			for _, weight := range weights {
				selectedMaximum = math.Max(selectedMaximum, weight)
			}
			selectedSum = 0
			for slot, weight := range weights {
				weights[slot] = math.Exp(weight - selectedMaximum)
				selectedSum += weights[slot]
			}
			for slot := range weights {
				weights[slot] /= selectedSum
			}
		}
		for slot := range topK {
			if attributes.NormalizeTopKProb && selectedSum > 0 {
				weights[slot] = weights[slot] / selectedSum * float64(attributes.Scale)
			} else {
				weights[slot] *= float64(attributes.Scale)
			}
		}
		for outputChannel := 0; outputChannel < hidden; outputChannel++ {
			var routed float64
			for slot, routedExpert := range selected {
				expert := routedExpert / expertIndexDivisor
				var expertOutput float64
				for inner := 0; inner < intermediate; inner++ {
					gateBase := (expert*intermediate + inner) * hidden
					upBase := gateBase
					if attributes.FusedGateUp {
						gateBase = (expert*2*intermediate + inner) * hidden
						upBase = gateBase + intermediate*hidden
					}
					var gateDot, upDot float64
					for channel := 0; channel < hidden; channel++ {
						if attributes.Gated {
							gateDot += float64(x[channel]) * float64(gate.Data[gateBase+channel])
						}
						upDot += float64(x[channel]) * float64(up.Data[upBase+channel])
					}
					if gateBias != nil {
						gateDot += float64(gateBias[expert*intermediate+inner])
						upDot += float64(upBias[expert*intermediate+inner])
					}
					var activation float64
					switch attributes.Activation {
					case tensor.MoEActivationSiLU:
						activation = upDot / (1 + math.Exp(-upDot))
						if attributes.Gated {
							gateActivation := gateDot / (1 + math.Exp(-gateDot))
							if attributes.SwiGLUClamp > 0 {
								limit := float64(attributes.SwiGLUClamp)
								upDot = math.Max(-limit, math.Min(limit, upDot))
								gateActivation = math.Min(limit, gateActivation)
							}
							activation = gateActivation * upDot
						}
					case tensor.MoEActivationReLU:
						activation = math.Max(upDot, 0)
						if attributes.Gated {
							activation = math.Max(gateDot, 0) * upDot
						}
					case tensor.MoEActivationReLUSquared:
						activation = math.Max(upDot, 0)
						activation *= activation
					case tensor.MoEActivationGELU:
						activation = moeGELU(upDot)
						if attributes.Gated {
							activation = moeGELU(gateDot) * upDot
						}
					case tensor.MoEActivationSwiGLUOAI:
						x := math.Min(gateDot, 7)
						y := math.Max(-7, math.Min(7, upDot))
						activation = x / (1 + math.Exp(-1.702*x)) * (y + 1)
					default:
						return Value{}, errors.New("invalid MoE activation")
					}
					downIndex := (expert*hidden+outputChannel)*intermediate + inner
					expertOutput += activation * float64(down.Data[downIndex])
				}
				if downBias != nil {
					expertOutput += float64(downBias[expert*hidden+outputChannel])
				}
				if expertScale != nil {
					expertOutput *= float64(expertScale[expert])
				}
				routed += weights[slot] * expertOutput
			}
			output[token*hidden+outputChannel] = float32(routed)
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

func moeGELU(value float64) float64 {
	if value <= -10 {
		return 0
	}
	if value >= 10 {
		return value
	}
	x := float64(float16Round(float32(value)))
	result := float32(0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*x*(1+0.044715*x*x))))
	return float64(float16Round(result))
}

func fwht(shape tensor.Shape, input Value) (Value, error) {
	width := int(shape.Dims[0])
	if width == 0 || width&(width-1) != 0 || len(input.Data)%width != 0 {
		return Value{}, errors.New("invalid FWHT dimensions")
	}
	output := append([]float32(nil), input.Data...)
	for row := 0; row < len(output); row += width {
		for stride := 1; stride < width; stride *= 2 {
			for base := 0; base < width; base += 2 * stride {
				for offset := 0; offset < stride; offset++ {
					first := row + base + offset
					second := first + stride
					a, b := output[first], output[second]
					output[first], output[second] = a+b, a-b
				}
			}
		}
	}
	scale := float32(1 / math.Sqrt(float64(width)))
	for index := range output {
		output[index] *= scale
	}
	return Value{Shape: shape, Data: output}, nil
}

func topK(
	shape tensor.Shape,
	input Value,
	attributes tensor.TopKAttributes,
) (Value, error) {
	width := int(input.Shape.Dims[0])
	k := int(attributes.K)
	if width == 0 || k <= 0 || k > width || len(input.Data)%width != 0 {
		return Value{}, errors.New("invalid TopK dimensions")
	}
	rows := len(input.Data) / width
	output := make([]float32, rows*k)
	for row := range rows {
		selected := output[row*k : (row+1)*k]
		for index := range selected {
			selected[index] = -1
		}
		for candidate := range width {
			candidateValue := input.Data[row*width+candidate]
			insert := k
			for slot := range k {
				selectedIndex := int(selected[slot])
				if selectedIndex < 0 || topKBefore(
					candidateValue, candidate,
					input.Data[row*width+selectedIndex], selectedIndex,
				) {
					insert = slot
					break
				}
			}
			if insert == k {
				continue
			}
			copy(selected[insert+1:], selected[insert:k-1])
			selected[insert] = float32(candidate)
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

func topKBefore(left float32, leftIndex int, right float32, rightIndex int) bool {
	leftNaN, rightNaN := math.IsNaN(float64(left)), math.IsNaN(float64(right))
	if leftNaN != rightNaN {
		return !leftNaN
	}
	if left == right || leftNaN {
		return leftIndex < rightIndex
	}
	return left > right
}

func gatherLast(shape tensor.Shape, input, indices Value) (Value, error) {
	last := int(input.Shape.Rank) - 1
	rows := int(input.Shape.Dims[last])
	inner := 1
	for _, dimension := range input.Shape.Slice()[:last] {
		inner *= int(dimension)
	}
	if rows <= 0 || inner <= 0 || len(input.Data) != rows*inner {
		return Value{}, errors.New("invalid GatherLast input dimensions")
	}
	output := make([]float32, inner*len(indices.Data))
	for indexPosition, raw := range indices.Data {
		row, err := exactTensorIndex(raw, rows)
		if err != nil {
			return Value{}, fmt.Errorf("GatherLast index %d: %w", indexPosition, err)
		}
		copy(output[indexPosition*inner:(indexPosition+1)*inner], input.Data[row*inner:(row+1)*inner])
	}
	return Value{Shape: shape, Data: output}, nil
}

func sparseAttention(
	shape tensor.Shape,
	inputs []Value,
	attributes tensor.SparseAttentionAttributes,
) (Value, error) {
	query, key, value, indices := inputs[0], inputs[1], inputs[2], inputs[3]
	keyWidth := int(query.Shape.Dims[0])
	valueWidth := int(value.Shape.Dims[0])
	queryHeads := int(query.Shape.Dims[1])
	keyValueHeads := int(key.Shape.Dims[1])
	queryTokens := int(query.Shape.Dims[2])
	keyValueTokens := int(key.Shape.Dims[2])
	selected := int(indices.Shape.Dims[0])
	if keyWidth <= 0 || valueWidth <= 0 || queryHeads <= 0 || keyValueHeads <= 0 ||
		queryTokens <= 0 || keyValueTokens <= 0 || selected <= 0 || queryHeads%keyValueHeads != 0 {
		return Value{}, errors.New("invalid SparseAttention dimensions")
	}
	indexRows := make([]int, len(indices.Data))
	for index, raw := range indices.Data {
		row, err := exactTensorIndex(raw, keyValueTokens)
		if err != nil {
			return Value{}, fmt.Errorf("SparseAttention index %d: %w", index, err)
		}
		indexRows[index] = row
	}
	output := make([]float32, valueWidth*queryHeads*queryTokens)
	groupSize := queryHeads / keyValueHeads
	for queryToken := range queryTokens {
		selectedTokens := make([]int, 0, selected)
		for slot := range selected {
			keyToken := indexRows[queryToken*selected+slot]
			if attributes.Causal && keyToken > int(attributes.QueryStart)+queryToken {
				continue
			}
			selectedTokens = append(selectedTokens, keyToken)
		}
		if len(selectedTokens) == 0 {
			return Value{}, fmt.Errorf("SparseAttention query token %d has no valid keys", queryToken)
		}
		scores := make([]float64, len(selectedTokens))
		for queryHead := range queryHeads {
			keyValueHead := queryHead / groupSize
			queryOffset := (queryToken*queryHeads + queryHead) * keyWidth
			maximum := math.Inf(-1)
			for slot, keyToken := range selectedTokens {
				keyOffset := (keyToken*keyValueHeads + keyValueHead) * keyWidth
				var dot float64
				for channel := range keyWidth {
					dot += float64(query.Data[queryOffset+channel]) * float64(key.Data[keyOffset+channel])
				}
				scores[slot] = dot * float64(attributes.Scale)
				maximum = max(maximum, scores[slot])
			}
			var sum float64
			for slot := range selectedTokens {
				scores[slot] = math.Exp(scores[slot] - maximum)
				sum += scores[slot]
			}
			outputOffset := (queryToken*queryHeads + queryHead) * valueWidth
			for channel := range valueWidth {
				var weighted float64
				for slot, keyToken := range selectedTokens {
					valueOffset := (keyToken*keyValueHeads + keyValueHead) * valueWidth
					weighted += scores[slot] * float64(value.Data[valueOffset+channel])
				}
				output[outputOffset+channel] = float32(weighted / sum)
			}
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

func exactTensorIndex(value float32, limit int) (int, error) {
	if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || value < 0 || value >= float32(limit) {
		return 0, errors.New("index is outside tensor")
	}
	index := int(value)
	if float32(index) != value {
		return 0, errors.New("index is not an exact integer")
	}
	return index, nil
}

func indexerScore(
	shape tensor.Shape,
	inputs []Value,
	attributes tensor.IndexerScoreAttributes,
) (Value, error) {
	query, key, weights := inputs[0], inputs[1], inputs[2]
	width := int(query.Shape.Dims[0])
	heads := int(query.Shape.Dims[1])
	queryTokens := int(query.Shape.Dims[2])
	keyTokens := int(key.Shape.Dims[2])
	if width <= 0 || heads <= 0 || queryTokens <= 0 || keyTokens <= 0 {
		return Value{}, errors.New("invalid IndexerScore dimensions")
	}
	output := make([]float32, keyTokens*queryTokens)
	for queryToken := range queryTokens {
		for keyToken := range keyTokens {
			if keyToken > int(attributes.QueryStart)+queryToken {
				output[queryToken*keyTokens+keyToken] = float32(math.Inf(-1))
				continue
			}
			var score float64
			for head := range heads {
				queryOffset := (queryToken*heads + head) * width
				keyOffset := keyToken * width
				var dot float64
				for channel := range width {
					dot += float64(query.Data[queryOffset+channel]) * float64(key.Data[keyOffset+channel])
				}
				if dot > 0 {
					score += dot * float64(weights.Data[queryToken*heads+head])
				}
			}
			output[queryToken*keyTokens+keyToken] = float32(score * float64(attributes.Scale))
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

func attention(
	shape tensor.Shape,
	query, key, value Value,
	bias, sinks *Value,
	attributes tensor.AttentionAttributes,
) (Value, error) {
	keyWidth := int(query.Shape.Dims[0])
	valueWidth := int(value.Shape.Dims[0])
	queryHeads := int(query.Shape.Dims[1])
	keyValueHeads := int(key.Shape.Dims[1])
	queryTokens := int(query.Shape.Dims[2])
	keyValueTokens := int(key.Shape.Dims[2])
	if keyWidth <= 0 || valueWidth <= 0 || queryHeads <= 0 ||
		keyValueHeads <= 0 || queryTokens <= 0 || keyValueTokens <= 0 {
		return Value{}, errors.New("invalid attention dimensions")
	}
	if queryHeads%keyValueHeads != 0 {
		return Value{}, errors.New("attention query heads are not divisible by KV heads")
	}
	if uint64(attributes.QueryStart)+uint64(queryTokens) > uint64(keyValueTokens) {
		return Value{}, errors.New("attention query range exceeds KV tokens")
	}
	output := make([]float32, valueWidth*queryHeads*queryTokens)
	groupSize := queryHeads / keyValueHeads
	nHeadLog2 := 1
	for nHeadLog2*2 <= queryHeads {
		nHeadLog2 *= 2
	}
	m0 := math.Pow(2, -float64(attributes.MaxALiBiBias)/float64(nHeadLog2))
	m1 := math.Pow(2, -float64(attributes.MaxALiBiBias/2)/float64(nHeadLog2))
	scores := make([]float64, keyValueTokens)
	for queryToken := 0; queryToken < queryTokens; queryToken++ {
		keyLimit := keyValueTokens
		if attributes.Causal {
			keyLimit = int(attributes.QueryStart) + queryToken + 1
		}
		keyFirst := 0
		if attributes.ChunkedWindow {
			queryPosition := int(attributes.QueryStart) + queryToken
			keyFirst = queryPosition / int(attributes.Window) * int(attributes.Window)
		} else if attributes.SymmetricWindow {
			halfWindow := int(attributes.Window / 2)
			queryPosition := int(attributes.QueryStart) + queryToken
			keyFirst = max(0, queryPosition-halfWindow)
			keyLimit = min(keyValueTokens, queryPosition+halfWindow+1)
		} else if attributes.Window > 0 && keyLimit > int(attributes.Window) {
			keyFirst = keyLimit - int(attributes.Window)
		}
		for queryHead := 0; queryHead < queryHeads; queryHead++ {
			keyValueHead := queryHead / groupSize
			alibiSlope := 0.0
			if attributes.MaxALiBiBias > 0 {
				if queryHead < nHeadLog2 {
					alibiSlope = math.Pow(m0, float64(queryHead+1))
				} else {
					alibiSlope = math.Pow(m1, float64(2*(queryHead-nHeadLog2)+1))
				}
			}
			maximum := math.Inf(-1)
			if sinks != nil {
				maximum = float64(sinks.Data[queryHead])
			}
			queryOffset := (queryToken*queryHeads + queryHead) * keyWidth
			for keyToken := keyFirst; keyToken < keyLimit; keyToken++ {
				keyOffset := (keyToken*keyValueHeads + keyValueHead) * keyWidth
				var dot float64
				for channel := 0; channel < keyWidth; channel++ {
					dot += float64(query.Data[queryOffset+channel]) * float64(key.Data[keyOffset+channel])
				}
				score := dot * float64(attributes.Scale)
				if alibiSlope != 0 {
					queryPosition := int(attributes.QueryStart) + queryToken
					score -= math.Abs(float64(queryPosition-keyToken)) * alibiSlope
				}
				if bias != nil {
					bucket := relativePositionBucket(
						queryToken,
						keyToken,
						int(attributes.RelativeBuckets),
					)
					score += float64(bias.Data[bucket*queryHeads+queryHead])
				}
				if attributes.Softcap > 0 {
					cap := float64(attributes.Softcap)
					score = cap * math.Tanh(score/cap)
				}
				scores[keyToken] = score
				if score > maximum {
					maximum = score
				}
			}
			var sum float64
			if sinks != nil {
				sum = math.Exp(float64(sinks.Data[queryHead]) - maximum)
			}
			for keyToken := keyFirst; keyToken < keyLimit; keyToken++ {
				probability := math.Exp(scores[keyToken] - maximum)
				scores[keyToken] = probability
				sum += probability
			}
			outputOffset := (queryToken*queryHeads + queryHead) * valueWidth
			for channel := 0; channel < valueWidth; channel++ {
				var weighted float64
				for keyToken := keyFirst; keyToken < keyLimit; keyToken++ {
					valueOffset := (keyToken*keyValueHeads + keyValueHead) * valueWidth
					weighted += scores[keyToken] * float64(value.Data[valueOffset+channel])
				}
				output[outputOffset+channel] = float32(weighted / sum)
			}
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

func relativePositionBucket(query, key, buckets int) int {
	half := buckets / 2
	maxExact := half / 2
	relative := key - query
	bucket := 0
	if relative > 0 {
		bucket += half
	}
	if relative < 0 {
		relative = -relative
	}
	if relative < maxExact {
		return bucket + relative
	}
	large := maxExact + int(math.Floor(
		math.Log(float64(relative)/float64(maxExact))*
			float64(half-maxExact)/math.Log(128.0/float64(maxExact)),
	))
	if large >= half {
		large = half - 1
	}
	return bucket + large
}

func concat(shape tensor.Shape, left, right Value, axis uint32) (Value, error) {
	leftElements, err := left.Shape.Elements()
	if err != nil {
		return Value{}, err
	}
	rightElements, err := right.Shape.Elements()
	if err != nil {
		return Value{}, err
	}
	if leftElements > uint64(len(left.Data)) || rightElements > uint64(len(right.Data)) {
		return Value{}, errors.New("invalid concat input storage")
	}
	if axis == 0 {
		outputElements, shapeErr := shape.Elements()
		if shapeErr != nil || outputElements > uint64(maxInt()) {
			return Value{}, errors.New("invalid concat output size")
		}
		leftWidth := int(left.Shape.Dims[0])
		rightWidth := int(right.Shape.Dims[0])
		outputWidth := leftWidth + rightWidth
		output := make([]float32, int(outputElements))
		for outer := 0; outer < len(output)/outputWidth; outer++ {
			copy(
				output[outer*outputWidth:outer*outputWidth+leftWidth],
				left.Data[outer*leftWidth:(outer+1)*leftWidth],
			)
			copy(
				output[outer*outputWidth+leftWidth:(outer+1)*outputWidth],
				right.Data[outer*rightWidth:(outer+1)*rightWidth],
			)
		}
		return Value{Shape: shape, Data: output}, nil
	}
	if axis != uint32(left.Shape.Rank-1) {
		return Value{}, errors.New("unsupported concat axis")
	}
	output := make([]float32, 0, len(left.Data)+len(right.Data))
	output = append(output, left.Data...)
	output = append(output, right.Data...)
	return Value{Shape: shape, Data: output}, nil
}

func maxInt() int {
	return int(^uint(0) >> 1)
}
