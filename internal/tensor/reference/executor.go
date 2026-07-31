package reference

import (
	"errors"
	"fmt"
	"math"

	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/dtype"
)

// Value is a contiguous F32 reference tensor.
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

// Execute evaluates outputs using feeds for input nodes.
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
	case tensor.OpRMSNorm:
		attributes, ok := node.Attrs.(tensor.RMSNormAttributes)
		if !ok {
			return Value{}, errors.New("invalid RMSNorm attributes")
		}
		return rmsNorm(node.Shape, inputs[0], attributes.Epsilon)
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
	case tensor.OpL2Norm:
		attributes, ok := node.Attrs.(tensor.L2NormAttributes)
		if !ok {
			return Value{}, errors.New("invalid L2Norm attributes")
		}
		return l2Norm(node.Shape, inputs[0], attributes.Epsilon)
	case tensor.OpSSMConv:
		return ssmConv(node.Shape, inputs[0], inputs[1])
	case tensor.OpGatedDeltaNet:
		return gatedDeltaNet(node.Shape, inputs)
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
		return ropeNeoX(node.Shape, inputs[0], attributes)
	case tensor.OpRoPENormal:
		attributes, ok := node.Attrs.(tensor.RoPEAttributes)
		if !ok {
			return Value{}, errors.New("invalid rope_normal attributes")
		}
		return ropeNormal(node.Shape, inputs[0], attributes)
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
		var bias *Value
		if len(inputs) == 4 {
			bias = &inputs[3]
		}
		return attention(node.Shape, inputs[0], inputs[1], inputs[2], bias, attributes)
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

func gatedDeltaNet(shape tensor.Shape, inputs []Value) (Value, error) {
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
				queryBase := ((sequence*tokens+token)*qHeads + head%qHeads) * size
				keyBase := ((sequence*tokens+token)*kHeads + head%kHeads) * size
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

func ropeNeoX(shape tensor.Shape, input Value, attributes tensor.RoPENeoXAttributes) (Value, error) {
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
	output := append([]float32(nil), input.Data...)
	half := rotary / 2
	for batch := 0; batch < batches; batch++ {
		for tokenIndex, position := range attributes.Positions {
			for head := 0; head < heads; head++ {
				offset := ((batch*tokens+tokenIndex)*heads + head) * width
				for pairIndex := 0; pairIndex < half; pairIndex++ {
					theta := float64(position) * float64(attributes.FrequencyScale) * math.Pow(
						float64(attributes.FrequencyBase),
						-2*float64(pairIndex)/float64(rotary),
					)
					cosine := float32(math.Cos(theta))
					sine := float32(math.Sin(theta))
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

func ropeNormal(shape tensor.Shape, input Value, attributes tensor.RoPEAttributes) (Value, error) {
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
	output := append([]float32(nil), input.Data...)
	for batch := 0; batch < batches; batch++ {
		for tokenIndex, position := range attributes.Positions {
			for head := 0; head < heads; head++ {
				offset := ((batch*tokens+tokenIndex)*heads + head) * width
				for pairIndex := 0; pairIndex < rotary/2; pairIndex++ {
					theta := float64(position) * float64(attributes.FrequencyScale) * math.Pow(
						float64(attributes.FrequencyBase),
						-2*float64(pairIndex)/float64(rotary),
					)
					cosine := float32(math.Cos(theta))
					sine := float32(math.Sin(theta))
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
					theta := float64(attributes.Positions[axis][token]) * math.Pow(
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

func attention(
	shape tensor.Shape,
	query, key, value Value,
	bias *Value,
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
	scores := make([]float64, keyValueTokens)
	for queryToken := 0; queryToken < queryTokens; queryToken++ {
		keyLimit := keyValueTokens
		if attributes.Causal {
			keyLimit = int(attributes.QueryStart) + queryToken + 1
		}
		keyFirst := 0
		if attributes.Window > 0 && keyLimit > int(attributes.Window) {
			keyFirst = keyLimit - int(attributes.Window)
		}
		for queryHead := 0; queryHead < queryHeads; queryHead++ {
			keyValueHead := queryHead / groupSize
			maximum := math.Inf(-1)
			queryOffset := (queryToken*queryHeads + queryHead) * keyWidth
			for keyToken := keyFirst; keyToken < keyLimit; keyToken++ {
				keyOffset := (keyToken*keyValueHeads + keyValueHead) * keyWidth
				var dot float64
				for channel := 0; channel < keyWidth; channel++ {
					dot += float64(query.Data[queryOffset+channel]) * float64(key.Data[keyOffset+channel])
				}
				score := dot * float64(attributes.Scale)
				if bias != nil {
					bucket := relativePositionBucket(
						queryToken,
						keyToken,
						int(attributes.RelativeBuckets),
					)
					score += float64(bias.Data[bucket*queryHeads+queryHead])
				}
				scores[keyToken] = score
				if score > maximum {
					maximum = score
				}
			}
			var sum float64
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
	if axis != 2 {
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
