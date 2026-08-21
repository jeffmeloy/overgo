package reference

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"overgo/internal/hostmath"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// Value: contiguous F32 reference tensor
type Value struct {
	Shape   tensor.Shape
	Data    []float32
	Storage ValueStorage
}

// ValueStorage: feed backing policy.
type ValueStorage uint8

const (
	ValueMaterialized ValueStorage = iota
	ValueImplicitZero
)

// ZeroValue: shape-only zero feed.
func ZeroValue(shape tensor.Shape) Value {
	return Value{Shape: shape, Storage: ValueImplicitZero}
}

func NewValue(shape tensor.Shape, data []float32) (Value, error) {
	elements, err := shape.Elements()
	if err != nil {
		return Value{}, err
	}
	if elements != uint64(len(data)) {
		return Value{}, fmt.Errorf("reference data has %d elements, need %d", len(data), elements)
	}
	copied := slices.Clone(data)
	return Value{Shape: shape, Data: copied}, nil
}

func (v Value) validate() (uint64, error) {
	elements, err := v.Shape.Elements()
	if err != nil {
		return 0, err
	}
	switch v.Storage {
	case ValueMaterialized:
		if elements != uint64(len(v.Data)) {
			return 0, fmt.Errorf("reference data has %d elements, need %d", len(v.Data), elements)
		}
	case ValueImplicitZero:
		if len(v.Data) != 0 {
			return 0, errors.New("reference implicit-zero value has materialized data")
		}
	default:
		return 0, errors.New("reference value storage is invalid")
	}
	return elements, nil
}

func (v Value) materialized() (Value, error) {
	elements, err := v.validate()
	if err != nil || v.Storage == ValueMaterialized {
		return v, err
	}
	if elements > uint64(math.MaxInt) {
		return Value{}, errors.New("reference zero value exceeds addressable memory")
	}
	return Value{Shape: v.Shape, Data: make([]float32, int(elements))}, nil
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
				if embedded, embeddedOK := node.Attrs.(tensor.EmbeddedInputAttributes); embeddedOK {
					value = Value{Shape: node.Shape, Data: embedded.Data}
					ok = true
				}
			}
			if !ok {
				return nil, fmt.Errorf("missing feed for input %q", node.Name)
			}
			if !value.Shape.Equal(node.Shape) {
				return nil, fmt.Errorf("feed shape for %q does not match graph", node.Name)
			}
			value, err = value.materialized()
			if err != nil {
				return nil, fmt.Errorf("feed storage for %q: %w", node.Name, err)
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
		value, err := ExecuteOperation(node, inputs)
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

// ExecuteOperation: single-node correctness bridge.
func ExecuteOperation(node *tensor.Tensor, inputs []Value) (Value, error) {
	if node == nil {
		return Value{}, errors.New("reference operation is nil")
	}
	descriptor, ok := tensor.DescribeOperation(node.Op)
	if !ok || descriptor.Backends&tensor.BackendReference == 0 {
		return Value{}, fmt.Errorf("unsupported reference operation %s", node.Op)
	}
	view, aliases, err := tensor.ResolveStorageView(node)
	if err != nil {
		return Value{}, err
	}
	if aliases {
		if view.Input < 0 || view.Input >= len(inputs) {
			return Value{}, errors.New("reference storage view input is unavailable")
		}
		return materializeStorageView(node.Shape, inputs[view.Input], view.ElementOffset)
	}
	return executeNode(node, inputs)
}

func executeNode(node *tensor.Tensor, inputs []Value) (Value, error) {
	switch node.Op {
	case tensor.OpAdd:
		return elementwiseBroadcast(node.Shape, inputs[0], inputs[1], func(a, b float32) float32 { return a + b })
	case tensor.OpMultiply:
		return elementwiseBroadcast(node.Shape, inputs[0], inputs[1], func(a, b float32) float32 { return a * b })
	case tensor.OpDivide:
		return elementwiseBroadcast(node.Shape, inputs[0], inputs[1], func(a, b float32) float32 { return a / b })
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
	case tensor.OpBF16Round:
		output := make([]float32, len(inputs[0].Data))
		for i, value := range inputs[0].Data {
			output[i] = dtype.RoundBF16(value)
		}
		return Value{Shape: node.Shape, Data: output}, nil
	case tensor.OpRMSNorm:
		attributes, ok := node.Attrs.(tensor.RMSNormAttributes)
		if !ok {
			return Value{}, errors.New("invalid RMSNorm attributes")
		}
		return rmsNorm(node.Shape, inputs[0], attributes.Epsilon)
	case tensor.OpMADNorm:
		attributes, ok := node.Attrs.(tensor.MADNormAttributes)
		if !ok {
			return Value{}, errors.New("invalid MADNorm attributes")
		}
		return madNorm(node.Shape, inputs[0], attributes.Epsilon)
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
			gelu := float32(hostmath.GELUTanh(x))
			output[i] = float16Round(gelu)
		}
		return Value{Shape: node.Shape, Data: output}, nil
	case tensor.OpGELUErf:
		output := make([]float32, len(inputs[0].Data))
		for i, value := range inputs[0].Data {
			output[i] = float32(0.5 * float64(value) * (1 + math.Erf(float64(value)/math.Sqrt2)))
		}
		return Value{Shape: node.Shape, Data: output}, nil
	case tensor.OpReLU:
		output := make([]float32, len(inputs[0].Data))
		for i, value := range inputs[0].Data {
			if value > 0 {
				output[i] = value
			}
		}
		return Value{Shape: node.Shape, Data: output}, nil
	case tensor.OpConv1DSame:
		attributes, ok := node.Attrs.(tensor.Conv1DAttributes)
		if !ok {
			return Value{}, errors.New("invalid same Conv1D attributes")
		}
		return conv1DSame(node.Shape, inputs[0], inputs[1], inputs[2], attributes.Depthwise)
	case tensor.OpConv2D:
		attributes, ok := node.Attrs.(tensor.Conv2DAttributes)
		if !ok {
			return Value{}, errors.New("invalid Conv2D attributes")
		}
		var bias Value
		if attributes.HasBias {
			bias = inputs[2]
		}
		return conv2D(node.Shape, inputs[0], inputs[1], bias, attributes)
	case tensor.OpWindowPartition2D:
		return windowPartition2D(node.Shape, inputs[0], node.Attrs.(tensor.Window2DAttributes), false)
	case tensor.OpWindowUnpartition2D:
		return windowPartition2D(node.Shape, inputs[0], node.Attrs.(tensor.Window2DAttributes), true)
	case tensor.OpPixelShuffle2D:
		return pixelShuffle2D(node.Shape, inputs[0], node.Attrs.(tensor.PixelShuffle2DAttributes))
	case tensor.OpSAMAttention:
		return samAttention(node.Shape, inputs, node.Attrs.(tensor.SAMAttentionAttributes))
	case tensor.OpGroupNorm:
		attributes, ok := node.Attrs.(tensor.GroupNormAttributes)
		if !ok {
			return Value{}, errors.New("invalid group norm attributes")
		}
		return groupNorm(node.Shape, inputs[0], inputs[1], inputs[2], attributes.Groups, attributes.Epsilon)
	case tensor.OpHyperConnectionInit:
		return hyperConnectionInit(node.Shape, inputs, node.Attrs.(tensor.HyperConnectionAttributes))
	case tensor.OpHyperConnectionPre:
		return hyperConnectionPre(node.Shape, inputs, node.Attrs.(tensor.HyperConnectionAttributes))
	case tensor.OpHyperConnectionPost:
		return hyperConnectionPost(node.Shape, inputs, node.Attrs.(tensor.HyperConnectionAttributes))
	case tensor.OpHyperConnectionHead:
		return hyperConnectionHead(node.Shape, inputs, node.Attrs.(tensor.HyperConnectionAttributes))
	case tensor.OpCompressedAttention:
		return compressedAttention(node.Shape, inputs, node.Attrs.(tensor.CompressedAttentionAttributes))
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
	case tensor.OpAtan:
		output := make([]float32, len(inputs[0].Data))
		for i, value := range inputs[0].Data {
			output[i] = float32(math.Atan(float64(value)))
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
	case tensor.OpWKV6:
		return wkv6(node.Shape, inputs)
	case tensor.OpSumRows:
		return sumRows(node.Shape, inputs[0])
	case tensor.OpWKV7:
		return wkv7(node.Shape, inputs)
	case tensor.OpFWHT:
		return fwht(node.Shape, inputs[0])
	case tensor.OpTopK:
		return topK(node.Shape, inputs[0], node.Attrs.(tensor.TopKAttributes))
	case tensor.OpTopKPairs:
		return topKPairs(node.Shape, inputs[0], node.Attrs.(tensor.TopKAttributes))
	case tensor.OpTopKPartials:
		return topKPartials(node.Shape, inputs[0], node.Attrs.(tensor.TopKAttributes))
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
	case tensor.OpLoRAMerge:
		attributes, ok := node.Attrs.(tensor.LoRAMergeAttributes)
		if !ok {
			return Value{}, errors.New("invalid LoRA merge attributes")
		}
		return loraMerge(node.Shape, inputs, attributes)
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
	case tensor.OpAttention:
		attributes, ok := node.Attrs.(tensor.AttentionAttributes)
		if !ok {
			return Value{}, errors.New("invalid attention attributes")
		}
		// Optional inputs follow q/k/v from index 3 in a fixed order:
		// (bias|sinks), blockIDs, keyBias -- each present per its attribute flag.
		var bias, sinks, blockIDs, keyBias *Value
		idx := 3
		if attributes.RelativeBuckets != 0 && idx < len(inputs) {
			bias = &inputs[idx]
			idx++
		} else if attributes.HasSinks && idx < len(inputs) {
			sinks = &inputs[idx]
			idx++
		}
		if attributes.HasBlockMask && idx < len(inputs) {
			blockIDs = &inputs[idx]
			idx++
		}
		if attributes.HasKeyBias && idx < len(inputs) {
			keyBias = &inputs[idx]
			idx++
		}
		return attention(node.Shape, inputs[0], inputs[1], inputs[2], bias, sinks, blockIDs, keyBias, attributes)
	case tensor.OpConcat:
		attributes, ok := node.Attrs.(tensor.ConcatAttributes)
		if !ok {
			return Value{}, errors.New("invalid concat attributes")
		}
		return concat(node.Shape, inputs[0], inputs[1], attributes.Axis)
	case tensor.OpCacheAppend:
		attributes, ok := node.Attrs.(tensor.CacheAppendAttributes)
		if !ok {
			return Value{}, errors.New("invalid cache append attributes")
		}
		return cacheAppend(node.Shape, inputs[0], inputs[1], attributes)
	default:
		return Value{}, fmt.Errorf("unsupported operation %s", node.Op)
	}
}
