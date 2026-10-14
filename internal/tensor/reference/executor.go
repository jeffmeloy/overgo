package reference

import (
	"errors"
	"fmt"
	"math"
	"slices"

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
	value, err := BorrowedValue(shape, data)
	if err != nil {
		return Value{}, err
	}
	value.Data = slices.Clone(data)
	return value, nil
}

// BorrowedValue validates data against shape without copying it. The caller
// must keep data immutable for the lifetime of the returned value.
func BorrowedValue(shape tensor.Shape, data []float32) (Value, error) {
	elements, err := shape.Elements()
	if err != nil {
		return Value{}, err
	}
	if elements != uint64(len(data)) {
		return Value{}, fmt.Errorf("reference data has %d elements, need %d", len(data), elements)
	}
	return Value{Shape: shape, Data: data}, nil
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

// Execute evaluates one graph from fixture feeds.
func Execute(outputs []*tensor.Tensor, feeds map[*tensor.Tensor]Value) (map[*tensor.Tensor]Value, error) {
	program, err := Compile(outputs...)
	if err != nil {
		return nil, err
	}
	inputs := program.NewInputs()
	for node, value := range feeds {
		if program.HasInput(node) {
			if err := inputs.Set(node, value); err != nil {
				return nil, err
			}
		}
	}
	return program.Execute(inputs, program.NewWorkspace())
}

// ExecuteOperation: single-node correctness bridge.
//   - attributes ValidateOperationAttributes refuses are refused, because executeNode asserts each operation's attributes by type
func ExecuteOperation(node *tensor.Tensor, inputs []Value) (Value, error) {
	if node == nil {
		return Value{}, errors.New("reference operation is nil")
	}
	descriptor, ok := tensor.DescribeOperation(node.Op)
	if !ok || descriptor.Backends&tensor.BackendReference == 0 {
		return Value{}, fmt.Errorf("unsupported reference operation %s", node.Op)
	}
	if err := tensor.ValidateOperationAttributes(node.Op, node.Attrs); err != nil {
		return Value{}, err
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

func mapValues(shape tensor.Shape, input Value, transform func(float32) float32) Value {
	output := make([]float32, len(input.Data))
	for index, value := range input.Data {
		output[index] = transform(value)
	}
	return Value{Shape: shape, Data: output}
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
		attributes := node.Attrs.(tensor.ScaleAttributes)
		output := make([]float32, len(inputs[0].Data))
		for i, value := range inputs[0].Data {
			output[i] = value * attributes.Value
		}
		return Value{Shape: node.Shape, Data: output}, nil
	case tensor.OpClamp:
		attributes := node.Attrs.(tensor.ClampAttributes)
		output := make([]float32, len(inputs[0].Data))
		for i, value := range inputs[0].Data {
			output[i] = min(max(value, attributes.Minimum), attributes.Maximum)
		}
		return Value{Shape: node.Shape, Data: output}, nil
	case tensor.OpBF16Round:
		return mapValues(node.Shape, inputs[0], dtype.RoundBF16), nil
	case tensor.OpRMSNorm:
		return rmsNorm(node.Shape, inputs[0], node.Attrs.(tensor.RMSNormAttributes).Epsilon)
	case tensor.OpMADNorm:
		return madNorm(node.Shape, inputs[0], node.Attrs.(tensor.MADNormAttributes).Epsilon)
	case tensor.OpLayerNorm:
		return layerNorm(node.Shape, inputs[0], node.Attrs.(tensor.LayerNormAttributes).Epsilon)
	case tensor.OpSoftmax:
		return softmax(node.Shape, inputs[0])
	case tensor.OpSiLU:
		return mapValues(node.Shape, inputs[0], func(value float32) float32 {
			return value / (1 + float32(math.Exp(float64(-value))))
		}), nil
	case tensor.OpGELU:
		return mapValues(node.Shape, inputs[0], RoundedGELUTanh), nil
	case tensor.OpGELUErf:
		return mapValues(node.Shape, inputs[0], func(value float32) float32 {
			return float32(0.5 * float64(value) * (1 + math.Erf(float64(value)/math.Sqrt2)))
		}), nil
	case tensor.OpReLU:
		return mapValues(node.Shape, inputs[0], func(value float32) float32 {
			if value > 0 {
				return value
			}
			return 0
		}), nil
	case tensor.OpConv1DSame:
		return conv1DSame(node.Shape, inputs[0], inputs[1], inputs[2], node.Attrs.(tensor.Conv1DAttributes).Depthwise)
	case tensor.OpConv2D:
		attributes := node.Attrs.(tensor.Conv2DAttributes)
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
		attributes := node.Attrs.(tensor.GroupNormAttributes)
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
		attributes := node.Attrs.(tensor.XIELUAttributes)
		output := make([]float32, len(inputs[0].Data))
		for i, value := range inputs[0].Data {
			if value > 0 {
				output[i] = attributes.AlphaP*value*value + attributes.Beta*value
				continue
			}
			minimum := float32(min(float64(value), float64(attributes.Epsilon)))
			output[i] = (float32(math.Expm1(float64(minimum)))-value)*attributes.AlphaN +
				attributes.Beta*value
		}
		return Value{Shape: node.Shape, Data: output}, nil
	case tensor.OpReLUSquared:
		return mapValues(node.Shape, inputs[0], func(value float32) float32 {
			if value > 0 {
				return value * value
			}
			return 0
		}), nil
	case tensor.OpSigmoid:
		return mapValues(node.Shape, inputs[0], func(value float32) float32 {
			return 1 / (1 + float32(math.Exp(float64(-value))))
		}), nil
	case tensor.OpSoftplus:
		return mapValues(node.Shape, inputs[0], func(value float32) float32 {
			absolute := math.Abs(float64(value))
			return float32(max(float64(value), 0) + math.Log1p(math.Exp(-absolute)))
		}), nil
	case tensor.OpTanh:
		return mapValues(node.Shape, inputs[0], func(value float32) float32 {
			return float32(math.Tanh(float64(value)))
		}), nil
	case tensor.OpAtan:
		return mapValues(node.Shape, inputs[0], func(value float32) float32 {
			return float32(math.Atan(float64(value)))
		}), nil
	case tensor.OpExp:
		return mapValues(node.Shape, inputs[0], func(value float32) float32 {
			return float32(math.Exp(float64(value)))
		}), nil
	case tensor.OpL2Norm:
		return l2Norm(node.Shape, inputs[0], node.Attrs.(tensor.L2NormAttributes).Epsilon)
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
		return moe(node.Shape, inputs, node.Attrs.(tensor.MoEAttributes))
	case tensor.OpLoRAMerge:
		return loraMerge(node.Shape, inputs, node.Attrs.(tensor.LoRAMergeAttributes))
	case tensor.OpRepeatHeads:
		if node.Attrs.(tensor.RepeatHeadsAttributes).Heads == 0 {
			return Value{}, errors.New("invalid RepeatHeads attributes")
		}
		return repeatHeads(node.Shape, inputs[0]), nil
	case tensor.OpTranspose2D:
		return transpose2D(node.Shape, inputs[0])
	case tensor.OpGroupSlice:
		return groupSlice(node.Shape, inputs[0], node.Attrs.(tensor.GroupSliceAttributes))
	case tensor.OpMulMat:
		return mulMat(node.Shape, inputs[0], inputs[1])
	case tensor.OpGroupedMulMat:
		return groupedMulMat(node.Shape, inputs[0], inputs[1])
	case tensor.OpGetRows:
		return getRows(node.Shape, inputs[0], node.Attrs.(tensor.GetRowsAttributes).Rows)
	case tensor.OpRoPENeoX:
		return rope(node.Shape, inputs, node.Attrs.(tensor.RoPEAttributes), node.Op)
	case tensor.OpRoPENormal:
		return rope(node.Shape, inputs, node.Attrs.(tensor.RoPEAttributes), node.Op)
	case tensor.OpRoPEMulti:
		return ropeMulti(node.Shape, inputs[0], node.Attrs.(tensor.RoPEMultiAttributes))
	case tensor.OpAttention:
		attributes := node.Attrs.(tensor.AttentionAttributes)
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
		return concat(node.Shape, inputs[0], inputs[1], node.Attrs.(tensor.ConcatAttributes).Axis)
	case tensor.OpCacheAppend:
		return cacheAppend(node.Shape, inputs[0], inputs[1], node.Attrs.(tensor.CacheAppendAttributes))
	default:
		return Value{}, fmt.Errorf("unsupported operation %s", node.Op)
	}
}
