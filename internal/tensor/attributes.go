package tensor

import (
	"errors"
	"fmt"
	"math"
)

// Attributes: closed operation-attribute contract.
type Attributes interface {
	validFor(Op) bool
}

type emptyAttributes struct{}

func sameOp(got, want Op) bool { return got == want }

func (emptyAttributes) validFor(op Op) bool {
	switch op {
	case OpInput, OpAdd, OpMultiply, OpSoftmax, OpSiLU, OpMulMat, OpReshape,
		OpSigmoid, OpSoftplus, OpSSMConv, OpSSMScan, OpTranspose2D, OpGELU,
		OpReLUSquared, OpGroupedMulMat, OpTanh, OpExp, OpWKV6, OpSumRows,
		OpWKV7, OpFWHT, OpGatherLast, OpReLU, OpDivide, OpBF16Round, OpGELUErf,
		OpAtan:
		return true
	default:
		return false
	}
}

func (ScaleAttributes) validFor(op Op) bool     { return sameOp(op, OpScale) }
func (ClampAttributes) validFor(op Op) bool     { return sameOp(op, OpClamp) }
func (RMSNormAttributes) validFor(op Op) bool   { return sameOp(op, OpRMSNorm) }
func (MADNormAttributes) validFor(op Op) bool   { return sameOp(op, OpMADNorm) }
func (LayerNormAttributes) validFor(op Op) bool { return sameOp(op, OpLayerNorm) }
func (L2NormAttributes) validFor(op Op) bool    { return sameOp(op, OpL2Norm) }
func (XIELUAttributes) validFor(op Op) bool     { return sameOp(op, OpXIELU) }
func (MulMatAttributes) validFor(op Op) bool    { return sameOp(op, OpMulMat) }
func (GetRowsAttributes) validFor(op Op) bool   { return sameOp(op, OpGetRows) }
func (RoPEAttributes) validFor(op Op) bool {
	return sameOp(op, OpRoPENeoX) || sameOp(op, OpRoPENormal)
}
func (RoPEMultiAttributes) validFor(op Op) bool { return sameOp(op, OpRoPEMulti) }
func (AttentionAttributes) validFor(op Op) bool { return sameOp(op, OpAttention) }
func (Conv1DAttributes) validFor(op Op) bool    { return sameOp(op, OpConv1DSame) }
func (Conv2DAttributes) validFor(op Op) bool    { return sameOp(op, OpConv2D) }
func (Window2DAttributes) validFor(op Op) bool {
	return sameOp(op, OpWindowPartition2D) || sameOp(op, OpWindowUnpartition2D)
}
func (PixelShuffle2DAttributes) validFor(op Op) bool { return sameOp(op, OpPixelShuffle2D) }
func (SAMAttentionAttributes) validFor(op Op) bool   { return sameOp(op, OpSAMAttention) }
func (GroupNormAttributes) validFor(op Op) bool      { return sameOp(op, OpGroupNorm) }
func (MoEAttributes) validFor(op Op) bool            { return sameOp(op, OpMoE) }
func (CompressedAttentionAttributes) validFor(op Op) bool {
	return sameOp(op, OpCompressedAttention)
}
func (GatedDeltaNetAttributes) validFor(op Op) bool { return sameOp(op, OpGatedDeltaNet) }
func (GatedLinearAttentionAttributes) validFor(op Op) bool {
	return sameOp(op, OpGatedLinearAttention)
}
func (RepeatHeadsAttributes) validFor(op Op) bool { return sameOp(op, OpRepeatHeads) }
func (ConcatAttributes) validFor(op Op) bool      { return sameOp(op, OpConcat) }
func (GroupSliceAttributes) validFor(op Op) bool  { return sameOp(op, OpGroupSlice) }
func (FlatSliceAttributes) validFor(op Op) bool   { return sameOp(op, OpFlatSlice) }
func (CacheAppendAttributes) validFor(op Op) bool { return sameOp(op, OpCacheAppend) }
func (SparseAttentionAttributes) validFor(op Op) bool {
	return sameOp(op, OpSparseAttention)
}
func (IndexerScoreAttributes) validFor(op Op) bool  { return sameOp(op, OpIndexerScore) }
func (EmbeddedInputAttributes) validFor(op Op) bool { return sameOp(op, OpInput) }
func (LoRAMergeAttributes) validFor(op Op) bool     { return sameOp(op, OpLoRAMerge) }

func (HyperConnectionAttributes) validFor(op Op) bool {
	return sameOp(op, OpHyperConnectionInit) || sameOp(op, OpHyperConnectionPre) ||
		sameOp(op, OpHyperConnectionPost) || sameOp(op, OpHyperConnectionHead)
}

func (TopKAttributes) validFor(op Op) bool {
	return sameOp(op, OpTopK) || sameOp(op, OpTopKPairs) || sameOp(op, OpTopKPartials)
}

type runtimeWordAttributes interface {
	runtimeWords(*Tensor, []uint32) (int, error)
}

func copyRuntimeWords(destination, values []uint32) int {
	if destination != nil {
		copy(destination, values)
	}
	return len(values)
}

func (attributes GetRowsAttributes) runtimeWords(_ *Tensor, destination []uint32) (int, error) {
	return copyRuntimeWords(destination, attributes.Rows), nil
}

func (attributes RoPEAttributes) runtimeWords(_ *Tensor, destination []uint32) (int, error) {
	return copyRuntimeWords(destination, attributes.Positions), nil
}

func (attributes RoPEMultiAttributes) runtimeWords(_ *Tensor, destination []uint32) (int, error) {
	written := FirstOffset
	for axis := range attributes.Positions {
		var target []uint32
		if destination != nil {
			target = destination[written:]
		}
		written += copyRuntimeWords(target, attributes.Positions[axis])
	}
	return written, nil
}

func (attributes AttentionAttributes) runtimeWords(node *Tensor, destination []uint32) (int, error) {
	tokens := attributes.KeyValueTokens
	if tokens == FirstOffset {
		if node == nil || len(node.Inputs) <= SingletonExtent {
			return FirstOffset, errors.New("attention KV input is unavailable")
		}
		extent := node.Inputs[SingletonExtent].Shape.Dims[PairedExtent]
		if extent > math.MaxUint32 {
			return FirstOffset, errors.New("attention KV capacity exceeds uint32")
		}
		tokens = uint32(extent)
	}
	return copyRuntimeWords(destination, []uint32{tokens}), nil
}

func (attributes CacheAppendAttributes) runtimeWords(_ *Tensor, destination []uint32) (int, error) {
	return copyRuntimeWords(destination, []uint32{attributes.Offset}), nil
}

// RuntimeAttributeWords compiles device-visible typed attributes.
func RuntimeAttributeWords(node *Tensor, attributes Attributes, destination []uint32) (int, error) {
	if dynamic, ok := attributes.(runtimeWordAttributes); ok {
		return dynamic.runtimeWords(node, destination)
	}
	return FirstOffset, nil
}

// ValidateOperationAttributes checks one operation/descriptor pair.
func ValidateOperationAttributes(op Op, attributes Attributes) error {
	if _, ok := DescribeOperation(op); !ok {
		return fmt.Errorf("operation %d is invalid", op)
	}
	if attributes == nil {
		attributes = emptyAttributes{}
	}
	if attributes.validFor(op) {
		return nil
	}
	return fmt.Errorf("%s attributes are invalid", op)
}
