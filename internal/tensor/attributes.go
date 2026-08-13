package tensor

import (
	"errors"
	"fmt"
)

// Attributes: closed graph-attribute descriptor set.
type Attributes interface {
	tensorAttributes()
}

func (ScaleAttributes) tensorAttributes()                {}
func (ClampAttributes) tensorAttributes()                {}
func (RMSNormAttributes) tensorAttributes()              {}
func (LayerNormAttributes) tensorAttributes()            {}
func (L2NormAttributes) tensorAttributes()               {}
func (XIELUAttributes) tensorAttributes()                {}
func (MulMatAttributes) tensorAttributes()               {}
func (GetRowsAttributes) tensorAttributes()              {}
func (RoPEAttributes) tensorAttributes()                 {}
func (RoPEMultiAttributes) tensorAttributes()            {}
func (AttentionAttributes) tensorAttributes()            {}
func (Conv1DAttributes) tensorAttributes()               {}
func (Conv2DAttributes) tensorAttributes()               {}
func (Window2DAttributes) tensorAttributes()             {}
func (SAMAttentionAttributes) tensorAttributes()         {}
func (GroupNormAttributes) tensorAttributes()            {}
func (MoEAttributes) tensorAttributes()                  {}
func (HyperConnectionAttributes) tensorAttributes()      {}
func (CompressedAttentionAttributes) tensorAttributes()  {}
func (GatedDeltaNetAttributes) tensorAttributes()        {}
func (GatedLinearAttentionAttributes) tensorAttributes() {}
func (RepeatHeadsAttributes) tensorAttributes()          {}
func (ConcatAttributes) tensorAttributes()               {}
func (GroupSliceAttributes) tensorAttributes()           {}
func (FlatSliceAttributes) tensorAttributes()            {}
func (CacheAppendAttributes) tensorAttributes()          {}
func (TopKAttributes) tensorAttributes()                 {}
func (SparseAttentionAttributes) tensorAttributes()      {}
func (IndexerScoreAttributes) tensorAttributes()         {}
func (EmbeddedInputAttributes) tensorAttributes()        {}
func (LoRAMergeAttributes) tensorAttributes()            {}

type attributeKind uint8

const (
	attributeNone attributeKind = iota
	attributeScale
	attributeClamp
	attributeRMSNorm
	attributeLayerNorm
	attributeL2Norm
	attributeXIELU
	attributeMulMat
	attributeGetRows
	attributeRoPE
	attributeRoPEMulti
	attributeAttention
	attributeConv1D
	attributeConv2D
	attributeWindow2D
	attributeSAMAttention
	attributeGroupNorm
	attributeMoE
	attributeHyperConnection
	attributeCompressedAttention
	attributeGatedDeltaNet
	attributeGatedLinearAttention
	attributeRepeatHeads
	attributeConcat
	attributeGroupSlice
	attributeFlatSlice
	attributeTopK
	attributeSparseAttention
	attributeIndexerScore
	attributeEmbeddedInput
	attributeLoRAMerge
	attributeCacheAppend
)

var operationAttributeKinds = [...]attributeKind{
	OpScale:                attributeScale,
	OpClamp:                attributeClamp,
	OpRMSNorm:              attributeRMSNorm,
	OpLayerNorm:            attributeLayerNorm,
	OpL2Norm:               attributeL2Norm,
	OpXIELU:                attributeXIELU,
	OpMulMat:               attributeMulMat,
	OpGetRows:              attributeGetRows,
	OpRoPENeoX:             attributeRoPE,
	OpRoPENormal:           attributeRoPE,
	OpRoPEMulti:            attributeRoPEMulti,
	OpAttention:            attributeAttention,
	OpConv1DSame:           attributeConv1D,
	OpConv2D:               attributeConv2D,
	OpWindowPartition2D:    attributeWindow2D,
	OpWindowUnpartition2D:  attributeWindow2D,
	OpSAMAttention:         attributeSAMAttention,
	OpGroupNorm:            attributeGroupNorm,
	OpMoE:                  attributeMoE,
	OpHyperConnectionInit:  attributeHyperConnection,
	OpHyperConnectionPre:   attributeHyperConnection,
	OpHyperConnectionPost:  attributeHyperConnection,
	OpHyperConnectionHead:  attributeHyperConnection,
	OpCompressedAttention:  attributeCompressedAttention,
	OpGatedDeltaNet:        attributeGatedDeltaNet,
	OpGatedLinearAttention: attributeGatedLinearAttention,
	OpRepeatHeads:          attributeRepeatHeads,
	OpConcat:               attributeConcat,
	OpGroupSlice:           attributeGroupSlice,
	OpFlatSlice:            attributeFlatSlice,
	OpTopK:                 attributeTopK,
	OpTopKPairs:            attributeTopK,
	OpTopKPartials:         attributeTopK,
	OpSparseAttention:      attributeSparseAttention,
	OpIndexerScore:         attributeIndexerScore,
	OpLoRAMerge:            attributeLoRAMerge,
	OpCacheAppend:          attributeCacheAppend,
}

func validateOperationAttributes(op Op, attributes Attributes) error {
	if _, ok := DescribeOperation(op); !ok {
		return fmt.Errorf("operation %d is invalid", op)
	}
	if op == OpInput {
		if attributes == nil {
			return nil
		}
		if _, ok := attributes.(EmbeddedInputAttributes); ok {
			return nil
		}
		return errors.New("input attributes are invalid")
	}
	if op == OpMulMat && attributes == nil {
		return nil
	}
	expected := attributeNone
	if int(op) < len(operationAttributeKinds) {
		expected = operationAttributeKinds[op]
	}
	actual := attributeKindOf(attributes)
	if actual != expected {
		return fmt.Errorf("%s attributes are invalid", op)
	}
	return nil
}

// ValidateOperationAttributes checks one operation/descriptor pair.
func ValidateOperationAttributes(op Op, attributes Attributes) error {
	return validateOperationAttributes(op, attributes)
}

func attributeKindOf(attributes Attributes) attributeKind {
	switch attributes.(type) {
	case nil:
		return attributeNone
	case ScaleAttributes:
		return attributeScale
	case ClampAttributes:
		return attributeClamp
	case RMSNormAttributes:
		return attributeRMSNorm
	case LayerNormAttributes:
		return attributeLayerNorm
	case L2NormAttributes:
		return attributeL2Norm
	case XIELUAttributes:
		return attributeXIELU
	case MulMatAttributes:
		return attributeMulMat
	case GetRowsAttributes:
		return attributeGetRows
	case RoPEAttributes:
		return attributeRoPE
	case RoPEMultiAttributes:
		return attributeRoPEMulti
	case AttentionAttributes:
		return attributeAttention
	case Conv1DAttributes:
		return attributeConv1D
	case Conv2DAttributes:
		return attributeConv2D
	case Window2DAttributes:
		return attributeWindow2D
	case SAMAttentionAttributes:
		return attributeSAMAttention
	case GroupNormAttributes:
		return attributeGroupNorm
	case MoEAttributes:
		return attributeMoE
	case HyperConnectionAttributes:
		return attributeHyperConnection
	case CompressedAttentionAttributes:
		return attributeCompressedAttention
	case GatedDeltaNetAttributes:
		return attributeGatedDeltaNet
	case GatedLinearAttentionAttributes:
		return attributeGatedLinearAttention
	case RepeatHeadsAttributes:
		return attributeRepeatHeads
	case ConcatAttributes:
		return attributeConcat
	case GroupSliceAttributes:
		return attributeGroupSlice
	case FlatSliceAttributes:
		return attributeFlatSlice
	case TopKAttributes:
		return attributeTopK
	case SparseAttentionAttributes:
		return attributeSparseAttention
	case IndexerScoreAttributes:
		return attributeIndexerScore
	case EmbeddedInputAttributes:
		return attributeEmbeddedInput
	case LoRAMergeAttributes:
		return attributeLoRAMerge
	case CacheAppendAttributes:
		return attributeCacheAppend
	default:
		return attributeNone
	}
}
