package quant

import (
	"overgo/internal/binaryschema"
	"overgo/internal/tensor/dtype"
)

type codecField struct {
	start int
	size  int
}

func field(start, size int) codecField {
	return codecField{start: start, size: size}
}

func (f codecField) bytes(block []byte) []byte {
	return block[f.start : f.start+f.size]
}

func (f codecField) end() int {
	return f.start + f.size
}

type blockCodecLayout struct {
	dataType dtype.Type
	elements int
	size     int
}

func blockLayout(dataType dtype.Type) blockCodecLayout {
	traits, ok := dataType.Traits()
	if !ok {
		panic("quant: static codec has no dtype traits")
	}
	return blockCodecLayout{
		dataType: dataType,
		elements: int(traits.BlockSize),
		size:     int(traits.TypeSize),
	}
}

func (l blockCodecLayout) input(values []float32, block int) []float32 {
	return values[block*l.elements : (block+1)*l.elements]
}

func (l blockCodecLayout) storage(data []byte, block int) []byte {
	return data[block*l.size : (block+1)*l.size]
}

func (l blockCodecLayout) storage64(data []byte, block uint64) []byte {
	start := int(block) * l.size
	return data[start : start+l.size]
}

type affineKCodecLayout struct {
	block   blockCodecLayout
	delta   codecField
	minimum codecField
	scales  codecField
	high    codecField
	packed  codecField
}

// GGML IQ block layout facts. Offsets derive from preceding fields.
const (
	iqScaleBytes      = binaryschema.Uint16Bytes
	iqPackedGridBytes = 64
	iqPackedGridStart = iqScaleBytes
	iqAuxiliaryStart  = iqPackedGridStart + iqPackedGridBytes

	iq2XSScaleBytes = 8
	iq2XSScaleStart = iqAuxiliaryStart

	iq2SHighBytes  = 8
	iq2SGridBytes  = iqPackedGridBytes / 2
	iq2SGridStart  = iqPackedGridStart
	iq2SSignBytes  = iqPackedGridBytes - iq2SGridBytes
	iq2SSignStart  = iq2SGridStart + iq2SGridBytes
	iq2SHighStart  = iqAuxiliaryStart
	iq2SScaleBytes = 8
	iq2SScaleStart = iq2SHighStart + iq2SHighBytes

	iq3XXSScaleSignBytes = 32
	iq3XXSScaleSignStart = iqAuxiliaryStart

	iq3SHighBytes  = 8
	iq3SHighStart  = iqAuxiliaryStart
	iq3SSignBytes  = 32
	iq3SSignStart  = iq3SHighStart + iq3SHighBytes
	iq3SScaleBytes = 4
	iq3SScaleStart = iq3SSignStart + iq3SSignBytes

	iq1DeltaMagnitude = float32(0.125)

	nvfp4ScaleCount  = 4
	nvfp4PackedStart = nvfp4ScaleCount

	tq1WideLaneWidth     = 32
	tq1NarrowLaneWidth   = 16
	tq1TailLaneWidth     = 4
	tq1MainTritCount     = 5
	tq1TailTritCount     = 4
	tq1NarrowInputStart  = tq1WideLaneWidth * tq1MainTritCount
	tq1TailInputStart    = tq1NarrowInputStart + tq1NarrowLaneWidth*tq1MainTritCount
	tq1NarrowPackedStart = tq1WideLaneWidth
	tq1TailPackedStart   = tq1NarrowPackedStart + tq1NarrowLaneWidth
	tq1PackedBytes       = tq1TailPackedStart + tq1TailLaneWidth
	tq1ScaleStart        = tq1PackedBytes

	tq2SectionCount = 2
	tq2LaneWidth    = 32
	tq2GroupCount   = 4
	tq2SectionWidth = tq2LaneWidth * tq2GroupCount
	tq2PackedBytes  = tq2SectionCount * tq2LaneWidth
	tq2ScaleStart   = tq2PackedBytes

	kLaneWidth = 32

	q2KScaleMinBytes = 16
	q2KScaleMinStart = 0
	q2KPackedBytes   = 64
	q2KPackedStart   = q2KScaleMinStart + q2KScaleMinBytes
	q2KScaleStart    = q2KPackedStart + q2KPackedBytes
	q2KMinimumStart  = q2KScaleStart + iqScaleBytes

	q3KHighMaskBytes = 32
	q3KHighMaskStart = 0
	q3KPackedBytes   = 64
	q3KPackedStart   = q3KHighMaskStart + q3KHighMaskBytes
	q3KScaleBytes    = 12
	q3KScaleStart    = q3KPackedStart + q3KPackedBytes
	q3KDeltaStart    = q3KScaleStart + q3KScaleBytes

	q45KDeltaStart    = 0
	q45KMinimumStart  = q45KDeltaStart + iqScaleBytes
	q45KScaleMinBytes = 12
	q45KScaleMinStart = q45KMinimumStart + iqScaleBytes
	q45KPayloadStart  = q45KScaleMinStart + q45KScaleMinBytes
	q45KPackedBytes   = 128
	q5KHighMaskBytes  = 32
	q5KPackedStart    = q45KPayloadStart + q5KHighMaskBytes

	q6KLowerBytes = 128
	q6KLowerStart = 0
	q6KHighBytes  = 64
	q6KHighStart  = q6KLowerStart + q6KLowerBytes
	q6KScaleBytes = 16
	q6KScaleStart = q6KHighStart + q6KHighBytes
	q6KDeltaStart = q6KScaleStart + q6KScaleBytes

	q6KScaleMagnitude = 128
	q6KLevelMagnitude = 32

	q1PackedBytes = 16
)

var (
	iq2XXSBlockLayout = blockLayout(dtype.IQ2XXS)
	iq2XSBlockLayout  = blockLayout(dtype.IQ2XS)
	iq2SBlockLayout   = blockLayout(dtype.IQ2S)
	iq3XXSBlockLayout = blockLayout(dtype.IQ3XXS)
	iq3SBlockLayout   = blockLayout(dtype.IQ3S)
	iq1SBlockLayout   = blockLayout(dtype.IQ1S)
	iq1MBlockLayout   = blockLayout(dtype.IQ1M)
	iq4NLBlockLayout  = blockLayout(dtype.IQ4NL)
	iq4XSBlockLayout  = blockLayout(dtype.IQ4XS)
	tq1BlockLayout    = blockLayout(dtype.TQ1_0)
	tq2BlockLayout    = blockLayout(dtype.TQ2_0)
	mxfp4BlockLayout  = blockLayout(dtype.MXFP4)
	nvfp4BlockLayout  = blockLayout(dtype.NVFP4)
	q1BlockLayout     = blockLayout(dtype.Q1_0)
	q2BlockLayout     = blockLayout(dtype.Q2_0)
	q4BlockLayout     = blockLayout(dtype.Q4_0)
	q5BlockLayout     = blockLayout(dtype.Q5_0)
	q8BlockLayout     = blockLayout(dtype.Q8_0)
	q8KBlockLayout    = blockLayout(dtype.Q8K)
	q2KCodec          = affineKCodecLayout{
		block: blockLayout(dtype.Q2K),
		delta: field(q2KScaleStart, iqScaleBytes), minimum: field(q2KMinimumStart, iqScaleBytes),
		scales: field(q2KScaleMinStart, q2KScaleMinBytes), packed: field(q2KPackedStart, q2KPackedBytes),
	}
	q3KCodec = affineKCodecLayout{
		block: blockLayout(dtype.Q3K),
		delta: field(q3KDeltaStart, iqScaleBytes), scales: field(q3KScaleStart, q3KScaleBytes),
		high: field(q3KHighMaskStart, q3KHighMaskBytes), packed: field(q3KPackedStart, q3KPackedBytes),
	}
	q4KCodec = affineKCodecLayout{
		block: blockLayout(dtype.Q4K),
		delta: field(q45KDeltaStart, iqScaleBytes), minimum: field(q45KMinimumStart, iqScaleBytes),
		scales: field(q45KScaleMinStart, q45KScaleMinBytes), packed: field(q45KPayloadStart, q45KPackedBytes),
	}
	q5KCodec = affineKCodecLayout{
		block: blockLayout(dtype.Q5K),
		delta: field(q45KDeltaStart, iqScaleBytes), minimum: field(q45KMinimumStart, iqScaleBytes),
		scales: field(q45KScaleMinStart, q45KScaleMinBytes), high: field(q45KPayloadStart, q5KHighMaskBytes),
		packed: field(q5KPackedStart, q45KPackedBytes),
	}
	q6KCodec = affineKCodecLayout{
		block: blockLayout(dtype.Q6K),
		delta: field(q6KDeltaStart, iqScaleBytes), scales: field(q6KScaleStart, q6KScaleBytes),
		high: field(q6KHighStart, q6KHighBytes), packed: field(q6KLowerStart, q6KLowerBytes),
	}
)
