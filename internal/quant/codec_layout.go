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
	group   affineGroupLayout
	delta   codecField
	minimum codecField
	scales  codecField
	high    codecField
	packed  codecField
}

type affineGroupLayout struct {
	width           int
	levelMax        int
	levelZero       int
	packedBits      uint
	scaleMax        int
	scaleBits       uint
	scalePackedBits uint
	scaleZero       int
}

func (l affineGroupLayout) packedLevelMax() int {
	return 1<<l.packedBits - 1
}

func (l affineGroupLayout) scaleMask() byte {
	return byte(1<<l.scaleBits - 1)
}

func (l affineKCodecLayout) groupCount() int {
	return l.block.elements / l.group.width
}

func (l affineKCodecLayout) groupInput(values []float32, group int) []float32 {
	start := group * l.group.width
	return values[start : start+l.group.width]
}

func (l affineKCodecLayout) groupLevels(values []byte, group int) []byte {
	start := group * l.group.width
	return values[start : start+l.group.width]
}

func (l affineKCodecLayout) groupLevels8(values []int8, group int) []int8 {
	start := group * l.group.width
	return values[start : start+l.group.width]
}

func (l affineKCodecLayout) setScaleMinimum(data []byte, group, scale, minimum int) {
	split := l.groupCount() / 2
	if group < split {
		data[group] = byte(scale)
		data[group+split] = byte(minimum)
		return
	}
	lowMask := 1<<l.group.scalePackedBits - 1
	highShift := binaryschema.BitsPerByte - (l.group.scaleBits - l.group.scalePackedBits)
	data[group+split] = byte(scale&lowMask | (minimum&lowMask)<<l.group.scalePackedBits)
	data[group-split] |= byte((scale >> l.group.scalePackedBits) << highShift)
	data[group] |= byte((minimum >> l.group.scalePackedBits) << highShift)
}

func (l affineKCodecLayout) scaleMinimum(data []byte, group int) (int, int) {
	split := l.groupCount() / 2
	if group < split {
		mask := int(l.group.scaleMask())
		return int(data[group]) & mask, int(data[group+split]) & mask
	}
	lowMask := 1<<l.group.scalePackedBits - 1
	highShift := binaryschema.BitsPerByte - (l.group.scaleBits - l.group.scalePackedBits)
	scale := int(data[group+split])&lowMask |
		int(data[group-split]>>highShift)<<l.group.scalePackedBits
	minimum := int(data[group+split]>>l.group.scalePackedBits) |
		int(data[group]>>highShift)<<l.group.scalePackedBits
	return scale, minimum
}

type scalarCodecLayout struct {
	block      blockCodecLayout
	levels     int
	zeroPoint  int
	packedBits uint
	scale      codecField
	minimum    codecField
	high       codecField
	packed     codecField
	tail       codecField
	tailGroup  int
}

type scalarCodecFields struct {
	scale   int
	minimum int
	high    int
}

func q8KScalarCodec() scalarCodecLayout {
	layout := scalarCodec(dtype.Q8K, 8, scalarCodecFields{scale: binaryschema.Uint32Bytes})
	layout.tailGroup = kLaneWidth / 2
	tailBytes := layout.block.elements / layout.tailGroup * binaryschema.Uint16Bytes
	layout.packed.size -= tailBytes
	layout.tail = field(layout.packed.end(), tailBytes)
	return layout
}

func scalarCodec(dataType dtype.Type, bits uint, fields scalarCodecFields) scalarCodecLayout {
	block := blockLayout(dataType)
	minimumStart := fields.scale
	highStart := minimumStart + fields.minimum
	packedStart := highStart + fields.high
	levels := 1<<bits - 1
	zeroPoint := 0
	if fields.minimum == 0 {
		zeroPoint = (levels + 1) / 2
	}
	packedBits := bits
	if fields.high != 0 {
		packedBits--
	}
	return scalarCodecLayout{
		block: block, levels: levels, zeroPoint: zeroPoint, packedBits: packedBits,
		scale: codecField{size: fields.scale}, minimum: field(minimumStart, fields.minimum),
		high: field(highStart, fields.high), packed: field(packedStart, block.size-packedStart),
	}
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
	scalarCodecs      = [dtype.Count]scalarCodecLayout{
		dtype.Q1_0: scalarCodec(dtype.Q1_0, 1, scalarCodecFields{scale: binaryschema.Uint16Bytes}),
		dtype.Q2_0: scalarCodec(dtype.Q2_0, 2, scalarCodecFields{scale: binaryschema.Uint16Bytes}),
		dtype.Q4_0: scalarCodec(dtype.Q4_0, 4, scalarCodecFields{scale: binaryschema.Uint16Bytes}),
		dtype.Q4_1: scalarCodec(dtype.Q4_1, 4, scalarCodecFields{scale: binaryschema.Uint16Bytes, minimum: binaryschema.Uint16Bytes}),
		dtype.Q5_0: scalarCodec(dtype.Q5_0, 5, scalarCodecFields{scale: binaryschema.Uint16Bytes, high: binaryschema.Uint32Bytes}),
		dtype.Q5_1: scalarCodec(dtype.Q5_1, 5, scalarCodecFields{scale: binaryschema.Uint16Bytes, minimum: binaryschema.Uint16Bytes, high: binaryschema.Uint32Bytes}),
		dtype.Q8_0: scalarCodec(dtype.Q8_0, 8, scalarCodecFields{scale: binaryschema.Uint16Bytes}),
		dtype.Q8_1: scalarCodec(dtype.Q8_1, 8, scalarCodecFields{scale: binaryschema.Uint16Bytes, minimum: binaryschema.Uint16Bytes}),
		dtype.Q8K:  q8KScalarCodec(),
	}
	q2KCodec = affineKCodecLayout{
		block: blockLayout(dtype.Q2K),
		group: affineGroupLayout{width: 16, levelMax: 3, packedBits: 2, scaleMax: 15, scaleBits: 4, scalePackedBits: 4},
		delta: field(q2KScaleStart, iqScaleBytes), minimum: field(q2KMinimumStart, iqScaleBytes),
		scales: field(q2KScaleMinStart, q2KScaleMinBytes), packed: field(q2KPackedStart, q2KPackedBytes),
	}
	q3KCodec = affineKCodecLayout{
		block: blockLayout(dtype.Q3K),
		group: affineGroupLayout{width: 16, levelMax: 7, levelZero: 4, packedBits: 2, scaleMax: 63, scaleBits: 6, scalePackedBits: 4, scaleZero: 32},
		delta: field(q3KDeltaStart, iqScaleBytes), scales: field(q3KScaleStart, q3KScaleBytes),
		high: field(q3KHighMaskStart, q3KHighMaskBytes), packed: field(q3KPackedStart, q3KPackedBytes),
	}
	q4KCodec = affineKCodecLayout{
		block: blockLayout(dtype.Q4K),
		group: affineGroupLayout{width: kLaneWidth, levelMax: 15, packedBits: 4, scaleMax: 63, scaleBits: 6, scalePackedBits: 4},
		delta: field(q45KDeltaStart, iqScaleBytes), minimum: field(q45KMinimumStart, iqScaleBytes),
		scales: field(q45KScaleMinStart, q45KScaleMinBytes), packed: field(q45KPayloadStart, q45KPackedBytes),
	}
	q5KCodec = affineKCodecLayout{
		block: blockLayout(dtype.Q5K),
		group: affineGroupLayout{width: kLaneWidth, levelMax: 31, packedBits: 4, scaleMax: 63, scaleBits: 6, scalePackedBits: 4},
		delta: field(q45KDeltaStart, iqScaleBytes), minimum: field(q45KMinimumStart, iqScaleBytes),
		scales: field(q45KScaleMinStart, q45KScaleMinBytes), high: field(q45KPayloadStart, q5KHighMaskBytes),
		packed: field(q5KPackedStart, q45KPackedBytes),
	}
	q6KCodec = affineKCodecLayout{
		block: blockLayout(dtype.Q6K),
		group: affineGroupLayout{width: 16, levelMax: 63, levelZero: 32, packedBits: 4, scaleMax: 127, scaleBits: 8, scalePackedBits: 8, scaleZero: 128},
		delta: field(q6KDeltaStart, iqScaleBytes), scales: field(q6KScaleStart, q6KScaleBytes),
		high: field(q6KHighStart, q6KHighBytes), packed: field(q6KLowerStart, q6KLowerBytes),
	}
)
