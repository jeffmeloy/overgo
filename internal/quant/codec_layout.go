package quant

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

type blockCodecLayout struct {
	width int
	size  int
}

func (l blockCodecLayout) input(values []float32, block int) []float32 {
	return values[block*l.width : (block+1)*l.width]
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
	iqScaleBytes      = 2
	iqPackedGridBytes = 64
	iqPackedGridStart = iqScaleBytes
	iqAuxiliaryStart  = iqPackedGridStart + iqPackedGridBytes

	iq2XXSAuxiliaryBytes = 0
	iq2XXSBlockBytes     = iqAuxiliaryStart + iq2XXSAuxiliaryBytes

	iq2XSScaleBytes = 8
	iq2XSScaleStart = iqAuxiliaryStart
	iq2XSBlockBytes = iq2XSScaleStart + iq2XSScaleBytes

	iq2SHighBytes  = 8
	iq2SGridBytes  = iqPackedGridBytes / 2
	iq2SGridStart  = iqPackedGridStart
	iq2SSignBytes  = iqPackedGridBytes - iq2SGridBytes
	iq2SSignStart  = iq2SGridStart + iq2SGridBytes
	iq2SHighStart  = iqAuxiliaryStart
	iq2SScaleBytes = 8
	iq2SScaleStart = iq2SHighStart + iq2SHighBytes
	iq2SBlockBytes = iq2SScaleStart + iq2SScaleBytes

	iq3XXSScaleSignBytes = 32
	iq3XXSScaleSignStart = iqAuxiliaryStart
	iq3XXSBlockBytes     = iq3XXSScaleSignStart + iq3XXSScaleSignBytes

	iq3SHighBytes  = 8
	iq3SHighStart  = iqAuxiliaryStart
	iq3SSignBytes  = 32
	iq3SSignStart  = iq3SHighStart + iq3SHighBytes
	iq3SScaleBytes = 4
	iq3SScaleStart = iq3SSignStart + iq3SSignBytes
	iq3SBlockBytes = iq3SScaleStart + iq3SScaleBytes

	tqBlockWidth   = 256
	tq1PackedBytes = 52
	tq1ScaleStart  = tq1PackedBytes
	tq1BlockBytes  = tq1ScaleStart + iqScaleBytes
	tq2PackedBytes = 64
	tq2ScaleStart  = tq2PackedBytes
	tq2BlockBytes  = tq2ScaleStart + iqScaleBytes

	kBlockWidth = 256
	kLaneWidth  = 32

	q2KScaleMinBytes = 16
	q2KScaleMinStart = 0
	q2KPackedBytes   = 64
	q2KPackedStart   = q2KScaleMinStart + q2KScaleMinBytes
	q2KScaleStart    = q2KPackedStart + q2KPackedBytes
	q2KMinimumStart  = q2KScaleStart + iqScaleBytes
	q2KBlockBytes    = q2KMinimumStart + iqScaleBytes

	q3KHighMaskBytes = 32
	q3KHighMaskStart = 0
	q3KPackedBytes   = 64
	q3KPackedStart   = q3KHighMaskStart + q3KHighMaskBytes
	q3KScaleBytes    = 12
	q3KScaleStart    = q3KPackedStart + q3KPackedBytes
	q3KDeltaStart    = q3KScaleStart + q3KScaleBytes
	q3KBlockBytes    = q3KDeltaStart + iqScaleBytes

	q45KDeltaStart    = 0
	q45KMinimumStart  = q45KDeltaStart + iqScaleBytes
	q45KScaleMinBytes = 12
	q45KScaleMinStart = q45KMinimumStart + iqScaleBytes
	q45KPayloadStart  = q45KScaleMinStart + q45KScaleMinBytes
	q45KPackedBytes   = 128
	q4KBlockBytes     = q45KPayloadStart + q45KPackedBytes
	q5KHighMaskBytes  = 32
	q5KPackedStart    = q45KPayloadStart + q5KHighMaskBytes
	q5KBlockBytes     = q5KPackedStart + q45KPackedBytes

	q6KLowerBytes = 128
	q6KLowerStart = 0
	q6KHighBytes  = 64
	q6KHighStart  = q6KLowerStart + q6KLowerBytes
	q6KScaleBytes = 16
	q6KScaleStart = q6KHighStart + q6KHighBytes
	q6KDeltaStart = q6KScaleStart + q6KScaleBytes
	q6KBlockBytes = q6KDeltaStart + iqScaleBytes

	q6KScaleMagnitude = 128
	q6KLevelMagnitude = 32

	q1BlockWidth  = 128
	q1PackedBytes = q1BlockWidth / 8
	q1BlockBytes  = iqScaleBytes + q1PackedBytes
)

var (
	q2KCodec = affineKCodecLayout{
		block: blockCodecLayout{width: kBlockWidth, size: q2KBlockBytes},
		delta: field(q2KScaleStart, iqScaleBytes), minimum: field(q2KMinimumStart, iqScaleBytes),
		scales: field(q2KScaleMinStart, q2KScaleMinBytes), packed: field(q2KPackedStart, q2KPackedBytes),
	}
	q3KCodec = affineKCodecLayout{
		block: blockCodecLayout{width: kBlockWidth, size: q3KBlockBytes},
		delta: field(q3KDeltaStart, iqScaleBytes), scales: field(q3KScaleStart, q3KScaleBytes),
		high: field(q3KHighMaskStart, q3KHighMaskBytes), packed: field(q3KPackedStart, q3KPackedBytes),
	}
	q4KCodec = affineKCodecLayout{
		block: blockCodecLayout{width: kBlockWidth, size: q4KBlockBytes},
		delta: field(q45KDeltaStart, iqScaleBytes), minimum: field(q45KMinimumStart, iqScaleBytes),
		scales: field(q45KScaleMinStart, q45KScaleMinBytes), packed: field(q45KPayloadStart, q45KPackedBytes),
	}
	q5KCodec = affineKCodecLayout{
		block: blockCodecLayout{width: kBlockWidth, size: q5KBlockBytes},
		delta: field(q45KDeltaStart, iqScaleBytes), minimum: field(q45KMinimumStart, iqScaleBytes),
		scales: field(q45KScaleMinStart, q45KScaleMinBytes), high: field(q45KPayloadStart, q5KHighMaskBytes),
		packed: field(q5KPackedStart, q45KPackedBytes),
	}
	q6KCodec = affineKCodecLayout{
		block: blockCodecLayout{width: kBlockWidth, size: q6KBlockBytes},
		delta: field(q6KDeltaStart, iqScaleBytes), scales: field(q6KScaleStart, q6KScaleBytes),
		high: field(q6KHighStart, q6KHighBytes), packed: field(q6KLowerStart, q6KLowerBytes),
	}
)
