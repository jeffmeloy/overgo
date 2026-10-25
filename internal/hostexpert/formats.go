package hostexpert

import (
	"encoding/binary"
	"math"

	"overgo/internal/quant"
)

// iq3SBlockBytes is an IQ3_S block's size: 256 values as an f16 scale, 64
// bytes of grid indices' low bits, 8 of their high bits, 32 of signs and 4 of
// 4-bit scales, two 32-value sub-blocks a byte.
const iq3SBlockBytes = 110

// The parts of an IQ3_S block, and the widths a sub-block's indices, signs
// and grid entries take: eight 9-bit indices, each four magnitudes of the
// grid, a sign bit a value, and the step of a sub-block's odd scale 2s+1.
const (
	iq3SIndicesAt     = 2
	iq3SHighAt        = 66
	iq3SSignsAt       = 74
	iq3SScalesAt      = 106
	iq3SSubIndices    = 8
	iq3SSubSignBytes  = 4
	iq3SGridValues    = 4
	iq3SIndexHighBit  = 0x100
	iq3SIndexHighSpan = 8
	signBits          = 8
	iq3SScaleStep     = 2
)

// iq3SGrid is ggml's iq3s_grid, from the quant package.
var iq3SGrid = quant.IQ3SGrid()

// iq3SRows sets out[t][r] to the product of row r of weights, IQ3_S rows of
// depth values one after another, with token t's Q8_K activations, for the
// rows from first to end.
func iq3SRows(weights []byte, depth int, activations [][]byte, out [][]float32, first, end int) error {
	return rows(weights, depth, superBlockValues, iq3SBlockBytes, q8KBlockBytes, activations, out, first, end, iq3SKernel)
}

// q80Rows sets out[t][r] to the product of row r of weights, Q8_0 rows of
// depth values one after another, with token t's Q8_0 activations, for the
// rows from first to end.
func q80Rows(weights []byte, depth int, activations [][]byte, out [][]float32, first, end int) error {
	return rows(weights, depth, blockValues, q80BlockBytes, q80BlockBytes, activations, out, first, end, q80Kernel)
}

// iq3SDot is ggml_vec_dot_iq3_s_q8_K: each 32-value sub-block's integer sum
// of grid magnitudes, their signs and the activations, times the sub-block's
// odd scale 2s+1, summed over the block in integers and weighed by the
// block's two scales once.
func iq3SDot(row, activations []byte, blocks int) float32 {
	var sum float32
	for block := range blocks {
		weights := row[block*iq3SBlockBytes : (block+1)*iq3SBlockBytes]
		values := activations[block*q8KBlockBytes : (block+1)*q8KBlockBytes]
		scale := float32(quant.Float16ToFloat32(binary.LittleEndian.Uint16(weights)) * f32At(values))
		var total int32
		for sub := range subBlocks {
			subScale := int32(weights[iq3SScalesAt+sub/lowScalesPerByte]>>(lowScaleBits*(sub%lowScalesPerByte))&codeMask)*iq3SScaleStep + 1
			high := uint16(weights[iq3SHighAt+sub])
			signs := weights[iq3SSignsAt+sub*iq3SSubSignBytes:]
			quantized := values[q8KValuesAt+sub*blockValues:]
			var subTotal int32
			for index := range iq3SSubIndices {
				grid := iq3SGrid[uint16(weights[iq3SIndicesAt+sub*iq3SSubIndices+index])|high<<(iq3SIndexHighSpan-index)&iq3SIndexHighBit]
				signByte := signs[index/(signBits/iq3SGridValues)]
				for lane := range iq3SGridValues {
					value := int32(byte(grid >> (signBits * lane)))
					if signByte&(1<<(index%(signBits/iq3SGridValues)*iq3SGridValues+lane)) != 0 {
						value = -value
					}
					subTotal += value * int32(int8(quantized[index*iq3SGridValues+lane]))
				}
			}
			total += subTotal * subScale
		}
		sum += float32(scale * float32(total))
	}
	return sum
}

// q80Dot is ggml_vec_dot_q8_0_q8_0: each block's integer sum of weights times
// activations, weighed by the two blocks' scales, added in order.
func q80Dot(row, activations []byte, blocks int) float32 {
	var sum float32
	for block := range blocks {
		weights := row[block*q80BlockBytes : (block+1)*q80BlockBytes]
		values := activations[block*q80BlockBytes : (block+1)*q80BlockBytes]
		var total int32
		for index := range blockValues {
			total += int32(int8(weights[q80ValuesAt+index])) * int32(int8(values[q80ValuesAt+index]))
		}
		scale := float32(quant.Float16ToFloat32(binary.LittleEndian.Uint16(weights)) * quant.Float16ToFloat32(binary.LittleEndian.Uint16(values)))
		sum += float32(float32(total) * scale)
	}
	return sum
}

// f32At reads the little-endian f32 at the start of bytes, a Q8_K block's
// scale.
func f32At(bytes []byte) float32 {
	return math.Float32frombits(binary.LittleEndian.Uint32(bytes))
}
