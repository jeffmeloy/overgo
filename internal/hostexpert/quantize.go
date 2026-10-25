package hostexpert

import (
	"encoding/binary"
	"math"

	"overgo/internal/quant"
)

// The parts of a Q8_K block: its f32 scale, its 256 signed bytes and its 16
// sums of 16 bytes each; and of a Q8_0 block, its f16 scale and its 32 bytes.
const (
	q8KValuesAt = 4
	q8KSumsAt   = q8KValuesAt + superBlockValues
	q8KSumWidth = 16
	// q8KSumBytes is a sum's size, an int16.
	q8KSumBytes = 2
	q80ValuesAt = 2
	// q8Limit is the largest magnitude a quantized activation takes.
	q8Limit = 127
	// roundingMagic is ggml's nearest_int addend, 1.5 * 2^23: a float32
	// below 2^22 in magnitude added to it rounds to an integer, half to even,
	// held in the sum's low mantissa bits.
	roundingMagic = 12582912
	mantissaMask  = 0x007fffff
	mantissaBias  = 0x00400000
)

// quantizeQ8K quantizes values, a whole number of 256-value blocks, into
// blocks as ggml's quantize_row_q8_K_ref does: each block's scale is its
// first value of the largest magnitude over -127, every value is rounded to
// the nearest integer of it over the scale, half to even, at most 127, and
// the sums of each 16 are kept. A block of zeros quantizes to zeros.
func quantizeQ8K(values []float32, blocks []byte) error {
	if len(values)%superBlockValues != 0 {
		return errDepth(len(values), superBlockValues)
	}
	if len(blocks) != len(values)/superBlockValues*q8KBlockBytes {
		return errShape
	}
	for block := range len(values) / superBlockValues {
		source := values[block*superBlockValues : (block+1)*superBlockValues]
		target := blocks[block*q8KBlockBytes : (block+1)*q8KBlockBytes]
		var largest, magnitude float32
		for _, value := range source {
			if absolute := float32(math.Abs(float64(value))); absolute > magnitude {
				magnitude, largest = absolute, value
			}
		}
		if magnitude == 0 {
			clear(target)
			continue
		}
		inverse := -q8Limit / largest
		for index, value := range source {
			target[q8KValuesAt+index] = byte(int8(min(q8Limit, nearestInt(float32(inverse*value)))))
		}
		for sum := range superBlockValues / q8KSumWidth {
			total := int16(0)
			for _, value := range target[q8KValuesAt+sum*q8KSumWidth : q8KValuesAt+(sum+1)*q8KSumWidth] {
				total += int16(int8(value))
			}
			binary.LittleEndian.PutUint16(target[q8KSumsAt+q8KSumBytes*sum:], uint16(total))
		}
		binary.LittleEndian.PutUint32(target, math.Float32bits(1/inverse))
	}
	return nil
}

// nearestInt is ggml's nearest_int: value rounded to an integer, half to
// even, through the float32 sum with roundingMagic.
func nearestInt(value float32) int32 {
	sum := value + roundingMagic
	return int32(math.Float32bits(sum)&mantissaMask) - mantissaBias
}

// quantizeQ80 quantizes values, a whole number of 32-value blocks, into
// blocks as ggml's quantize_row_q8_0_ref does: each block's scale is its
// largest magnitude over 127, kept as f16, and every value is its product
// with the inverse of the f32 scale rounded half away from zero.
func quantizeQ80(values []float32, blocks []byte) error {
	if len(values)%blockValues != 0 {
		return errDepth(len(values), blockValues)
	}
	if len(blocks) != len(values)/blockValues*q80BlockBytes {
		return errShape
	}
	for block := range len(values) / blockValues {
		source := values[block*blockValues : (block+1)*blockValues]
		target := blocks[block*q80BlockBytes : (block+1)*q80BlockBytes]
		var magnitude float32
		for _, value := range source {
			magnitude = max(magnitude, float32(math.Abs(float64(value))))
		}
		scale := magnitude / q8Limit
		inverse := float32(0)
		if scale != 0 {
			inverse = 1 / scale
		}
		binary.LittleEndian.PutUint16(target, quant.Float32ToFloat16(scale))
		for index, value := range source {
			target[q80ValuesAt+index] = byte(int8(math.Round(float64(float32(value * inverse)))))
		}
	}
	return nil
}
