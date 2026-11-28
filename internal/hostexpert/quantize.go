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

// quantizeQ8KBlock quantizes a block's 256 values as ggml's quantize_row_q8_K_ref does: the block's scale is its first
// value of the largest magnitude over -127, every value is rounded to the nearest integer of it over the scale, half to
// even, at most 127, and the sums of each 16 are kept. A block of zeros quantizes to zeros.
func quantizeQ8KBlock(source []float32, target []byte) {
	var largest, magnitude float32
	for _, value := range source {
		if absolute := math.Float32frombits(math.Float32bits(value) &^ float32Sign); absolute > magnitude {
			magnitude, largest = absolute, value
		}
	}
	if magnitude == 0 {
		clear(target)
		return
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

// splitParts is how many quantized rows a split row is (quantizeSplitQ8K): the values' and what their rounding left.
const splitParts = 2

// quantizeSplitQ8K quantizes values as Q8_K twice over into blocks, its two halves: the first half the values, each
// block as ggml quantizes it (quantizeQ8KBlock), the second what that rounding left of each, so a row's product with
// the two halves summed stands about 250 times nearer its product with the values than with the first half alone, at
// one more integer product a row: with the first alone, ggml's CPU arithmetic, Flash-Next's host experts stood sixty
// times the card's distance from the f32 reference (flashnext-projection-exact).
func quantizeSplitQ8K(values []float32, blocks []byte) error {
	if len(values)%superBlockValues != 0 {
		return errDepth(len(values), superBlockValues)
	}
	half := len(values) / superBlockValues * q8KBlockBytes
	if len(blocks) != splitParts*half {
		return errShape
	}
	var left [superBlockValues]float32
	for block := range len(values) / superBlockValues {
		source := values[block*superBlockValues : (block+1)*superBlockValues]
		first := blocks[block*q8KBlockBytes : (block+1)*q8KBlockBytes]
		quantizeQ8KBlock(source, first)
		scale := math.Float32frombits(binary.LittleEndian.Uint32(first))
		for index, value := range source {
			left[index] = value - scale*float32(int8(first[q8KValuesAt+index]))
		}
		quantizeQ8KBlock(left[:], blocks[half+block*q8KBlockBytes:half+(block+1)*q8KBlockBytes])
	}
	return nil
}

// quantizeSplitQ80 is quantizeSplitQ8K's split of 32-value Q8_0 blocks, a down row's activations: the first half of
// blocks the values (quantizeQ80), the second what that rounding left of each, read back through the first's f16 scale.
func quantizeSplitQ80(values []float32, blocks []byte) error {
	if len(values)%blockValues != 0 {
		return errDepth(len(values), blockValues)
	}
	half := len(values) / blockValues * q80BlockBytes
	if len(blocks) != splitParts*half {
		return errShape
	}
	if err := quantizeQ80(values, blocks[:half]); err != nil {
		return err
	}
	var left [blockValues]float32
	for block := range len(values) / blockValues {
		first := blocks[block*q80BlockBytes : (block+1)*q80BlockBytes]
		scale := quant.Float16ToFloat32(binary.LittleEndian.Uint16(first))
		for index, value := range values[block*blockValues : (block+1)*blockValues] {
			left[index] = value - scale*float32(int8(first[q80ValuesAt+index]))
		}
		if err := quantizeQ80(left[:], blocks[half+block*q80BlockBytes:half+(block+1)*q80BlockBytes]); err != nil {
			return err
		}
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
			magnitude = max(magnitude, math.Float32frombits(math.Float32bits(value)&^float32Sign))
		}
		scale := magnitude / q8Limit
		inverse := float32(0)
		if scale != 0 {
			inverse = 1 / scale
		}
		binary.LittleEndian.PutUint16(target, quant.Float32ToFloat16(scale))
		for index, value := range source {
			target[q80ValuesAt+index] = byte(int8(roundHalfAway(float64(value * inverse))))
		}
	}
	return nil
}

// float32Sign is a float32's sign bit.
const float32Sign = 1 << 31

// roundHalfAway is math.Round for a quantized value's magnitudes, far below 2^63: the value rounded to the nearest
// integer and a half away from zero, by the conversion to an integer the processor truncates in one instruction.
func roundHalfAway(value float64) float64 {
	whole := float64(int64(value))
	if part := value - whole; part >= roundHalf {
		whole++
	} else if part <= -roundHalf {
		whole--
	}
	return whole
}

// roundHalf is the fraction from which a value rounds away from zero.
const roundHalf = 0.5
