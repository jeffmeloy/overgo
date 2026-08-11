package executor

import (
	"encoding/binary"
	"math"
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// fp8PatternByte: deterministic finite e4m3 byte with broad coverage (both
// signs, mantissas 0..7, exponents 0..9 -- includes subnormals, excludes the
// sole NaN code 0x7f/0xff and the largest exponents so magnitudes stay moderate).
func fp8PatternByte(index int) byte {
	value := (index*61 + 17) & 0xff
	exponent := (value >> 3) & 0x0f
	if exponent > 9 {
		exponent %= 10
	}
	return byte((value & 0x80) | (exponent << 3) | (value & 0x07))
}

// TestExecutorFP8MulMatMatchesReference: native-FP8(E4M3) residency serves matmul
// weights at 1 byte + a per-output-row F32 scale carried in the same resident
// buffer ([rows*inner e4m3 | rows*4 scale]), through the SAME native-dtype
// mechanism as F16/BF16. Decode (right rows == 1) reads the e4m3 quads directly
// and folds scale[row] once; prefill (right rows > 1) upconverts the weight to
// F32 with the per-row scale folded in and runs the identical SGEMM. Both must
// match the exact e4m3-dequant-times-scale F32 reference; greedy argmax exact.
func TestExecutorFP8MulMatMatchesReference(t *testing.T) {
	const inner, rows = 64, 19 // inner % 4 == 0 for the packed-quad decode
	leftShape := tensor.MustShape(inner, rows)

	weightBytes := make([]byte, rows*inner)
	scales := make([]float32, rows)
	dequantized := make([]float32, rows*inner)
	for row := 0; row < rows; row++ {
		scales[row] = 0.125 * float32(1+row%7) // positive, varied per output row
		for column := 0; column < inner; column++ {
			b := fp8PatternByte(row*inner + column)
			weightBytes[row*inner+column] = b
			dequantized[row*inner+column] = dtype.F8E4M3ToFloat32(b) * scales[row]
		}
	}
	// Combined resident buffer: all e4m3 bytes, then the per-row F32 scales.
	storage := make([]byte, rows*inner+rows*4)
	copy(storage, weightBytes)
	for row := 0; row < rows; row++ {
		binary.LittleEndian.PutUint32(storage[rows*inner+row*4:], math.Float32bits(scales[row]))
	}

	checkResidentMulMat(t, dtype.F8E4M3, leftShape, storage, dequantized)
}
