//go:build windows

package executor

import (
	"math/rand/v2"
	"testing"

	"overgo/internal/quant"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// TestExecutorQuantizedTensorCoreMulMatMatchesReference: past the column
// floor, a quantized weight prefills through f16 staging and the
// tensor-core GEMM, and agrees with the F32 reference over the same
// dequantized weight for every type with a dequantize kernel. Below the
// floor the span kernels still serve, checked on the same graph.
func TestExecutorQuantizedTensorCoreMulMatMatchesReference(t *testing.T) {
	const (
		inner   = uint64(512)
		rows    = uint64(24)
		columns = uint64(96)
	)
	leftShape := tensor.MustShape(inner, rows)
	rightValue := patternedValue(tensor.MustShape(inner, columns), 7, 0.05, 0.02)
	rng := rand.New(rand.NewPCG(17, 23))
	for _, dataType := range []dtype.Type{dtype.Q8_0, dtype.Q4K, dtype.Q5K, dtype.Q6K} {
		t.Run(dataType.String(), func(t *testing.T) {
			var storage []byte
			if dataType == dtype.Q8_0 {
				var err error
				storage, err = quant.Quantize(dataType, patternedValue(leftShape, 11, 0.03, -0.1).Data)
				if err != nil {
					t.Fatal(err)
				}
			} else {
				// Every byte pattern is a valid K-quant block; random blocks
				// exercise the scale and minimum unpacking. The block's
				// half-precision super-scales are pinned small so the
				// dequantized weights stay of order one and the absolute
				// tolerance reads as a relative one.
				bytes, err := leftShape.Bytes(dataType)
				if err != nil {
					t.Fatal(err)
				}
				storage = make([]byte, bytes)
				for index := range storage {
					storage[index] = byte(rng.IntN(256))
				}
				blockBytes, scaleOffsets := kQuantBlockScales(dataType)
				for block := 0; block+blockBytes <= len(storage); block += blockBytes {
					for _, offset := range scaleOffsets {
						// f16 0x1000 is 2^-11.
						storage[block+offset], storage[block+offset+1] = 0x00, 0x10
					}
				}
			}
			dequantized, err := quant.Dequantize(dataType, storage, inner*rows)
			if err != nil {
				t.Fatal(err)
			}
			build := func(builder *tensor.Builder, left, right *tensor.Tensor) *tensor.Tensor {
				return builder.MulMat(left, right)
			}
			tensorCore := func(builder *tensor.Builder) {
				builder.SetMulMatCompute(tensor.MulMatComputeNativeTensorCore)
			}
			// The staged path rounds the dequantized weight and the
			// activation to f16 (eleven significant bits each) and
			// accumulates in F32; against the reference's F32 dot products
			// it holds the measured half-staged bound on weights of order
			// one. Below the floor the span kernels serve as before; they
			// quantize the activation to eight bits and are held by their
			// own tests.
			checkResidentBinaryGraphWithPolicy(t, dataType, leftShape, storage, dequantized, rightValue, build, accuracyHalfStaged, tensorCore)
		})
	}
}

// kQuantBlockScales: a K-quant block's byte size and the offsets of its
// half-precision super-scales (scale and, where present, minimum).
func kQuantBlockScales(dataType dtype.Type) (int, []int) {
	switch dataType {
	case dtype.Q4K:
		return 144, []int{0, 2}
	case dtype.Q5K:
		return 176, []int{0, 2}
	case dtype.Q6K:
		return 210, []int{208}
	}
	return 0, nil
}
