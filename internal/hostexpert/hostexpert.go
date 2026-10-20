// Package hostexpert computes routed-expert rows in host memory on the CPU,
// as Strata computes the experts its card does not hold (its
// src/kernels/cpu): a row's weights in their quantized format against
// activations quantized as ggml quantizes them, Q8_K for the 256-value block
// formats and Q8_0 for the 32-value ones.
//
// The formulas are ggml's (ggml-cpu/quants.c): ggml_vec_dot_iq4_xs_q8_K and
// ggml_vec_dot_iq4_nl_q8_0, whose integer sums are exact and whose scales
// multiply in float. The AVX2 kernels decode a 32-value chunk of weights once
// and apply it to several tokens, Strata's iq_avx2 scheme; they sum the same
// integers and order the float products otherwise, so they agree with the
// formulas to rounding.
package hostexpert

import (
	"encoding/binary"
	"errors"
	"fmt"

	"overgo/internal/quant"
	"overgo/internal/tensor/dtype"
)

// The values of a 256-value block format's block and of a 32-value one's.
const (
	superBlockValues = 256
	blockValues      = 32
)

// iq4XSBlockBytes is an IQ4_XS block's size: 256 values as an f16 scale, 16
// bits of high sub-scale bits, 4 bytes of low ones and 128 bytes of codes.
const iq4XSBlockBytes = 136

// iq4NLBlockBytes is an IQ4_NL block's size: 32 values as an f16 scale and
// 16 bytes of codes.
const iq4NLBlockBytes = 18

// q8KBlockBytes is a Q8_K block's size: 256 activations as an f32 scale,
// 256 signed bytes and the 16 sums of each 16.
const q8KBlockBytes = 292

// q80BlockBytes is a Q8_0 block's size: 32 activations as an f16 scale and
// 32 signed bytes.
const q80BlockBytes = 34

// iq4Codes is how many values a 4-bit code of IQ4_NL and IQ4_XS stands for.
const iq4Codes = 16

// iq4Codebook is ggml's kvalues_iq4nl, the value each 4-bit code stands for,
// read from the quant package's decoding of the codes at scale one; the AVX2
// kernels look codes up in it.
var iq4Codebook = func() [iq4Codes]int8 {
	block := make([]byte, iq4NLBlockBytes)
	binary.LittleEndian.PutUint16(block, quant.Float32ToFloat16(1))
	for code := range iq4Codes {
		block[iq4NLCodesAt+code] = byte(code)
	}
	values, err := quant.Dequantize(dtype.IQ4NL, block, blockValues)
	if err != nil {
		panic(err)
	}
	var codebook [iq4Codes]int8
	for code := range codebook {
		codebook[code] = int8(values[code])
	}
	return codebook
}()

// errDepth reports a row depth that is no whole number of blocks.
func errDepth(depth, block int) error {
	return fmt.Errorf("hostexpert: a row of %d values is no whole number of %d-value blocks", depth, block)
}

var errShape = errors.New("hostexpert: the weights, activations and outputs disagree in size")
