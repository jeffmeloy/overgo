package hostexpert

import (
	"encoding/binary"
	"math"

	"overgo/internal/quant"
)

// The parts of the weight blocks: IQ4_XS's high sub-scale bits, low
// sub-scale nibbles and codes, and IQ4_NL's codes; a 32-value sub-block's
// 16 bytes of codes, each byte two values 16 apart; and the bias a 6-bit
// IQ4_XS sub-scale is read against.
const (
	iq4XSHighAt   = 2
	iq4XSLowAt    = 4
	iq4XSCodesAt  = 8
	iq4NLCodesAt  = 2
	subBlockBytes = 16
	subBlocks     = superBlockValues / blockValues
	scaleBias     = 32
	codeMask      = 0x0f
	codeShift     = 4
	highScaleBits = 2
	lowScaleBits  = 4
	highScaleMask = 0x30
	// lowScalesPerByte: two sub-blocks share a low-scale byte, a nibble each.
	lowScalesPerByte = 2
)

// tokensPerKernel is how many tokens a kernel call applies a row's decoded
// weights to.
const tokensPerKernel = 4

// iq4XSRows sets out[t][r] to the product of row r of weights, IQ4_XS rows
// of depth values one after another, with token t's Q8_K activations, for
// the rows from first to end.
func iq4XSRows(weights []byte, depth int, activations [][]byte, out [][]float32, first, end int) error {
	return rows(weights, depth, superBlockValues, iq4XSBlockBytes, q8KBlockBytes, activations, out, first, end, iq4XSKernel)
}

// iq4NLRows sets out[t][r] to the product of row r of weights, IQ4_NL rows
// of depth values one after another, with token t's Q8_0 activations, for
// the rows from first to end.
func iq4NLRows(weights []byte, depth int, activations [][]byte, out [][]float32, first, end int) error {
	return rows(weights, depth, blockValues, iq4NLBlockBytes, q80BlockBytes, activations, out, first, end, iq4NLKernel)
}

// rowKernel applies one row of blocks weight blocks to the activations of
// up to tokensPerKernel tokens, writing each token's product to results.
type rowKernel func(row []byte, blocks int, activations [][]byte, results []float32)

func rows(weights []byte, depth, values, weightBytes, activationBytes int, activations [][]byte, out [][]float32,
	first, end int, kernel rowKernel) error {
	if depth <= 0 || depth%values != 0 {
		return errDepth(depth, values)
	}
	blocks := depth / values
	rowBytes := blocks * weightBytes
	if first < 0 || end < first || end*rowBytes > len(weights) || len(activations) != len(out) {
		return errShape
	}
	for token, activation := range activations {
		if len(activation) < blocks*activationBytes || len(out[token]) < end {
			return errShape
		}
	}
	var results [tokensPerKernel]float32
	for row := first; row < end; row++ {
		weightsRow := weights[row*rowBytes : (row+1)*rowBytes]
		for start := 0; start < len(activations); start += tokensPerKernel {
			group := activations[start:min(start+tokensPerKernel, len(activations))]
			kernel(weightsRow, blocks, group, results[:len(group)])
			for index := range group {
				out[start+index][row] = results[index]
			}
		}
	}
	return nil
}

// iq4XSDot is ggml_vec_dot_iq4_xs_q8_K: each 32-value sub-block's integer
// sum of code values times activations, weighed by the block's two scales and
// the sub-block's signed 6-bit scale, added in order.
func iq4XSDot(row, activations []byte, blocks int) float32 {
	var sum float32
	for block := range blocks {
		weights := row[block*iq4XSBlockBytes : (block+1)*iq4XSBlockBytes]
		values := activations[block*q8KBlockBytes : (block+1)*q8KBlockBytes]
		scale := float32(quant.Float16ToFloat32(binary.LittleEndian.Uint16(weights)) *
			math.Float32frombits(binary.LittleEndian.Uint32(values)))
		high := binary.LittleEndian.Uint16(weights[iq4XSHighAt:])
		for sub := range subBlocks {
			low := weights[iq4XSLowAt+sub/lowScalesPerByte] >> (lowScaleBits * (sub % lowScalesPerByte)) & codeMask
			subScale := int(low) | int(high>>(highScaleBits*sub)<<lowScaleBits&highScaleMask)
			codes := weights[iq4XSCodesAt+sub*subBlockBytes:]
			quantized := values[q8KValuesAt+sub*blockValues:]
			sum += float32(float32(scale*float32(subScale-scaleBias)) * float32(codeDot(codes, quantized)))
		}
	}
	return sum
}

// iq4NLDot is ggml_vec_dot_iq4_nl_q8_0: each block's integer sum of code
// values times activations, weighed by the two blocks' scales, added in
// order.
func iq4NLDot(row, activations []byte, blocks int) float32 {
	var sum float32
	for block := range blocks {
		weights := row[block*iq4NLBlockBytes : (block+1)*iq4NLBlockBytes]
		values := activations[block*q80BlockBytes : (block+1)*q80BlockBytes]
		scale := float32(quant.Float16ToFloat32(binary.LittleEndian.Uint16(values)) *
			quant.Float16ToFloat32(binary.LittleEndian.Uint16(weights)))
		sum += float32(scale * float32(codeDot(weights[iq4NLCodesAt:], values[q80ValuesAt:])))
	}
	return sum
}

// codeDot is a 32-value sub-block's integer product: 16 bytes of codes, the
// low nibbles standing for values 0 to 15 and the high ones for 16 to 31,
// against 32 signed activations.
func codeDot(codes, activations []byte) int32 {
	var low, high int32
	for index := range subBlockBytes {
		low += int32(int8(activations[index])) * int32(iq4Codebook[codes[index]&codeMask])
		high += int32(int8(activations[index+subBlockBytes])) * int32(iq4Codebook[codes[index]>>codeShift])
	}
	return low + high
}

// scalarKernel applies dot to each token in turn, the kernel where the CPU
// has no AVX2 and the reference the AVX2 kernels are held to.
func scalarKernel(dot func(row, activations []byte, blocks int) float32) rowKernel {
	return func(row []byte, blocks int, activations [][]byte, results []float32) {
		for token, activation := range activations {
			results[token] = dot(row, activation, blocks)
		}
	}
}
