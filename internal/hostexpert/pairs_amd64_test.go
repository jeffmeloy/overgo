//go:build amd64

package hostexpert

import (
	"math"
	"math/rand/v2"
	"testing"

	"overgo/internal/quant"
	"overgo/internal/tensor/dtype"
	"overgo/internal/testcheck"
)

// TestATwoRowGroupMatchesTheFourRowKernel holds each format's two-row kernel, which a split decode token's two rows
// take (host-expert-pairs), to the four-row kernel's products of the same two rows, to the bit, in each format and
// lane width the processor runs: the pool's layer and its serial one group a token's rows differently, and both must
// give the same products. A decode token's expert had taken the four-row kernel with two repeats, 71 us where its one
// row had taken 54.
func TestATwoRowGroupMatchesTheFourRowKernel(t *testing.T) {
	t.Parallel()
	if !avx2 {
		t.Skip("the processor or the system runs no AVX2 kernels")
	}
	type kernels = func(*byte, int, *[tokensPerKernel]*byte, *[tokensPerKernel]float32)
	formats := []struct {
		name      string
		format    dtype.Type
		depth     int
		split     func([]float32, []byte) error
		splitSize int
		two, four kernels
	}{
		{"iq4_xs", dtype.IQ4XS, testHidden, quantizeSplitQ8K, Q8KBytes(testHidden), iq4XSDots2, iq4XSDots4},
		{"iq3_s", dtype.IQ3S, testHidden, quantizeSplitQ8K, Q8KBytes(testHidden), iq3SDots2, iq3SDots4},
		{"iq4_nl", dtype.IQ4NL, testIntermediate, quantizeSplitQ80, splitParts * testIntermediate / blockValues * q80BlockBytes, iq4NLDots2, iq4NLDots4},
		{"q8_0", dtype.Q8_0, testIntermediate, quantizeSplitQ80, splitParts * testIntermediate / blockValues * q80BlockBytes, q80Dots2, q80Dots4},
	}
	if avx512 {
		formats = append(formats, formats[1])
		formats[len(formats)-1].name, formats[len(formats)-1].two, formats[len(formats)-1].four = "iq3_s in 512-bit lanes", iq3SDots2Z, iq3SDots4Z
	}
	draws := rand.New(rand.NewPCG(testHidden, testIntermediate))
	for _, format := range formats {
		values := make([]float32, format.depth)
		for index := range values {
			values[index] = float32(draws.NormFloat64()) * 0.02
		}
		row := testcheck.Must(quant.Quantize(format.format, values))(t)
		input := make([]float32, format.depth)
		for index := range input {
			input[index] = float32(draws.NormFloat64())
		}
		split := make([]byte, format.splitSize)
		testcheck.NoError(t, format.split(input, split))
		half := len(split) / splitParts
		pointers := [tokensPerKernel]*byte{&split[0], &split[half], &split[0], &split[0]}
		traits, _ := format.format.Traits()
		blocks := format.depth / int(traits.BlockSize)
		var two, four [tokensPerKernel]float32
		format.two(&row[0], blocks, &pointers, &two)
		format.four(&row[0], blocks, &pointers, &four)
		for part := range splitParts {
			if math.Float32bits(two[part]) != math.Float32bits(four[part]) {
				t.Errorf("%s: the two-row kernel's product of row %d is %g, the four-row kernel's %g", format.name, part, two[part], four[part])
			}
		}
	}
}
