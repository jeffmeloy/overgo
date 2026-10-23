//go:build amd64

package hostexpert

import (
	"math"
	"math/rand/v2"
	"testing"

	"overgo/internal/quant"
	"overgo/internal/tensor/dtype"
)

// TestIQ3SRowsInWideLanes holds the IQ3_S kernels in 512-bit lanes to the formula, for one token and four, on a
// processor that runs them: Flash-Next's gate and up rows take them where it can, the grid lookups having bounded the
// AVX2 kernel at a quarter of IQ4_XS's rate.
func TestIQ3SRowsInWideLanes(t *testing.T) {
	if !avx512 {
		t.Skip("the processor or the system runs no 512-bit kernels")
	}
	const (
		depth, rows = 2560, 6
		// tolerance is the relative distance a sum's float order moves it.
		tolerance = 1e-5
	)
	draws := rand.New(rand.NewPCG(depth, rows))
	values := make([]float32, depth*rows)
	for index := range values {
		values[index] = float32(draws.NormFloat64()) * 0.02
	}
	weights, err := quant.Quantize(dtype.IQ3S, values)
	if err != nil {
		t.Fatal(err)
	}
	var activations [tokensPerKernel][]byte
	var pointers [tokensPerKernel]*byte
	for token := range activations {
		input := make([]float32, depth)
		for index := range input {
			input[index] = float32(draws.NormFloat64())
		}
		activations[token] = make([]byte, depth/superBlockValues*q8KBlockBytes)
		if err := quantizeQ8K(input, activations[token]); err != nil {
			t.Fatal(err)
		}
		pointers[token] = &activations[token][0]
	}
	decoded, err := quant.Dequantize(dtype.IQ3S, weights, uint64(len(values)))
	if err != nil {
		t.Fatal(err)
	}
	rowBytes, blocks := len(weights)/rows, depth/superBlockValues
	for row := range rows {
		weight := weights[row*rowBytes : (row+1)*rowBytes]
		var four [tokensPerKernel]float32
		iq3SDots4Z(&weight[0], blocks, &pointers, &four)
		for token, activation := range activations {
			formula := float64(iq3SDot(weight, activation, blocks))
			var magnitude float64
			for index, value := range dequantizeQ8K(activation) {
				magnitude += math.Abs(float64(decoded[row*depth+index]) * value)
			}
			bound := tolerance * magnitude
			if got := float64(four[token]); math.Abs(got-formula) > bound {
				t.Fatalf("row %d token %d of four: the kernel %v, the formula %v", row, token, got, formula)
			}
			if got := float64(iq3SDots1Z(&weight[0], blocks, &activation[0])); math.Abs(got-formula) > bound {
				t.Fatalf("row %d token %d alone: the kernel %v, the formula %v", row, token, got, formula)
			}
		}
	}
}
