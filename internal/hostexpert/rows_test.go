package hostexpert

import (
	"encoding/binary"
	"math"
	"math/rand/v2"
	"testing"

	"overgo/internal/quant"
	"overgo/internal/tensor/dtype"
)

// TestCodebookIsGGMLs holds iq4Codebook, the quant package's decoding of the
// codes at scale one, to ggml's kvalues_iq4nl.
func TestCodebookIsGGMLs(t *testing.T) {
	if ggml := [iq4Codes]int8{-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113}; iq4Codebook != ggml {
		t.Fatalf("codebook %v, ggml's %v", iq4Codebook, ggml)
	}
}

// TestQuantizersRoundAsGGML holds the activation quantizers to ggml's: Q8_K
// scales by the first value of the largest magnitude over -127 and rounds
// half to even, its sums the sums of each 16; Q8_0 scales by the largest
// magnitude over 127 and rounds half away from zero; a block of zeros
// quantizes to zeros.
func TestQuantizersRoundAsGGML(t *testing.T) {
	values := make([]float32, superBlockValues)
	values[0], values[1], values[2], values[3] = -127, 2.5, 3.5, -127
	blocks := make([]byte, q8KBlockBytes)
	if err := quantizeQ8K(values, blocks); err != nil {
		t.Fatal(err)
	}
	// The scale is 1/(-127/-127) = 1, and the first of the two largest magnitudes is -127.
	if scale := math.Float32frombits(binary.LittleEndian.Uint32(blocks)); scale != 1 {
		t.Fatalf("Q8_K scale %v, want 1", scale)
	}
	if got := [4]int8{int8(blocks[4]), int8(blocks[5]), int8(blocks[6]), int8(blocks[7])}; got != [4]int8{-127, 2, 4, -127} {
		t.Fatalf("Q8_K values %v, want -127 2 4 -127: halves to even", got)
	}
	if sum := int16(binary.LittleEndian.Uint16(blocks[q8KSumsAt:])); sum != -127+2+4-127 {
		t.Fatalf("Q8_K first sum %d, want %d", sum, -127+2+4-127)
	}
	short := make([]float32, blockValues)
	short[0], short[1], short[2] = 127, 2.5, -3.5
	block := make([]byte, q80BlockBytes)
	if err := quantizeQ80(short, block); err != nil {
		t.Fatal(err)
	}
	if got := [3]int8{int8(block[2]), int8(block[3]), int8(block[4])}; got != [3]int8{127, 3, -4} {
		t.Fatalf("Q8_0 values %v, want 127 3 -4: halves away from zero", got)
	}
	zeros := make([]byte, q8KBlockBytes)
	for index := range zeros {
		zeros[index] = 0xff
	}
	if err := quantizeQ8K(make([]float32, superBlockValues), zeros); err != nil {
		t.Fatal(err)
	}
	for index, value := range zeros {
		if value != 0 {
			t.Fatalf("a block of zeros keeps byte %d as %#x", index, value)
		}
	}
}

// TestRowsMatchTheFormulas holds iq4XSRows and iq4NLRows, the AVX2 kernels
// where the processor runs them, to ggml's formulas for 1 to 9 tokens over
// rows of Flash-Next's 2,560 values, and the formulas to the float64 sums of
// the same quantized weights and activations.
func TestRowsMatchTheFormulas(t *testing.T) {
	const (
		depth = 2560
		rows  = 6
		// tolerance is the relative distance a sum's float order moves it.
		tolerance = 1e-5
	)
	draws := rand.New(rand.NewPCG(depth, rows))
	for _, format := range []struct {
		name         string
		weights      dtype.Type
		weightValues int
		activation   func([]float32, []byte) error
		actBytes     int
		rows         func([]byte, int, [][]byte, [][]float32, int, int) error
		dot          func([]byte, []byte, int) float32
		dequantize   func([]byte) []float64
	}{
		{"iq4_xs", dtype.IQ4XS, superBlockValues, quantizeQ8K, q8KBlockBytes, iq4XSRows, iq4XSDot, dequantizeQ8K},
		{"iq4_nl", dtype.IQ4NL, blockValues, quantizeQ80, q80BlockBytes, iq4NLRows, iq4NLDot, dequantizeQ80},
	} {
		values := make([]float32, depth*rows)
		for index := range values {
			values[index] = float32(draws.NormFloat64()) * 0.02
		}
		weights, err := quant.Quantize(format.weights, values)
		if err != nil {
			t.Fatal(err)
		}
		decoded, err := quant.Dequantize(format.weights, weights, uint64(len(values)))
		if err != nil {
			t.Fatal(err)
		}
		for _, tokens := range []int{1, 2, 3, 4, 5, 9} {
			activations := make([][]byte, tokens)
			out := make([][]float32, tokens)
			for token := range tokens {
				input := make([]float32, depth)
				for index := range input {
					input[index] = float32(draws.NormFloat64())
				}
				activations[token] = make([]byte, depth/format.weightValues*format.actBytes)
				if err := format.activation(input, activations[token]); err != nil {
					t.Fatal(err)
				}
				out[token] = make([]float32, rows)
			}
			if err := format.rows(weights, depth, activations, out, 0, rows); err != nil {
				t.Fatal(err)
			}
			rowBytes := len(weights) / rows
			for token := range tokens {
				reference := format.dequantize(activations[token])
				for row := range rows {
					formula := format.dot(weights[row*rowBytes:(row+1)*rowBytes], activations[token], depth/format.weightValues)
					var exact, magnitude float64
					for index := range depth {
						product := float64(decoded[row*depth+index]) * reference[index]
						exact += product
						magnitude += math.Abs(product)
					}
					if math.Abs(float64(formula)-exact) > tolerance*magnitude {
						t.Fatalf("%s token %d row %d: the formula %v, the float64 sum %v", format.name, token, row, formula, exact)
					}
					if got := out[token][row]; math.Abs(float64(got-formula)) > tolerance*magnitude {
						t.Fatalf("%s %d tokens, token %d row %d: the kernel %v (AVX2 %t), the formula %v", format.name, tokens, token, row, got, avx2, formula)
					}
				}
			}
		}
	}
}

// dequantizeQ8K and dequantizeQ80 are the values quantized activations stand for.
func dequantizeQ8K(blocks []byte) []float64 {
	var values []float64
	for block := 0; block < len(blocks); block += q8KBlockBytes {
		scale := float64(math.Float32frombits(binary.LittleEndian.Uint32(blocks[block:])))
		for index := range superBlockValues {
			values = append(values, scale*float64(int8(blocks[block+q8KValuesAt+index])))
		}
	}
	return values
}

func dequantizeQ80(blocks []byte) []float64 {
	var values []float64
	for block := 0; block < len(blocks); block += q80BlockBytes {
		scale := float64(quant.Float16ToFloat32(binary.LittleEndian.Uint16(blocks[block:])))
		for index := range blockValues {
			values = append(values, scale*float64(int8(blocks[block+q80ValuesAt+index])))
		}
	}
	return values
}

// BenchmarkIQ4XSRows times a Flash-Next expert matrix's 512 rows of 2,560
// values against one token and four.
func BenchmarkIQ4XSRows(b *testing.B) {
	const depth, rows = 2560, 512
	draws := rand.New(rand.NewPCG(depth, rows))
	values := make([]float32, depth*rows)
	for index := range values {
		values[index] = float32(draws.NormFloat64()) * 0.02
	}
	weights, err := quant.Quantize(dtype.IQ4XS, values)
	if err != nil {
		b.Fatal(err)
	}
	for _, tokens := range []int{1, 4} {
		activations := make([][]byte, tokens)
		out := make([][]float32, tokens)
		for token := range tokens {
			input := make([]float32, depth)
			for index := range input {
				input[index] = float32(draws.NormFloat64())
			}
			activations[token] = make([]byte, depth/superBlockValues*q8KBlockBytes)
			if err := quantizeQ8K(input, activations[token]); err != nil {
				b.Fatal(err)
			}
			out[token] = make([]float32, rows)
		}
		b.Run(map[int]string{1: "one", 4: "four"}[tokens], func(b *testing.B) {
			b.SetBytes(int64(len(weights)))
			for b.Loop() {
				if err := iq4XSRows(weights, depth, activations, out, 0, rows); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
