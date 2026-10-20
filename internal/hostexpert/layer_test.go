package hostexpert

import (
	"encoding/binary"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"overgo/internal/quant"
	"overgo/internal/tensor/dtype"
)

// Flash-Next's expert widths: a token's 2,560 values and an expert's 640.
const (
	testHidden       = 2560
	testIntermediate = 640
)

// The formats of Flash-Next's expert layers: most IQ3_S gate and up with
// IQ4_NL down, one IQ4_XS gate and up, five Q8_0 down.
var testFormats = []Formats{
	{dtype.IQ3S, dtype.IQ3S, dtype.IQ4NL},
	{dtype.IQ4XS, dtype.IQ4XS, dtype.Q8_0},
}

func silu(gate, up float32) float32 {
	return float32(gate/float32(1+math.Exp(-float64(gate)))) * up
}

// testExperts makes len(counts) experts stored as formats with random
// weights, and the routed tokens' Q8_K activations, counts[e] tokens to
// expert e.
func testExperts(t testing.TB, formats Formats, counts []int) []Expert {
	t.Helper()
	draws := rand.New(rand.NewPCG(uint64(len(counts)), uint64(formats.Gate)))
	// encode draws random blocks of format, every byte of which is a valid code, index, sign or scale, and sets
	// each block's f16 scale to scale: quantizing random weights as IQ3_S takes minutes at these widths.
	encode := func(format dtype.Type, values int, scale float32) []byte {
		traits, _ := format.Traits()
		blocks := make([]byte, uint64(values)/traits.BlockSize*traits.TypeSize)
		for index := range blocks {
			blocks[index] = byte(draws.Uint32())
			if format == dtype.Q8_0 && int8(blocks[index]) == math.MinInt8 {
				blocks[index] = 0
			}
		}
		for at := 0; at < len(blocks); at += int(traits.TypeSize) {
			binary.LittleEndian.PutUint16(blocks[at:], quant.Float32ToFloat16(scale))
		}
		return blocks
	}
	experts := make([]Expert, len(counts))
	for index, count := range counts {
		work := Expert{
			Gate: encode(formats.Gate, testHidden*testIntermediate, 0.001),
			Up:   encode(formats.Up, testHidden*testIntermediate, 0.001),
			Down: encode(formats.Down, testIntermediate*testHidden, 0.002),
		}
		for range count {
			input := make([]float32, testHidden)
			for value := range input {
				input[value] = float32(draws.NormFloat64())
			}
			activation := make([]byte, Q8KBytes(testHidden))
			if err := quantizeQ8K(input, activation); err != nil {
				t.Fatal(err)
			}
			work.Activations = append(work.Activations, activation)
			work.Out = append(work.Out, make([]float32, testHidden))
		}
		experts[index] = work
	}
	return experts
}

// TestLayerMatchesTheSerialLayer holds a layer of five experts with 1 to 7
// tokens each, in each of Flash-Next's formats, spread across eight workers,
// to the same rows computed one after another: each row's arithmetic is the
// same, so the bits are.
func TestLayerMatchesTheSerialLayer(t *testing.T) {
	workers := NewPool(8)
	defer workers.Close()
	for _, formats := range testFormats {
		experts := testExperts(t, formats, []int{1, 2, 3, 4, 7})
		if err := workers.Layer(experts, formats, testHidden, testIntermediate, silu); err != nil {
			t.Fatal(err)
		}
		for index, work := range experts {
			tokens := len(work.Activations)
			gate, up := make([][]float32, tokens), make([][]float32, tokens)
			quantized, out := make([][]byte, tokens), make([][]float32, tokens)
			for token := range tokens {
				gate[token], up[token] = make([]float32, testIntermediate), make([]float32, testIntermediate)
				quantized[token] = make([]byte, testIntermediate/blockValues*q80BlockBytes)
				out[token] = make([]float32, testHidden)
			}
			if err := rowsOf[formats.Gate](work.Gate, testHidden, work.Activations, gate, 0, testIntermediate); err != nil {
				t.Fatal(err)
			}
			if err := rowsOf[formats.Up](work.Up, testHidden, work.Activations, up, 0, testIntermediate); err != nil {
				t.Fatal(err)
			}
			for token := range tokens {
				for row := range testIntermediate {
					gate[token][row] = silu(gate[token][row], up[token][row])
				}
				if err := quantizeQ80(gate[token], quantized[token]); err != nil {
					t.Fatal(err)
				}
			}
			if err := rowsOf[formats.Down](work.Down, testIntermediate, quantized, out, 0, testHidden); err != nil {
				t.Fatal(err)
			}
			for token := range tokens {
				if !slices.Equal(work.Out[token], out[token]) {
					t.Fatalf("%v expert %d token %d: the pool's products differ from the serial ones", formats, index, token)
				}
			}
		}
	}
}

// BenchmarkLayer times Flash-Next's common expert layer, IQ3_S gate and up
// with IQ4_NL down, for a prompt's host experts, 120 of 18 tokens each, and a
// decode token's, 3 of one, across 16 and 32 workers.
func BenchmarkLayer(b *testing.B) {
	for _, shape := range []struct {
		name    string
		experts int
		tokens  int
	}{{"prompt", 120, 18}, {"decode", 3, 1}} {
		counts := make([]int, shape.experts)
		for index := range counts {
			counts[index] = shape.tokens
		}
		experts := testExperts(b, testFormats[0], counts)
		for _, workers := range []int{16, 32} {
			b.Run(shape.name+"/"+map[int]string{16: "16", 32: "32"}[workers], func(b *testing.B) {
				workers := NewPool(workers)
				defer workers.Close()
				for b.Loop() {
					if err := workers.Layer(experts, testFormats[0], testHidden, testIntermediate, silu); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}
