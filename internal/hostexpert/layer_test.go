package hostexpert

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"overgo/internal/quant"
	"overgo/internal/tensor/dtype"
)

// Flash-Next's expert widths: a token's 2,560 values and an expert's 512.
const (
	testHidden       = 2560
	testIntermediate = 512
)

func silu(gate, up float32) float32 {
	return float32(gate/float32(1+math.Exp(-float64(gate)))) * up
}

// testExperts makes counts experts with random weights and the routed
// tokens' Q8_K activations, counts[e] tokens to expert e.
func testExperts(t testing.TB, counts []int) []expertWork {
	t.Helper()
	draws := rand.New(rand.NewPCG(uint64(len(counts)), testHidden))
	encode := func(format dtype.Type, values int, scale float64) []byte {
		weights := make([]float32, values)
		for index := range weights {
			weights[index] = float32(draws.NormFloat64() * scale)
		}
		encoded, err := quant.Quantize(format, weights)
		if err != nil {
			t.Fatal(err)
		}
		return encoded
	}
	experts := make([]expertWork, len(counts))
	for index, count := range counts {
		work := expertWork{
			gate: encode(dtype.IQ4XS, testHidden*testIntermediate, 0.02),
			up:   encode(dtype.IQ4XS, testHidden*testIntermediate, 0.02),
			down: encode(dtype.IQ4NL, testIntermediate*testHidden, 0.04),
		}
		for range count {
			input := make([]float32, testHidden)
			for value := range input {
				input[value] = float32(draws.NormFloat64())
			}
			activation := make([]byte, testHidden/superBlockValues*q8KBlockBytes)
			if err := quantizeQ8K(input, activation); err != nil {
				t.Fatal(err)
			}
			work.activations = append(work.activations, activation)
			work.out = append(work.out, make([]float32, testHidden))
		}
		experts[index] = work
	}
	return experts
}

// TestLayerMatchesTheSerialLayer holds a layer of five experts with 1 to 7
// tokens each, spread across eight workers, to the same rows computed one
// after another: each row's arithmetic is the same, so the bits are.
func TestLayerMatchesTheSerialLayer(t *testing.T) {
	experts := testExperts(t, []int{1, 2, 3, 4, 7})
	workers := newPool(8)
	defer workers.close()
	if err := workers.layer(experts, testHidden, testIntermediate, silu, &layerScratch{}); err != nil {
		t.Fatal(err)
	}
	for index, work := range experts {
		tokens := len(work.activations)
		gate, up := make([][]float32, tokens), make([][]float32, tokens)
		quantized, out := make([][]byte, tokens), make([][]float32, tokens)
		for token := range tokens {
			gate[token], up[token] = make([]float32, testIntermediate), make([]float32, testIntermediate)
			quantized[token] = make([]byte, testIntermediate/blockValues*q80BlockBytes)
			out[token] = make([]float32, testHidden)
		}
		if err := iq4XSRows(work.gate, testHidden, work.activations, gate, 0, testIntermediate); err != nil {
			t.Fatal(err)
		}
		if err := iq4XSRows(work.up, testHidden, work.activations, up, 0, testIntermediate); err != nil {
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
		if err := iq4NLRows(work.down, testIntermediate, quantized, out, 0, testHidden); err != nil {
			t.Fatal(err)
		}
		for token := range tokens {
			if !slices.Equal(work.out[token], out[token]) {
				t.Fatalf("expert %d token %d: the pool's products differ from the serial ones", index, token)
			}
		}
	}
}

// BenchmarkLayer times a prompt's host experts, 120 of 18 tokens each, and a
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
		experts := testExperts(b, counts)
		for _, workers := range []int{16, 32} {
			b.Run(shape.name+"/"+map[int]string{16: "16", 32: "32"}[workers], func(b *testing.B) {
				workers := newPool(workers)
				defer workers.close()
				scratch := &layerScratch{}
				for b.Loop() {
					if err := workers.layer(experts, testHidden, testIntermediate, silu, scratch); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}
