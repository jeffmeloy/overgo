package latentvideo

import (
	"slices"
	"testing"

	"overgo/internal/representation"
	"overgo/internal/tensor/dtype"
)

// TestPaddingEntersNoPromptRow holds what lets the encoder leave a prompt's
// padding out (EncodeTokensStreamed): a block gives the rows its mask keeps
// the same values, bit for bit, whether the rows after them are there or not.
// And it holds what lets the encoder multiply by a block's matrices as the
// checkpoint stores them: the block gives the values it gives of the same
// matrices widened beforehand.
func TestPaddingEntersNoPromptRow(t *testing.T) {
	const kept, padded, dim, heads, ffn, bucketCount, distance = 3, 9, 8, 2, 12, 8, 16
	state := uint32(1)
	values := func(count int) []float32 {
		out := make([]float32, count)
		for index := range out {
			state = state*1664525 + 1013904223
			out[index] = dtype.RoundBF16(float32(state>>8)/float32(1<<24) - 0.5)
		}
		return out
	}
	wide := func(count int) encoderMatrix { return encoderMatrix{wide: values(count)} }
	weights := encoderBlockWeights{
		Norm1: values(dim), Norm2: values(dim),
		Q: wide(dim * dim), K: wide(dim * dim), V: wide(dim * dim), O: wide(dim * dim),
		Gate: wide(ffn * dim), FC1: wide(ffn * dim), FC2: wide(dim * ffn),
		PosEmbedding: values(bucketCount * heads),
	}
	x, mask := values(padded*dim), make([]int, padded)
	for row := range kept {
		mask[row] = 1
	}
	block := func(weights encoderBlockWeights, rows int) []float32 {
		buckets, err := representation.RelativePositionBuckets(rows, rows, bucketCount, distance, true)
		if err != nil {
			t.Fatal(err)
		}
		out, err := encoderBlockForward(x[:rows*dim], mask[:rows], buckets, 1, rows, dim, heads, ffn, bucketCount, weights, 1e-6, true)
		if err != nil {
			t.Fatal(err)
		}
		return out[:kept*dim]
	}
	whole := block(weights, padded)
	if alone := block(weights, kept); !slices.Equal(whole, alone) {
		t.Fatalf("the rows a mask keeps differ with the rows after them left out:\n%v\n%v", whole, alone)
	}
	stored := func(matrix encoderMatrix) encoderMatrix {
		words := make([]uint16, len(matrix.wide))
		for index, value := range matrix.wide {
			words[index] = dtype.Float32ToBF16(value)
		}
		return encoderMatrix{stored: words}
	}
	asStored := weights
	asStored.Q, asStored.K, asStored.V, asStored.O = stored(weights.Q), stored(weights.K), stored(weights.V), stored(weights.O)
	asStored.Gate, asStored.FC1, asStored.FC2 = stored(weights.Gate), stored(weights.FC1), stored(weights.FC2)
	if unwidened := block(asStored, padded); !slices.Equal(whole, unwidened) {
		t.Fatalf("a block of matrices as stored differs from the block of them widened:\n%v\n%v", whole, unwidened)
	}
	if _, err := encoderBlockForward(x, mask, make([]int, padded*padded), 1, padded, dim, heads, ffn, bucketCount, asStored, 1e-6, false); err == nil {
		t.Fatal("matrices as stored were multiplied in the wider arithmetic")
	}
}
