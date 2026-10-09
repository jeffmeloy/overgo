package latentvideo

import (
	"slices"
	"testing"

	"overgo/internal/representation"
)

// TestPaddingEntersNoPromptRow holds what lets the encoder leave a prompt's
// padding out (EncodeTokensStreamed): a block gives the rows its mask keeps
// the same values, bit for bit, whether the rows after them are there or not.
func TestPaddingEntersNoPromptRow(t *testing.T) {
	const kept, padded, dim, heads, ffn, bucketCount, distance = 3, 9, 8, 2, 12, 8, 16
	state := uint32(1)
	values := func(count int) []float32 {
		out := make([]float32, count)
		for index := range out {
			state = state*1664525 + 1013904223
			out[index] = float32(state>>8)/float32(1<<24) - 0.5
		}
		return out
	}
	weights := encoderBlockWeights{
		Norm1: values(dim), Norm2: values(dim),
		Q: values(dim * dim), K: values(dim * dim), V: values(dim * dim), O: values(dim * dim),
		Gate: values(ffn * dim), FC1: values(ffn * dim), FC2: values(dim * ffn),
		PosEmbedding: values(bucketCount * heads),
	}
	x, mask := values(padded*dim), make([]int, padded)
	for row := range kept {
		mask[row] = 1
	}
	block := func(rows int) []float32 {
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
	if whole, alone := block(padded), block(kept); !slices.Equal(whole, alone) {
		t.Fatalf("the rows a mask keeps differ with the rows after them left out:\n%v\n%v", whole, alone)
	}
}
