package server

import (
	"math"
	"testing"

	"overgo/internal/inference"
	"overgo/internal/tensor/reference"
)

// TestAttentionWeightsHostMath locks the host recomputation: the column-major
// [dim, head, token] indexing, the causal mask, per-row softmax normalization,
// and the GQA head→kv-head mapping. Geometry: headDim=2, 2 query heads, 1 kv
// head (group=2), 2 tokens, scale=1.
func TestAttentionWeightsHostMath(t *testing.T) {
	// Query Data index = d + head*headDim + token*headDim*heads (headDim=2, heads=2).
	// head0 q=(1,0) both tokens; head1 q=(0,1) both tokens.
	query := []float32{
		1, 0, 0, 1, // token0: head0=(1,0), head1=(0,1)
		1, 0, 0, 1, // token1: head0=(1,0), head1=(0,1)
	}
	// Key Data index = d + kvHead*headDim + token*headDim*kvHeads (headDim=2, kvHeads=1).
	// key0=(1,0), key1=(0,1).
	key := []float32{1, 0, 0, 1}

	capture := inference.AttentionCapture{
		Heads: 2, KVHeads: 1, HeadDim: 2, Scale: 1,
		Query: reference.Value{Data: query},
		Key:   reference.Value{Data: key},
	}
	weights, err := attentionWeights(capture, 2)
	if err != nil {
		t.Fatalf("attentionWeights: %v", err)
	}
	if len(weights) != 2 {
		t.Fatalf("heads = %d, want 2", len(weights))
	}

	const eps = 1e-4
	// Causality: query token 0 attends only to key 0 (weight 1), key 1 is zero.
	for head := range 2 {
		if math.Abs(weights[head][0][0]-1) > eps || weights[head][0][1] != 0 {
			t.Fatalf("head %d row0 = %v, want [1 0] (causal)", head, weights[head][0])
		}
	}
	// Every row is a distribution: sums to 1.
	for head := range 2 {
		for row := range 2 {
			sum := weights[head][row][0] + weights[head][row][1]
			if math.Abs(sum-1) > eps {
				t.Fatalf("head %d row %d sum = %v, want 1", head, row, sum)
			}
		}
	}
	// head0 q=(1,0): token1 logits [dot k0, dot k1] = [1,0] → softmax [e/(e+1), 1/(e+1)].
	wantHi := math.Exp(1) / (math.Exp(1) + 1)
	wantLo := 1 / (math.Exp(1) + 1)
	if math.Abs(weights[0][1][0]-wantHi) > eps || math.Abs(weights[0][1][1]-wantLo) > eps {
		t.Fatalf("head0 row1 = %v, want [%v %v]", weights[0][1], wantHi, wantLo)
	}
	// head1 q=(0,1): token1 logits [0,1] → the mirror. Both heads share kv head 0
	// (GQA group=2), so this also confirms the head→kv mapping.
	if math.Abs(weights[1][1][0]-wantLo) > eps || math.Abs(weights[1][1][1]-wantHi) > eps {
		t.Fatalf("head1 row1 = %v, want [%v %v]", weights[1][1], wantLo, wantHi)
	}
}

// TestAttentionWeightsRejectsMismatchedGeometry guards the length check that
// protects the index arithmetic from a capture whose data does not match its
// declared geometry.
func TestAttentionWeightsRejectsMismatchedGeometry(t *testing.T) {
	capture := inference.AttentionCapture{
		Heads: 2, KVHeads: 1, HeadDim: 2, Scale: 1,
		Query: reference.Value{Data: []float32{1, 0, 0}}, // wrong length
		Key:   reference.Value{Data: []float32{1, 0, 0, 1}},
	}
	if _, err := attentionWeights(capture, 2); err == nil {
		t.Fatal("expected error for mismatched query length")
	}
}

func TestAttentionWeightsRejectsInvalidGQAAndNonFiniteInputs(t *testing.T) {
	tests := []inference.AttentionCapture{
		{Heads: 3, KVHeads: 2, HeadDim: 1, Scale: 1, Query: reference.Value{Data: make([]float32, 3)}, Key: reference.Value{Data: make([]float32, 2)}},
		{Heads: 1, KVHeads: 1, HeadDim: 1, Scale: 1, Query: reference.Value{Data: []float32{float32(math.NaN())}}, Key: reference.Value{Data: []float32{1}}},
		{Heads: 1, KVHeads: 1, HeadDim: 1, Scale: float32(math.Inf(1)), Query: reference.Value{Data: []float32{1}}, Key: reference.Value{Data: []float32{1}}},
	}
	for index, capture := range tests {
		if _, err := attentionWeights(capture, 1); err == nil {
			t.Fatalf("case %d accepted", index)
		}
	}
}
