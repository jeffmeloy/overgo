//go:build modeltest

package inference

import (
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/modeltest"
	"overgo/internal/tokenizer"
)

// TestExtractAttentionShapeGeometry is a live probe that pins the memory layout
// the /analyze/attention host recompute depends on: the captured query/key must
// be rank-3 with Dims [headDim, heads, tokens] (queries) and [headDim, kvHeads,
// tokens] (keys), column-major (Dims[0] innermost), matching the geometry the
// runner reports. It also checks the recomputed weights are causal and sum to 1.
//
// It runs on the smallest registered Qwen2 GGUF, on the device:
// capture reads dense-causal attention blocks, and Qwen3.5's gated
// attention layers expose none.
func TestExtractAttentionShapeGeometry(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	runner, err := openNativeFixtureRunner(modeltest.GGUF(t, "qwen2"), OpenOptions{DeviceOrdinal: 0})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()

	tokens := []tokenizer.TokenID{9707, 27, 785, 4062}
	ctx := t.Context()

	// Find a layer that exposes attention (a dense-causal block); hybrid models
	// have non-attention layers whose query is not captured.
	blocks := len(runner.weights.Layers)
	var capture AttentionCapture
	var found bool
	for _, layer := range []int32{int32(blocks / 2), 0, int32(blocks - 1)} {
		capture, err = runner.ExtractAttention(ctx, tokens, layer)
		if err == nil {
			found = true
			break
		}
		t.Logf("layer %d: %v", layer, err)
	}
	if !found {
		t.Fatalf("no layer exposed attention capture: %v", err)
	}

	headDim := int(runner.spec.KeyLength)
	if capture.HeadDim != headDim {
		t.Fatalf("HeadDim = %d, want KeyLength %d", capture.HeadDim, headDim)
	}
	// Query geometry.
	if capture.Query.Shape.Rank != 3 {
		t.Fatalf("query rank = %d, want 3", capture.Query.Shape.Rank)
	}
	wantQ := [3]uint64{uint64(headDim), uint64(capture.Heads), uint64(len(tokens))}
	gotQ := [3]uint64{capture.Query.Shape.Dims[0], capture.Query.Shape.Dims[1], capture.Query.Shape.Dims[2]}
	if gotQ != wantQ {
		t.Fatalf("query Dims = %v, want [headDim heads tokens] = %v", gotQ, wantQ)
	}
	if len(capture.Query.Data) != headDim*capture.Heads*len(tokens) {
		t.Fatalf("query data len = %d, want %d", len(capture.Query.Data), headDim*capture.Heads*len(tokens))
	}
	// Key geometry.
	wantK := [3]uint64{uint64(headDim), uint64(capture.KVHeads), uint64(len(tokens))}
	gotK := [3]uint64{capture.Key.Shape.Dims[0], capture.Key.Shape.Dims[1], capture.Key.Shape.Dims[2]}
	if capture.Key.Shape.Rank != 3 || gotK != wantK {
		t.Fatalf("key Dims = %v (rank %d), want [headDim kvHeads tokens] = %v", gotK, capture.Key.Shape.Rank, wantK)
	}
	if capture.Scale <= 0 {
		t.Fatalf("scale = %v, want > 0", capture.Scale)
	}
	t.Logf("layer %d: heads=%d kvHeads=%d headDim=%d tokens=%d scale=%g",
		capture.Layer, capture.Heads, capture.KVHeads, capture.HeadDim, capture.Tokens, capture.Scale)
}
