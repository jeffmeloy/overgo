package trainingworkflow

import (
	"bytes"
	"math"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

const (
	ggufFixtureEmbedding = 8
	ggufFixtureFFN       = 16
	ggufFixtureHeads     = 2
	ggufFixtureKVHeads   = 1
	ggufFixtureHeadWidth = ggufFixtureEmbedding / ggufFixtureHeads * ggufFixtureKVHeads
	ggufFixtureVocab     = 8
	ggufFixtureContext   = 32
)

// writeTrainingGGUF writes one hermetic dense qwen2 model: an
// unpermuted-RoPE conversion layout with an embedded SPM tokenizer, the
// same representation a converted checkpoint carries.
func writeTrainingGGUF(t *testing.T) string {
	t.Helper()
	metadata := []gguf.Metadata{
		testutil.GGUFScalar("general.architecture", gguf.ValueTypeString, "qwen2"),
		testutil.GGUFScalar("general.name", gguf.ValueTypeString, "gguf-training-fixture"),
		testutil.GGUFScalar("qwen2.block_count", gguf.ValueTypeUint32, uint32(1)),
		testutil.GGUFScalar("qwen2.context_length", gguf.ValueTypeUint32, uint32(ggufFixtureContext)),
		testutil.GGUFScalar("qwen2.embedding_length", gguf.ValueTypeUint32, uint32(ggufFixtureEmbedding)),
		testutil.GGUFScalar("qwen2.feed_forward_length", gguf.ValueTypeUint32, uint32(ggufFixtureFFN)),
		testutil.GGUFScalar("qwen2.attention.head_count", gguf.ValueTypeUint32, uint32(ggufFixtureHeads)),
		testutil.GGUFScalar("qwen2.attention.head_count_kv", gguf.ValueTypeUint32, uint32(ggufFixtureKVHeads)),
		testutil.GGUFScalar("qwen2.rope.freq_base", gguf.ValueTypeFloat32, float32(10_000)),
		testutil.GGUFScalar("qwen2.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
		testutil.GGUFScalar("tokenizer.ggml.model", gguf.ValueTypeString, "llama"),
		testutil.GGUFArray("tokenizer.ggml.tokens", gguf.ValueTypeString,
			[]string{"<unk>", "<s>", "</s>", "▁", "a", "b", "c", "d"}),
		testutil.GGUFArray("tokenizer.ggml.scores", gguf.ValueTypeFloat32, make([]float32, ggufFixtureVocab)),
		testutil.GGUFArray("tokenizer.ggml.token_type", gguf.ValueTypeInt32,
			[]int32{2, 3, 3, 1, 1, 1, 1, 1}),
		testutil.GGUFScalar("tokenizer.ggml.bos_token_id", gguf.ValueTypeUint32, uint32(1)),
		testutil.GGUFScalar("tokenizer.ggml.eos_token_id", gguf.ValueTypeUint32, uint32(2)),
	}
	tensors := []gguf.TensorData{
		testutil.GGUFTensorF32("token_embd.weight", []uint64{ggufFixtureEmbedding, ggufFixtureVocab}, 1),
		testutil.GGUFTensorF32("output_norm.weight", []uint64{ggufFixtureEmbedding}, 2),
		testutil.GGUFTensorF32("output.weight", []uint64{ggufFixtureEmbedding, ggufFixtureVocab}, 3),
		testutil.GGUFTensorF32("blk.0.attn_norm.weight", []uint64{ggufFixtureEmbedding}, 4),
		testutil.GGUFTensorF32("blk.0.attn_q.weight", []uint64{ggufFixtureEmbedding, ggufFixtureEmbedding}, 5),
		testutil.GGUFTensorF32("blk.0.attn_k.weight", []uint64{ggufFixtureEmbedding, ggufFixtureHeadWidth}, 6),
		testutil.GGUFTensorF32("blk.0.attn_v.weight", []uint64{ggufFixtureEmbedding, ggufFixtureHeadWidth}, 7),
		testutil.GGUFTensorF32("blk.0.attn_output.weight", []uint64{ggufFixtureEmbedding, ggufFixtureEmbedding}, 8),
		testutil.GGUFTensorF32("blk.0.ffn_norm.weight", []uint64{ggufFixtureEmbedding}, 9),
		testutil.GGUFTensorF32("blk.0.ffn_gate.weight", []uint64{ggufFixtureEmbedding, ggufFixtureFFN}, 10),
		testutil.GGUFTensorF32("blk.0.ffn_up.weight", []uint64{ggufFixtureEmbedding, ggufFixtureFFN}, 11),
		testutil.GGUFTensorF32("blk.0.ffn_down.weight", []uint64{ggufFixtureFFN, ggufFixtureEmbedding}, 12),
	}
	path := filepath.Join(t.TempDir(), "training-fixture.gguf")
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err = gguf.Write(file, metadata, tensors, gguf.WriteOptions{}); err != nil {
		_ = file.Close()
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	return path
}

// TestTrainingLoadsGGUFWeights pins GGUF training parity: a GGUF model
// file trains through exactly the same workflow as a safetensors
// directory — the recipe bootstraps against the file's own identity,
// tensors dequantize to the trainable catalog through the conversion
// name table, the tokenizer loads from GGUF metadata, one bounded step
// produces a finite loss, and the checkpoint records the GGUF model as
// its lineage parent.
func TestTrainingLoadsGGUFWeights(t *testing.T) {
	root := t.TempDir()
	modelPath := writeTrainingGGUF(t)
	datasetPath := filepath.Join(root, "dataset.txt")
	corpus := bytes.Repeat([]byte("a b c d "), 64)
	if err := os.WriteFile(datasetPath, corpus, 0o600); err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := t.Context()
	recipeID, err := BootstrapTokenRecipe(ctx, store, modelPath, datasetPath)
	if err != nil {
		t.Fatal(err)
	}
	result, err := Execute(ctx, Request{
		Repository: store, Recipe: recipeID, Observations: store,
		ModelDirectory: modelPath, DatasetPath: datasetPath,
		OutputDirectory: filepath.Join(root, "out"),
		Steps:           1, MaximumSequence: 16, Host: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Losses) == 0 || math.IsNaN(result.Losses[0]) || math.IsInf(result.Losses[0], 0) {
		t.Fatalf("gguf training losses = %v", result.Losses)
	}
	expected, err := identifyModel(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	if result.Checkpoint.Model != expected {
		t.Fatalf("checkpoint model = %s, want the GGUF file identity %s", result.Checkpoint.Model, expected)
	}
	if !result.Observation.Valid() {
		t.Fatal("gguf training recorded no session observation")
	}
}
