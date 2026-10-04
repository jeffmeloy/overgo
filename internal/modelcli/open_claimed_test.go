package modelcli

import (
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/discovery"
	"overgo/internal/gguf"
	"overgo/internal/inference"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/pathidentity"
	"overgo/internal/recipe"
	"overgo/internal/servingtest"
	"overgo/internal/testutil"
)

// writeServingFixture writes the smallest dense qwen2 model the host serves:
// one layer over an eight-token vocabulary. seed changes every weight and no
// size, so two seeds are two models in one layout.
func writeServingFixture(t *testing.T, path string, seed int) {
	t.Helper()
	const width, ffn, heads, kvHeads, vocabulary = 8, 16, 2, 1, 8
	const kvWidth = width / heads * kvHeads
	metadata := []gguf.Metadata{
		testutil.GGUFScalar("general.architecture", gguf.ValueTypeString, "qwen2"),
		testutil.GGUFScalar("qwen2.block_count", gguf.ValueTypeUint32, uint32(1)),
		testutil.GGUFScalar("qwen2.context_length", gguf.ValueTypeUint32, uint32(32)),
		testutil.GGUFScalar("qwen2.embedding_length", gguf.ValueTypeUint32, uint32(width)),
		testutil.GGUFScalar("qwen2.feed_forward_length", gguf.ValueTypeUint32, uint32(ffn)),
		testutil.GGUFScalar("qwen2.attention.head_count", gguf.ValueTypeUint32, uint32(heads)),
		testutil.GGUFScalar("qwen2.attention.head_count_kv", gguf.ValueTypeUint32, uint32(kvHeads)),
		testutil.GGUFScalar("qwen2.rope.freq_base", gguf.ValueTypeFloat32, float32(10_000)),
		testutil.GGUFScalar("qwen2.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
		testutil.GGUFScalar("tokenizer.ggml.model", gguf.ValueTypeString, "llama"),
		testutil.GGUFArray("tokenizer.ggml.tokens", gguf.ValueTypeString, []string{"<unk>", "<s>", "</s>", "▁", "a", "b", "c", "d"}),
		testutil.GGUFArray("tokenizer.ggml.scores", gguf.ValueTypeFloat32, make([]float32, vocabulary)),
		testutil.GGUFArray("tokenizer.ggml.token_type", gguf.ValueTypeInt32, []int32{2, 3, 3, 1, 1, 1, 1, 1}),
		testutil.GGUFScalar("tokenizer.ggml.bos_token_id", gguf.ValueTypeUint32, uint32(1)),
		testutil.GGUFScalar("tokenizer.ggml.eos_token_id", gguf.ValueTypeUint32, uint32(2)),
	}
	tensors := []gguf.TensorData{
		testutil.GGUFTensorF32("token_embd.weight", []uint64{width, vocabulary}, seed+1),
		testutil.GGUFTensorF32("output_norm.weight", []uint64{width}, seed+2),
		testutil.GGUFTensorF32("output.weight", []uint64{width, vocabulary}, seed+3),
		testutil.GGUFTensorF32("blk.0.attn_norm.weight", []uint64{width}, seed+4),
		testutil.GGUFTensorF32("blk.0.attn_q.weight", []uint64{width, width}, seed+5),
		testutil.GGUFTensorF32("blk.0.attn_k.weight", []uint64{width, kvWidth}, seed+6),
		testutil.GGUFTensorF32("blk.0.attn_v.weight", []uint64{width, kvWidth}, seed+7),
		testutil.GGUFTensorF32("blk.0.attn_output.weight", []uint64{width, width}, seed+8),
		testutil.GGUFTensorF32("blk.0.ffn_norm.weight", []uint64{width}, seed+9),
		testutil.GGUFTensorF32("blk.0.ffn_gate.weight", []uint64{width, ffn}, seed+10),
		testutil.GGUFTensorF32("blk.0.ffn_up.weight", []uint64{width, ffn}, seed+11),
		testutil.GGUFTensorF32("blk.0.ffn_down.weight", []uint64{ffn, width}, seed+12),
	}
	if err := gguf.WriteFileExclusive(path, metadata, tensors, gguf.WriteOptions{}); err != nil {
		t.Fatal(err)
	}
}

// capturedStderr runs open with the process's standard error redirected and
// returns what it wrote.
func capturedStderr(t *testing.T, open func()) string {
	t.Helper()
	reader, writer, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	original := os.Stderr
	os.Stderr = writer
	open()
	os.Stderr = original
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	text, err := io.ReadAll(reader)
	if err != nil {
		t.Fatal(err)
	}
	return string(text)
}

// TestOpenRunnerProvesTheRememberedIdentity opens host-served models whose
// weights the identity memo remembers. A true memo opens through its claim; a
// stale one, naming bytes the file no longer holds, never serves them under
// that identity: the open refuses, hashes the file, and serves it under the
// identity it has now, or refuses it when no recipe names that identity.
func TestOpenRunnerProvesTheRememberedIdentity(t *testing.T) {
	root := t.TempDir()
	storeRoot := filepath.Join(root, "store")
	path := filepath.Join(root, "model.gguf")
	rewritten := filepath.Join(root, "rewritten.gguf")
	unknown := filepath.Join(root, "unknown.gguf")
	writeServingFixture(t, path, 0)
	writeServingFixture(t, rewritten, 100)
	writeServingFixture(t, unknown, 200)

	ctx := t.Context()
	store, err := overgodb.Open(storeRoot)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	for _, published := range []string{path, rewritten} {
		err := servingtest.PublishActiveGGUFWithPolicy(
			ctx, store, published, recipe.PlacementHost, modelrecipe.DecodeSessionRequest, recipe.ResidencyHostReference,
		)
		if err != nil {
			t.Fatal(err)
		}
	}
	identityOf := func(file string) artifact.ID {
		inventory, err := modelartifact.FromGGUFPath(file)
		if err != nil {
			t.Fatal(err)
		}
		return inventory.Manifest.ID
	}
	original, replacement := identityOf(path), identityOf(rewritten)
	canonical, err := pathidentity.Canonical(path)
	if err != nil {
		t.Fatal(err)
	}
	memo := discovery.NewMemo()
	if _, err := discovery.WeightsIdentity(canonical, memo); err != nil {
		t.Fatal(err)
	}
	if err := discovery.PublishMemo(ctx, store, memo); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	modified, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	open := func(file string) (*inference.Runner, error, string) {
		var runner *inference.Runner
		var openErr error
		stderr := capturedStderr(t, func() {
			runner, openErr = OpenRunner(ctx, storeRoot, file, inference.OpenOptions{})
		})
		return runner, openErr, stderr
	}

	runner, err, stderr := open(path)
	if err != nil {
		t.Fatal(err)
	}
	if runner.ModelID() != original || stderr != "" {
		t.Fatalf("true memo: opened %s with stderr %q, want %s and none", runner.ModelID(), stderr, original)
	}
	if err := runner.Close(); err != nil {
		t.Fatal(err)
	}

	// The same size and modification time over other bytes: the memo cannot tell.
	replaceInPlace := func(source string) {
		data, err := os.ReadFile(source)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
		if err := os.Chtimes(path, modified.ModTime(), modified.ModTime()); err != nil {
			t.Fatal(err)
		}
	}
	replaceInPlace(rewritten)
	runner, err, stderr = open(path)
	if err != nil {
		t.Fatal(err)
	}
	if runner.ModelID() != replacement || !strings.Contains(stderr, "remembered identity") {
		t.Fatalf("stale memo: opened %s with stderr %q, want %s under the fallback notice", runner.ModelID(), stderr, replacement)
	}
	if err := runner.Close(); err != nil {
		t.Fatal(err)
	}

	replaceInPlace(unknown)
	runner, err, _ = open(path)
	if err == nil || !strings.Contains(err.Error(), "active inference recipe is absent") {
		if runner != nil {
			_ = runner.Close()
		}
		t.Fatalf("stale memo over bytes no recipe names: open = %v, want a missing recipe", err)
	}
}
