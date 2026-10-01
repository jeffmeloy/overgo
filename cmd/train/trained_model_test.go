package main

import (
	"bytes"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
	"overgo/internal/inference"
	"overgo/internal/modelartifact"
	"overgo/internal/modelcli"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/servingtest"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
	"overgo/internal/trainingworkflow"
)

// writeServingFixture writes the smallest dense qwen2 model training and
// serving both read: one layer over an eight-token SPM vocabulary.
func writeServingFixture(t *testing.T, directory string) string {
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
		testutil.GGUFTensorF32("token_embd.weight", []uint64{width, vocabulary}, 1),
		testutil.GGUFTensorF32("output_norm.weight", []uint64{width}, 2),
		testutil.GGUFTensorF32("output.weight", []uint64{width, vocabulary}, 3),
		testutil.GGUFTensorF32("blk.0.attn_norm.weight", []uint64{width}, 4),
		testutil.GGUFTensorF32("blk.0.attn_q.weight", []uint64{width, width}, 5),
		testutil.GGUFTensorF32("blk.0.attn_k.weight", []uint64{width, kvWidth}, 6),
		testutil.GGUFTensorF32("blk.0.attn_v.weight", []uint64{width, kvWidth}, 7),
		testutil.GGUFTensorF32("blk.0.attn_output.weight", []uint64{width, width}, 8),
		testutil.GGUFTensorF32("blk.0.ffn_norm.weight", []uint64{width}, 9),
		testutil.GGUFTensorF32("blk.0.ffn_gate.weight", []uint64{width, ffn}, 10),
		testutil.GGUFTensorF32("blk.0.ffn_up.weight", []uint64{width, ffn}, 11),
		testutil.GGUFTensorF32("blk.0.ffn_down.weight", []uint64{ffn, width}, 12),
	}
	path := filepath.Join(directory, "base.gguf")
	if err := gguf.WriteFileExclusive(path, metadata, tensors, gguf.WriteOptions{}); err != nil {
		t.Fatal(err)
	}
	return path
}

// TestTrainedModelLoadsForServing trains the smallest hermetic text model
// one step on the host through the recipe-bound workflow cmd/train runs,
// and holds what the run publishes to being a model the server opens: a
// GGUF in its base's layout, registered as an inference candidate with its
// lineage to the base, that the server's own open (modelcli.OpenRunner
// over an activated recipe) loads and scores a prompt with, differently
// from the base it trained from.
func TestTrainedModelLoadsForServing(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	base := writeServingFixture(t, root)
	dataset := filepath.Join(root, "dataset.txt")
	if err := os.WriteFile(dataset, bytes.Repeat([]byte("a b c d "), 64), 0o600); err != nil {
		t.Fatal(err)
	}
	storeRoot := filepath.Join(root, "store")
	store, err := overgodb.Open(storeRoot)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := t.Context()
	if _, err := modelrecipe.PublishArchitectureProfileCatalog(ctx, store); err != nil {
		t.Fatal(err)
	}
	recipeID, err := trainingworkflow.BootstrapTokenRecipe(ctx, store, base, dataset)
	if err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(root, "trained")
	result, err := trainingworkflow.Execute(ctx, trainingworkflow.Request{
		Repository: store, Recipe: recipeID, Observations: store,
		ModelDirectory: base, DatasetPath: dataset, OutputDirectory: out,
		Steps: 1, MaximumSequence: 16, Host: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	trained := filepath.Join(out, trainingprogram.CheckpointModel)
	file, err := gguf.Open(trained)
	if err != nil {
		t.Fatal(err)
	}
	inventory, err := modelartifact.FromGGUF(file, artifact.KindModel)
	if closeErr := file.Close(); err != nil || closeErr != nil {
		t.Fatal(err, closeErr)
	}
	definition, err := recipe.RequireDefinition(ctx, store, result.Candidate)
	if err != nil || definition.Task != recipe.TaskInference || definition.Model != inventory.Manifest.ID {
		t.Fatalf("candidate %s = (%+v, %v), want an inference candidate for the trained model %s", result.Candidate, definition.Model, err, inventory.Manifest.ID)
	}
	parents, err := store.Parents(ctx, definition.Model)
	if err != nil || !slices.Contains(parents, artifact.Lineage{Child: definition.Model, Parent: result.Checkpoint.Model, Relation: artifact.RelationTrainedFrom}) ||
		!slices.Contains(parents, artifact.Lineage{Child: definition.Model, Parent: result.Checkpoint.ID(), Relation: artifact.RelationProducedBy}) {
		t.Fatalf("trained model lineage = (%v, %v), want it trained from %s and produced by %s", parents, err, result.Checkpoint.Model, result.Checkpoint.ID())
	}
	if loaded, err := trainingprogram.LoadCheckpoint(out); err != nil || loaded.Weights != inventory.Components[0].ID {
		t.Fatalf("checkpoint = (%s, %v), want it bound to the trained model %s", loaded.Weights, err, inventory.Components[0].ID)
	}
	// The server opens what an activated recipe names; activate the trained
	// model on the host and open it the way the server does.
	profile, _ := definition.PrimaryDependency(recipe.DependencyProfile)
	modelDefinition, _ := definition.PrimaryDependency(recipe.DependencyDefinition)
	host, err := modelrecipe.InferenceWithModelDefinition(definition.Model, profile, modelDefinition, recipe.PlacementHost, modelrecipe.DecodeSessionRequest, recipe.ResidencyHostReference)
	if err != nil {
		t.Fatal(err)
	}
	if err := modelrecipetest.PublishActivation(ctx, store, "fixture/trained-model", host); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	served, err := modelcli.OpenRunner(ctx, storeRoot, trained, inference.OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer served.Close()
	loaded, err := servingtest.ResolveActiveGGUFWithPolicy(base, recipe.PlacementHost, modelrecipe.DecodeSessionRequest, recipe.ResidencyHostReference)
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := inference.OpenWithProgram(ctx, &loaded, inference.OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer baseline.Close()
	prompt, continuation := "a b", []string{" c d"}
	scores, err := served.ScoreContinuations(ctx, prompt, continuation)
	if err != nil {
		t.Fatal(err)
	}
	before, err := baseline.ScoreContinuations(ctx, prompt, continuation)
	if err != nil {
		t.Fatal(err)
	}
	if len(scores) != 1 || scores[0].Tokens == 0 || math.IsNaN(scores[0].LogProbability) || math.IsInf(scores[0].LogProbability, 0) ||
		scores[0].LogProbability == before[0].LogProbability {
		t.Fatalf("served trained model scores %+v against the base's %+v, want a finite score of its own", scores, before)
	}
}
