//go:build modeltest

package main

import (
	"math"
	"os"
	"path/filepath"
	"runtime/debug"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/gguf"
	"overgo/internal/inference"
	"overgo/internal/modelcli"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/modeltest"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/sampling"
	"overgo/internal/servingtest"
	"overgo/internal/trainingprogram"
)

// e4bTrainingText is the text the real E4B trains its one step on, and
// e4bHeldOutPrompt with e4bHeldOutAnswer the prompt it answers after,
// which the training never read.
const (
	e4bTrainingText = "The river rises in the northern hills and runs south through three valleys, " +
		"where the old mills once turned its water into flour for the towns along its banks. " +
		"Each spring the snowmelt fills the channel, and the ferrymen wait for the current to slow " +
		"before they carry travellers and their carts from one shore to the other."
	e4bHeldOutPrompt = "Question: What is the capital city of France?\nAnswer:"
	e4bHeldOutAnswer = " Paris"
)

// TestE4BRoutePublishesAServableModel trains the real Gemma-4 E4B one step
// through the e4b route on the device, into a fresh store holding the
// architecture profiles, and holds what it publishes to being a model the
// server serves: the checkpoint loads with its resume sidecar at the step
// it took, the trained model opens as its trained shard, whose trained
// tensors are f32, and the shard of tensors it never trained, and once its
// registered candidate is activated the server's own open answers a
// held-out prompt, scoring the held-out answer differently from the base.
func TestE4BRoutePublishesAServableModel(t *testing.T) {
	// Serial: trains and serves the real E4B, which takes the device and
	// about 72 GB of host memory to itself.
	cudatest.Require(t)
	base := modeltest.Directory(t, "gemma-4-E4B-it-bf16.gguf")
	ctx := t.Context()
	root := t.TempDir()
	dataset := filepath.Join(root, "dataset.txt")
	if err := os.WriteFile(dataset, []byte(e4bTrainingText), 0o600); err != nil {
		t.Fatal(err)
	}
	storePath, out := filepath.Join(root, "store"), filepath.Join(root, "trained")
	store, err := overgodb.Open(storePath)
	if err != nil {
		t.Fatal(err)
	}
	_, err = modelrecipe.PublishArchitectureProfileCatalog(ctx, store)
	if closeErr := store.Close(); err != nil || closeErr != nil {
		t.Fatal(err, closeErr)
	}
	if err := runRoute(ctx, []string{"e4b", "-model", base, "-dataset", dataset, "-steps", "1", "-seq", "64", "-store", storePath, "-out", out}); err != nil {
		t.Fatal(err)
	}
	// The trainer's masters and momentum are garbage once the route
	// returns; hand them back before the checkpoint and the server load.
	debug.FreeOSMemory()

	checkpoint, err := trainingprogram.LoadCheckpoint(out)
	if err != nil || checkpoint.Optimizer.Step != 1 || len(checkpoint.Optimizer.MomentumF32) != checkpoint.ParameterCount {
		t.Fatalf("checkpoint at step %d with %d f32 momentum values (%v), want step 1 with one per trained parameter", checkpoint.Optimizer.Step, len(checkpoint.Optimizer.MomentumF32), err)
	}
	trained := filepath.Join(out, trainingprogram.CheckpointModel)
	file, err := gguf.Open(trained)
	if err != nil {
		t.Fatal(err)
	}
	query, trainedQuery := file.Tensor("blk.0.attn_q.weight")
	embedding, frozenEmbedding := file.Tensor("per_layer_token_embd.weight")
	if err := file.Close(); err != nil || len(file.SourcePaths()) != 2 || !trainedQuery || query.Type != gguf.DTypeF32 || query.Shard != 0 ||
		!frozenEmbedding || embedding.Type != gguf.DTypeBF16 || embedding.Shard != 1 {
		t.Fatalf("trained model shards %v: blk.0.attn_q %s in shard %d, per_layer_token_embd %s in shard %d (%v), want the trained query in f32 in the first and the frozen table as stored in the second",
			file.SourcePaths(), query.Type, query.Shard, embedding.Type, embedding.Shard, err)
	}
	checkpoint.Optimizer.MomentumF32 = nil
	debug.FreeOSMemory()

	store, err = overgodb.Open(storePath)
	if err != nil {
		t.Fatal(err)
	}
	candidate := registeredCandidate(t, store, checkpoint.ID())
	err = modelrecipetest.ActivateCandidate(ctx, store, "e4b/trained-model", candidate)
	if closeErr := store.Close(); err != nil || closeErr != nil {
		t.Fatal(err, closeErr)
	}
	served, err := modelcli.OpenRunner(ctx, storePath, trained, inference.OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	greedy, err := sampling.New(sampling.Config{})
	if err != nil {
		t.Fatal(err)
	}
	generated := 0
	_, answer, err := served.Generate(ctx, e4bHeldOutPrompt, inference.GenerateOptions{
		MaxNewTokens: 8, Sampler: greedy, OnToken: func(inference.TokenEvent) error { generated++; return nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	scores, err := served.ScoreContinuations(ctx, e4bHeldOutPrompt, []string{e4bHeldOutAnswer})
	if closeErr := served.Close(); err != nil || closeErr != nil {
		t.Fatal(err, closeErr)
	}
	t.Logf("served trained E4B answers %q in %d tokens; held-out answer log-probability %.6f over %d tokens", answer, generated, scores[0].LogProbability, scores[0].Tokens)
	if generated == 0 || len(scores) != 1 || scores[0].Tokens == 0 || math.IsNaN(scores[0].LogProbability) || math.IsInf(scores[0].LogProbability, 0) {
		t.Fatalf("served trained E4B generated %d tokens (%q) with scores %+v, want an answer and a finite held-out score", generated, answer, scores)
	}
	debug.FreeOSMemory()

	// The base serves as intake registers a trained model's candidate.
	loaded, err := servingtest.ResolveActiveGGUFWithPolicy(base, recipe.PlacementHybrid, modelrecipe.DecodeSessionRequest, recipe.ResidencyHybridNative)
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := inference.OpenWithProgram(ctx, &loaded, inference.OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	before, err := baseline.ScoreContinuations(ctx, e4bHeldOutPrompt, []string{e4bHeldOutAnswer})
	if closeErr := baseline.Close(); err != nil || closeErr != nil {
		t.Fatal(err, closeErr)
	}
	t.Logf("base E4B held-out answer log-probability %.6f", before[0].LogProbability)
	if before[0].LogProbability == scores[0].LogProbability {
		t.Fatalf("the served trained model scores the held-out answer %.9f, as the base does: the server did not serve the trained weights", scores[0].LogProbability)
	}
}
