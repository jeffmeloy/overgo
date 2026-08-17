package main

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/densecausal"
	"overgo/internal/safetensors"
	"overgo/internal/testutil"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
)

func TestTokenBatchesUseResumableDatasetStream(t *testing.T) {
	encode := func(text string) ([]int, error) {
		result := make([]int, len(text))
		for index := range text {
			result[index] = int(text[index])
		}
		return result, nil
	}
	batches, state, err := tokenBatches(context.Background(), []byte("abcdef"), 3, 4, encode)
	if err != nil {
		t.Fatal(err)
	}
	if state.Position != 3 || len(batches) != 3 {
		t.Fatalf("state=%+v batches=%d", state, len(batches))
	}
	for index, batch := range batches {
		if len(batch) != 4 || !reflect.DeepEqual(batch, batches[0]) {
			t.Fatalf("batch %d = %v", index, batch)
		}
	}
}

func writeArtifactDir(t *testing.T, dir string, weights map[string][]float32, shapes map[string][]int) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := safetensors.Save(filepath.Join(dir, "model.safetensors"), weights, shapes, nil); err != nil {
		t.Fatalf("Save fixture: %v", err)
	}
	config := `{"model_type":"llama","num_attention_heads":2,"head_dim":4,"rope_theta":10000.0,"rms_norm_eps":1e-6,"tie_word_embeddings":true}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(config), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(`{"model":{"type":"BPE"}}`), 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestTrainGlueLoadStepSaveReload exercises cmd/train's real path end to end on
// the host: a synthetic on-disk model loads, trains one step, and the checkpoint
// saveCheckpoint writes reloads through densecausal.Load with matching geometry.
// This proves the writer produces loader-compatible artifacts and that the
// production caller is not inert.
func TestTrainGlueLoadStepSaveReload(t *testing.T) {
	weights, shapes := testutil.DenseCausalWeights(t, testutil.DenseCausalSpec{
		Vocab: 8, Hidden: 8, Heads: 2, HeadDim: 4,
		KVHeads: 1, Intermediate: 16, Layers: 1, Seed: 1,
	})
	src := filepath.Join(t.TempDir(), "src")
	writeArtifactDir(t, src, weights, shapes)

	model, err := densecausal.Load(src)
	if err != nil {
		t.Fatalf("Load fixture (writer not loader-compatible?): %v", err)
	}

	before := append([]float32(nil), model.Weights["model.embed_tokens.weight"]...)
	tokens := []int{1, 2, 3, 4, 5}
	traj, backend, state, err := runTrainingState(model, [][]int{tokens}, 0, 0.9, false, false, nil)
	if err != nil {
		t.Fatalf("runTraining: %v", err)
	}
	if backend != "host" {
		t.Fatalf("backend = %q, want host", backend)
	}
	if len(traj) != 1 || math.IsNaN(traj[0]) || math.IsInf(traj[0], 0) {
		t.Fatalf("trajectory = %v, want one finite loss", traj)
	}
	changed := false
	for i, v := range model.Weights["model.embed_tokens.weight"] {
		if v != before[i] {
			changed = true
			break
		}
	}
	if !changed {
		t.Fatalf("training left weights unchanged")
	}

	out := filepath.Join(t.TempDir(), "out")
	spec := checkpointSpecForTest(t, src, state, 1)
	if _, err := saveCheckpoint(src, out, model, spec); err != nil {
		t.Fatalf("saveCheckpoint: %v", err)
	}
	reloaded, err := densecausal.Load(out)
	if err != nil {
		t.Fatalf("reload checkpoint: %v", err)
	}
	if reloaded.Dims != model.Dims {
		t.Fatalf("reloaded dims %+v != %+v", reloaded.Dims, model.Dims)
	}
	got := reloaded.Weights["model.embed_tokens.weight"]
	for i, v := range model.Weights["model.embed_tokens.weight"] {
		if got[i] != v {
			t.Fatalf("reloaded weight[%d] = %v, want %v (trained value not persisted)", i, got[i], v)
		}
	}
	for _, name := range []string{"config.json", "tokenizer.json", "model.safetensors", trainingprogram.CheckpointFilename} {
		if _, err := os.Stat(filepath.Join(out, name)); err != nil {
			t.Fatalf("checkpoint missing %s: %v", name, err)
		}
	}
}

func TestProductionCheckpointResumeExact(t *testing.T) {
	weights, shapes := testutil.DenseCausalWeights(t, testutil.DenseCausalSpec{
		Vocab: 8, Hidden: 8, Heads: 2, HeadDim: 4,
		KVHeads: 1, Intermediate: 16, Layers: 1, Seed: 7,
	})
	root := t.TempDir()
	source := filepath.Join(root, "source")
	writeArtifactDir(t, source, weights, shapes)
	batches := [][]int{{1, 2, 3, 4, 5}, {5, 4, 3, 2, 1}}

	uninterrupted, err := densecausal.Load(source)
	if err != nil {
		t.Fatal(err)
	}
	_, _, uninterruptedState, err := runTrainingState(uninterrupted, batches, 0, 0.9, false, false, nil)
	if err != nil {
		t.Fatal(err)
	}

	staged, err := densecausal.Load(source)
	if err != nil {
		t.Fatal(err)
	}
	_, _, firstState, err := runTrainingState(staged, batches[:1], 0, 0.9, false, false, nil)
	if err != nil {
		t.Fatal(err)
	}
	checkpointDir := filepath.Join(root, "step-1")
	checkpoint, err := saveCheckpoint(source, checkpointDir, staged, checkpointSpecForTest(t, source, firstState, 1))
	if err != nil {
		t.Fatal(err)
	}
	loaded, err := trainingprogram.LoadCheckpoint(checkpointDir)
	if err != nil || loaded.ID() != checkpoint.ID() || loaded.Stream.Position != 1 {
		t.Fatalf("load checkpoint = (%s, %+v, %v)", loaded.ID(), loaded.Stream, err)
	}
	resumed, err := densecausal.Load(checkpointDir)
	if err != nil {
		t.Fatal(err)
	}
	resumeState := densecausal.TrainState(loaded.Optimizer)
	_, _, finalState, err := runTrainingState(resumed, batches[1:], 0, 0.9, false, false, &resumeState)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(uninterrupted.Weights, resumed.Weights) || !reflect.DeepEqual(uninterruptedState, finalState) {
		t.Fatal("resumed weights or Muon state differ from uninterrupted training")
	}
	if _, err := saveCheckpoint(source, checkpointDir, resumed, checkpointSpecForTest(t, source, finalState, 2)); err == nil {
		t.Fatal("checkpoint overwrite accepted")
	}
}

func checkpointSpecForTest(t *testing.T, source string, state densecausal.TrainState, position uint64) trainingprogram.CheckpointSpec {
	t.Helper()
	dataset, _ := artifact.IdentifyBytes(artifact.KindDataset, []byte("dataset"))
	split, _ := artifact.IdentifyBytes(artifact.KindDatasetShard, []byte("split"))
	processor, _ := artifact.IdentifyBytes(artifact.KindProfile, []byte("processor"))
	streamIdentity, _ := artifact.IdentifyBytes(artifact.KindDatasetShard, []byte("stream"))
	model, err := densecausal.Load(source)
	if err != nil {
		t.Fatal(err)
	}
	authority, err := compileTrainingAuthority(
		model, source, batchAuthority{Dataset: dataset, Split: split, Processor: processor},
		trainingdata.StreamState{Identity: streamIdentity, Position: position}, state.Config.BaseLearningRate,
		trainingprogram.Checkpoint{}, trainingprogram.ObjectiveTokenPrediction, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	spec, err := authority.checkpointSpec(state)
	if err != nil {
		t.Fatal(err)
	}
	return spec
}

func TestDPOCommandCompilesPreferenceBatches(t *testing.T) {
	raw := []byte("{\"id\":\"first\",\"prompt\":\"ab\",\"chosen\":\"c\",\"rejected\":\"d\"}\n")
	encode := func(text string) ([]int, error) {
		tokens := make([]int, len(text))
		for index := range text {
			tokens[index] = int(text[index])
		}
		return tokens, nil
	}
	batches, state, _, err := preferenceBatchesResume(context.Background(), raw, 2, encode, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(batches) != 2 || state.Position != 2 || batches[0].Pairs[0].SharedPrefix != 2 ||
		!batches[0].Pairs[0].Chosen.Completion[2] || !batches[0].Pairs[0].Rejected.Completion[2] {
		t.Fatalf("preference batches=%+v state=%+v", batches, state)
	}
	resumed, resumedState, _, err := preferenceBatchesResume(context.Background(), raw, 1, encode, &batches[0].State)
	if err != nil {
		t.Fatal(err)
	}
	if len(resumed) != 1 || resumedState.Position != batches[0].State.Position+1 {
		t.Fatalf("resumed batches=%+v state=%+v", resumed, resumedState)
	}
}
