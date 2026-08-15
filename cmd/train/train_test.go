package main

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
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
	writeArtifactDirTyped(t, dir, weights, shapes, "llama")
}

// writeArtifactDirTyped writes the fixture with the given model_type; qwen2 is
// the bias-bearing family (Load requires qkv biases for it), llama forbids them.
func writeArtifactDirTyped(t *testing.T, dir string, weights map[string][]float32, shapes map[string][]int, modelType string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := safetensors.Save(filepath.Join(dir, "model.safetensors"), weights, shapes, nil); err != nil {
		t.Fatalf("Save fixture: %v", err)
	}
	config := `{"model_type":"` + modelType + `","num_attention_heads":2,"head_dim":4,"rope_theta":10000.0,"rms_norm_eps":1e-6,"tie_word_embeddings":true}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(config), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(`{"model":{"type":"BPE"}}`), 0o644); err != nil {
		t.Fatal(err)
	}
}

// addAttnBias augments a dense-causal fixture with the q/k/v projection biases
// that make densecausal.Load derive Dims.AttnBias=true -- a trait the device
// training backend does not support. out dims match the projection out-dims
// (Heads*HeadDim for q, KVHeads*HeadDim for k/v).
func addAttnBias(weights map[string][]float32, shapes map[string][]int, layers, qOut, kvOut int) {
	for layer := 0; layer < layers; layer++ {
		prefix := "model.layers." + strconv.Itoa(layer) + ".self_attn."
		for _, b := range []struct {
			name string
			out  int
		}{
			{prefix + "q_proj.bias", qOut},
			{prefix + "k_proj.bias", kvOut},
			{prefix + "v_proj.bias", kvOut},
		} {
			weights[b.name] = make([]float32, b.out)
			shapes[b.name] = []int{b.out}
		}
	}
}

// TestTrainAttnBiasRoutedToHostAtSelection proves the CUDA training admission
// gate: a model with attention bias -- unsupported by the device training
// backend -- is routed to the HOST path at SELECTION time (before any device
// session is constructed), never failing mid-session. runTraining is asked to
// prefer the device (preferDevice=true); on GPU hardware, without the
// selection-time predicate this model would reach resident training and error on
// the unsupported bias. Instead it must return backend "host" with a finite
// trajectory. Host-only: no GPU required.
func TestTrainAttnBiasRoutedToHostAtSelection(t *testing.T) {
	const layers = 1
	weights, shapes := testutil.DenseCausalWeights(t, testutil.DenseCausalSpec{
		Vocab: 8, Hidden: 8, Heads: 2, HeadDim: 4,
		KVHeads: 1, Intermediate: 16, Layers: layers, Seed: 2,
	})
	addAttnBias(weights, shapes, layers, 2*4 /*Heads*HeadDim*/, 1*4 /*KVHeads*HeadDim*/)

	src := filepath.Join(t.TempDir(), "src")
	writeArtifactDirTyped(t, src, weights, shapes, "qwen2")

	model, err := densecausal.Load(src)
	if err != nil {
		t.Fatalf("Load bias fixture: %v", err)
	}
	if !model.Dims.AttnBias {
		t.Fatal("fixture must derive AttnBias=true")
	}
	// The device backend must refuse this model at selection time.
	if ok, reason := densecausal.DeviceTrainingSupported(model.Dims); ok {
		t.Fatal("attention-bias model must be refused by DeviceTrainingSupported")
	} else if reason == "" {
		t.Fatal("refusal must carry a reason")
	}

	tokens := []int{1, 2, 3, 4, 5}
	traj, backend, _, err := runTrainingState(model, [][]int{tokens}, 0, 0.9, true, false, nil)
	if err != nil {
		t.Fatalf("runTraining routed a bias model into a device session (mid-session failure): %v", err)
	}
	if backend != "host" {
		t.Fatalf("backend = %q, want host (bias model must fall back at selection time)", backend)
	}
	if len(traj) != 1 || math.IsNaN(traj[0]) || math.IsInf(traj[0], 0) {
		t.Fatalf("trajectory = %v, want one finite loss", traj)
	}
	if _, _, _, err := runTrainingState(model, [][]int{tokens}, 0, 0.9, true, true, nil); err == nil {
		t.Fatal("frozen lexical request silently fell back to host")
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
	spec, err := productionCheckpointSpec(model, source, batchAuthority{Dataset: dataset, Split: split, Processor: processor}, trainingdata.StreamState{Identity: streamIdentity, Position: position}, state, trainingprogram.Checkpoint{})
	if err != nil {
		t.Fatal(err)
	}
	return spec
}
