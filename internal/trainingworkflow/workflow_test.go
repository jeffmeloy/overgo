package trainingworkflow

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"overgo/internal/densecausal"
	"overgo/internal/safetensors"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

func TestNativeTrainingWorkflowDPOExactResume(t *testing.T) {
	weights, shapes := testutil.DenseCausalWeights(t, testutil.DenseCausalSpec{
		Vocab: 8, Hidden: 8, Heads: 2, HeadDim: 4, KVHeads: 1, Intermediate: 16, Layers: 1, Seed: 7,
	})
	root := t.TempDir()
	policy := filepath.Join(root, "policy")
	reference := filepath.Join(root, "reference")
	writeModel(t, policy, weights, shapes)
	writeModel(t, reference, weights, shapes)
	dataset := filepath.Join(root, "preference.jsonl")
	if err := os.WriteFile(dataset, []byte("{\"id\":\"pair\",\"prompt\":\"ab\",\"chosen\":\"c\",\"rejected\":\"d\"}\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	request := Request{
		ModelDirectory: policy, ReferenceDirectory: reference, DatasetPath: dataset,
		Steps: 2, LearningRate: 0, Momentum: 0.9, DPOScale: 0.1, Host: true,
	}
	request.OutputDirectory = filepath.Join(root, "uninterrupted")
	want, err := Execute(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	request.Steps = 1
	request.OutputDirectory = filepath.Join(root, "first")
	first, err := Execute(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	request.ModelDirectory = ""
	request.ResumeDirectory = request.OutputDirectory
	request.OutputDirectory = filepath.Join(root, "resumed")
	got, err := Execute(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	wantModel, err := densecausal.Load(filepath.Join(root, "uninterrupted"))
	if err != nil {
		t.Fatal(err)
	}
	gotModel, err := densecausal.Load(filepath.Join(root, "resumed"))
	if err != nil {
		t.Fatal(err)
	}
	wantCheckpoint, err := trainingprogram.LoadCheckpoint(filepath.Join(root, "uninterrupted"))
	if err != nil {
		t.Fatal(err)
	}
	gotCheckpoint, err := trainingprogram.LoadCheckpoint(filepath.Join(root, "resumed"))
	if err != nil {
		t.Fatal(err)
	}
	if want.Backend != "host" || first.StreamPosition+1 != got.StreamPosition ||
		!reflect.DeepEqual(wantModel.Weights, gotModel.Weights) ||
		!reflect.DeepEqual(wantCheckpoint.Optimizer, gotCheckpoint.Optimizer) {
		t.Fatalf("DPO resume differs: want=%+v first=%+v got=%+v", want, first, got)
	}
}

func writeModel(t *testing.T, directory string, weights map[string][]float32, shapes map[string][]int) {
	t.Helper()
	if err := os.MkdirAll(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := safetensors.Save(filepath.Join(directory, trainingprogram.CheckpointWeights), weights, shapes, nil); err != nil {
		t.Fatal(err)
	}
	config := `{"model_type":"llama","num_attention_heads":2,"head_dim":4,"rope_theta":10000.0,"rms_norm_eps":1e-6,"tie_word_embeddings":true}`
	tokenizer := `{"model":{"type":"BPE","vocab":{"a":1,"b":2,"c":3,"d":4},"merges":[]}}`
	if err := os.WriteFile(filepath.Join(directory, "config.json"), []byte(config), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, "tokenizer.json"), []byte(tokenizer), 0o600); err != nil {
		t.Fatal(err)
	}
}
