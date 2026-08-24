//go:build windows

package adaptiveparity_test

import (
	"context"
	"math"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/adaptertrain"
	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/gguf"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
)

func TestAlternatingWindowTrainingCells(t *testing.T) {
	cudatest.Require(t)
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	e4bPath := filepath.Join(roots.Checkpoints, "overgo-hfconvert", "gemma-4-E4B-it-bf16.gguf")
	input, target := adjacentTokenRows(t, e4bPath)
	trained, topology, err := adaptertrain.LoadArtifact(context.Background(), e4bPath, 0, trainingprogram.BuiltinOptimizerPolicy())
	if err != nil {
		t.Fatal(err)
	}
	layer, err := topology.Layer(0)
	if err != nil || !layer.PerLayerInput || trained.ParameterCount() == 0 {
		t.Fatalf("layer=%+v parameters=%d err=%v", layer, trained.ParameterCount(), err)
	}
	example, err := trained.BuildExample(context.Background(), e4bPath, nil, input, target, "text")
	if err != nil {
		t.Fatal(err)
	}
	loss, err := trained.Step(worker, example)
	if err != nil || math.IsNaN(loss) || math.IsInf(loss, 0) {
		t.Fatalf("loss=%g err=%v", loss, err)
	}
	t.Logf("E4B adapter cell: rows=%d parameters=%d loss=%.6f", example.Rows, trained.ParameterCount(), loss)

	gemma12Path := filepath.Join(roots.Checkpoints, "overgo-hfconvert", "gemma-4-12B-it-fp8-native.gguf")
	if _, _, err := adaptertrain.LoadArtifact(context.Background(), gemma12Path, 0, trainingprogram.BuiltinOptimizerPolicy()); err == nil ||
		!strings.Contains(err.Error(), "no per-layer input program") {
		t.Fatalf("12B adapter admission error=%v", err)
	}
}

func adjacentTokenRows(t *testing.T, path string) ([]uint32, []uint32) {
	t.Helper()
	file, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	vocabulary, err := tokenizer.Load(file)
	closeErr := file.Close()
	if err != nil || closeErr != nil {
		t.Fatalf("tokenizer=%v close=%v", err, closeErr)
	}
	ids, err := vocabulary.Encode("A short training example.", tokenizer.EncodeOptions{AddSpecial: true})
	if err != nil || len(ids) < 2 {
		t.Fatalf("tokens=%v err=%v", ids, err)
	}
	input, target, err := trainingdata.AdjacentTokenRows(ids)
	if err != nil {
		t.Fatal(err)
	}
	return input, target
}
