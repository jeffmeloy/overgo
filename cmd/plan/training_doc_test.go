package main

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestTrainingPlanMatchesImplementation(t *testing.T) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("caller path unavailable")
	}
	root := filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
	read := func(path string) string {
		raw, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(path)))
		if err != nil {
			t.Fatal(err)
		}
		return string(raw)
	}
	doc := read("docs/training_plan.md")
	normalized := strings.Join(strings.Fields(doc), " ")
	for _, fact := range []string{
		"dense host and CUDA forward, loss, backward and Muon matrix updates",
		"compiled scratch construction/program authority",
		"shared lazy dataset materialization",
		"`ScratchConstruction`",
		"Adaptive_new is an offline oracle, not a runtime dependency",
		"`cmd/train` selects the resident dense device loop",
		"dense production checkpoints atomically bind weights, Muon/RNG/data state",
		"`accumulation=0` is the only legal publication boundary",
		"compiled multimodal training authority binds dataset processor modalities",
		"no real Qwen3.5, Gemma E4B or Gemma4 12B artifact has completed",
		"real multimodal processor/projector/codec gradient",
	} {
		if !strings.Contains(normalized, fact) {
			t.Errorf("training plan lacks current fact %q", fact)
		}
	}
	for _, stale := range []string{"backward is host-only today", "already resident-trainable", "open, done or blocked"} {
		if strings.Contains(doc, stale) {
			t.Fatalf("training plan retains stale claim %q", stale)
		}
	}
	production := read("cmd/train/run_cuda_windows.go")
	if !strings.Contains(production, "TrainDeviceResident") || strings.Contains(production, "TrainDevice"+"Full") {
		t.Fatal("production trainer does not exclusively select resident execution")
	}
	if !strings.Contains(read("internal/densecausal/train_device_resident_cuda_windows.go"), "func (m *Model) TrainDeviceResidentBatches") {
		t.Fatal("resident trainer absent")
	}
	checkpoint := read("internal/trainingprogram/checkpoint.go")
	if !strings.Contains(checkpoint, "func PublishCheckpoint") || !strings.Contains(checkpoint, "func LoadCheckpoint") {
		t.Fatal("production checkpoint owner absent")
	}
}
