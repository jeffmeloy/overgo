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
		"Device backward and device Newton–Schulz are implemented",
		"`TrainingRunPlan` and `TrainingProgram` are design contracts, not implemented Go types",
		"`cmd/train` still selects `TrainDeviceFull`, not `TrainDeviceResident`",
		"production checkpoints are not atomic complete-state resumes",
		"compiled multimodal training authority is not implemented",
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
	if !strings.Contains(read("cmd/train/run_cuda_windows.go"), "TrainDeviceFull") {
		t.Fatal("production trainer changed without training-plan update")
	}
	if !strings.Contains(read("internal/densecausal/train_device_resident_cuda_windows.go"), "func (m *Model) TrainDeviceResident") {
		t.Fatal("resident trainer absent")
	}
}
