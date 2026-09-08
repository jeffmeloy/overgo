package main

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/testevidence"
)

// TestAcceptedFP8Guard checks the complete cohort; it never loads a model.
func TestAcceptedFP8Guard(t *testing.T) {
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip)
	}
	if os.Getenv(dataroot.Env) == "" {
		t.Skip("integration: set OVERGO_DATA_ROOT to check the exact FP8 guard cohort")
	}
	var selected struct {
		Producer string   `json:"producer"`
		Records  []string `json:"records"`
	}
	if err := jsonfile.Decode(filepath.Join("..", "..", "docs", "verification", "fp8-guard.json"), &selected); err != nil {
		t.Fatal(err)
	}
	if len(selected.Records) != len(guardCohort{}.repeats) {
		t.Fatal("FP8 acceptance requires exactly three records")
	}
	requireGuardCohorts(t, []guardCohort{{
		name:       "Gemma 12B FP8",
		historical: "evidence:sha256:e28cc3d5a6f8d19d984063e8f3f7c9893a582a85e31d770115f406a713422bf4",
		repeats:    [3]string(selected.Records),
	}}, selected.Producer)
	t.Log("one exact FP8 model, three complete current-source guard repeats; other models, modalities and full benchmarks remain separate")
}
