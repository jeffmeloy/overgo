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
	}}, selected.Producer, false)
	t.Log("one exact FP8 model, three complete current-source guard repeats; other models, modalities and full benchmarks remain separate")
}

// TestAcceptedGuardCatalog binds the complete live text denominator. Known
// historical references are fixed here; the selection cannot discard them.
func TestAcceptedGuardCatalog(t *testing.T) {
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip)
	}
	if os.Getenv(dataroot.Env) == "" {
		t.Skip("integration: set OVERGO_DATA_ROOT for complete guard catalog acceptance")
	}
	var selected struct {
		Producer string              `json:"producer"`
		Cohorts  map[string][]string `json:"cohorts"`
	}
	if err := jsonfile.Decode(filepath.Join("..", "..", "docs", "verification", "guard-catalog.json"), &selected); err != nil {
		t.Fatal(err)
	}
	fixtures := []guardCohort{
		{name: "Qwen 0.5B", historical: "evidence:sha256:0a6076ca812d7b2d0eb8bc17eef3fda6913ecec8dfbfd0e366e14f42eba12e51"},
		{name: "E4B", historical: "evidence:sha256:4d51503d37ae7b10d80a3cb939777390473324d59a814383fcfdb48cc7af635a"},
		{name: "MiniCPM 1B", historical: "evidence:sha256:4a9aeeabfe52471cd86826b4857b2ece592b6f6c7eb75334a5b203387a1a5d5f"},
		{name: "Qwen 3.5 4B", historical: "evidence:sha256:198df790d9f4ff6179087f8a823115448eb45c7c51739625fd7388e71a19b160"},
		{name: "Qwen 3.5 9B", historical: "evidence:sha256:b0977a1f38b87d56459e389a68dc315a2a76a0e38c3a71afa320d4943c6af7fe"},
		{name: "Gemma 12B FP8", historical: "evidence:sha256:e28cc3d5a6f8d19d984063e8f3f7c9893a582a85e31d770115f406a713422bf4"},
		// Legacy admission records lack the full-budget protocol. These exact
		// models need an initial complete cohort, not a relabeled old pass.
		{name: "Gemma 12B BF16", initialModel: "model:sha256:f5e632cd3f6050ab8ae06df55766de5737282689d211f3e2f172db52ff671401"},
		{name: "Qwen 27B quantized", initialModel: "model:sha256:73dc8d6fd4500f7b9bb76b801e4de3c0e8df971c86163e3cd348e64e2bc74ea6"},
	}
	if len(selected.Cohorts) != len(fixtures) {
		t.Fatal("catalog selection must name the complete eight-model denominator")
	}
	for index := range fixtures {
		fixture := &fixtures[index]
		records := selected.Cohorts[fixture.name]
		if len(records) != len(fixture.repeats) {
			t.Fatalf("%s requires three complete immutable records", fixture.name)
		}
		fixture.repeats = [3]string(records)
	}
	requireGuardCohorts(t, fixtures, selected.Producer, true)
}
