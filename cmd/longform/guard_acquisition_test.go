package main

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/longform"
	"overgo/internal/overgodb"
	"overgo/internal/testskip"
)

// TestAcceptedFP8Guard checks the complete cohort; it never loads a model.
func TestAcceptedFP8Guard(t *testing.T) {
	selected := readGuardCatalog(t)
	records := selected.Cohorts["Gemma 12B FP8"]
	if len(records) != len(guardCohort{}.repeats) {
		t.Fatal("FP8 acceptance requires exactly three records")
	}
	requireGuardCohorts(t, []guardCohort{{
		name:       "Gemma 12B FP8",
		historical: "evidence:sha256:e28cc3d5a6f8d19d984063e8f3f7c9893a582a85e31d770115f406a713422bf4",
		repeats:    [3]string(records),
	}}, selected.Producer, false)
	t.Log("one exact FP8 model, three complete current-source guard repeats; other models, modalities and full benchmarks remain separate")
}

// TestRetiredGuardRecords preserves historical evidence without live credit.
func TestRetiredGuardRecords(t *testing.T) {
	selected := readGuardCatalog(t)
	name := "MiniCPM 1B retired template-less"
	records := selected.Cohorts[name]
	if len(records) != len(guardCohort{}.repeats) {
		t.Fatal("retired MiniCPM requires its three immutable records")
	}
	requireGuardCohorts(t, []guardCohort{{
		name: name, retired: true,
		historical: "evidence:sha256:4a9aeeabfe52471cd86826b4857b2ece592b6f6c7eb75334a5b203387a1a5d5f",
		repeats:    [3]string(records),
	}}, selected.Producer, false)
}

type guardCatalog struct {
	Producer string              `json:"producer"`
	Cohorts  map[string][]string `json:"cohorts"`
	Pending  map[string]struct {
		Record  string `json:"record"`
		Finding string `json:"finding"`
	} `json:"pending,omitzero"`
	Prior struct {
		Producer string              `json:"producer"`
		Cohorts  map[string][]string `json:"cohorts"`
	} `json:"prior"`
}

func readGuardCatalog(t *testing.T) guardCatalog {
	t.Helper()
	if os.Getenv(testskip.StoreAcceptanceEnv) == "" {
		t.Skip(testskip.StoreAcceptance)
	}
	if testing.Short() {
		t.Skip(testskip.ShortIntegration)
	}
	if os.Getenv(dataroot.Env) == "" {
		t.Skip("integration: set OVERGO_DATA_ROOT for complete guard catalog acceptance")
	}
	var selected guardCatalog
	if err := jsonfile.Decode(filepath.Join("..", "..", "docs", "verification", "guard-catalog.json"), &selected); err != nil {
		t.Fatal(err)
	}
	return selected
}

// TestAcceptedGuardCatalog binds the live text denominator and fixed historical
// references; the selection cannot discard either.
func TestAcceptedGuardCatalog(t *testing.T) {
	selected := readGuardCatalog(t)
	if len(selected.Pending) != 0 {
		t.Fatal("full catalog admission has unresolved model obligations")
	}
	requireGuardCohorts(t, selectedGuardCohorts(t, selected), selected.Producer, true)
}

func selectedGuardCohorts(t *testing.T, selected guardCatalog) []guardCohort {
	t.Helper()
	fixtures := []guardCohort{
		{name: "Qwen 0.5B", historical: "evidence:sha256:0a6076ca812d7b2d0eb8bc17eef3fda6913ecec8dfbfd0e366e14f42eba12e51"},
		{name: "E4B", historical: "evidence:sha256:4d51503d37ae7b10d80a3cb939777390473324d59a814383fcfdb48cc7af635a"},
		{name: "MiniCPM 1B", initialModel: "model:sha256:3007c05b8ece556726a37980069cf6c0f1f966a48572b1c130c5643810c23a32"},
		{name: "MiniCPM 1B retired template-less", historical: "evidence:sha256:4a9aeeabfe52471cd86826b4857b2ece592b6f6c7eb75334a5b203387a1a5d5f", retired: true},
		{name: "Qwen 3.5 4B", historical: "evidence:sha256:198df790d9f4ff6179087f8a823115448eb45c7c51739625fd7388e71a19b160"},
		{name: "Qwen 3.5 9B", historical: "evidence:sha256:b0977a1f38b87d56459e389a68dc315a2a76a0e38c3a71afa320d4943c6af7fe"},
		{name: "Gemma 12B FP8", historical: "evidence:sha256:e28cc3d5a6f8d19d984063e8f3f7c9893a582a85e31d770115f406a713422bf4"},
		// Legacy admission records lack the full-budget protocol. These exact
		// models need an initial complete cohort, not a relabeled old pass.
		{name: "Gemma 12B BF16", initialModel: "model:sha256:f5e632cd3f6050ab8ae06df55766de5737282689d211f3e2f172db52ff671401"},
		{name: "Qwen 27B quantized", initialModel: "model:sha256:73dc8d6fd4500f7b9bb76b801e4de3c0e8df971c86163e3cd348e64e2bc74ea6"},
	}
	if len(selected.Cohorts) != len(fixtures) {
		t.Fatal("catalog selection must name eight live models and the retained retired MiniCPM cohort")
	}
	if selected.Prior.Producer != "43a2c7aa9858294d16367af109a8c248bcd6498c" || len(selected.Prior.Cohorts) != len(fixtures) {
		t.Fatal("catalog must retain the complete prior 43a2c7aa selection")
	}
	for index := range fixtures {
		fixture := &fixtures[index]
		records := selected.Cohorts[fixture.name]
		if len(records) != len(fixture.repeats) {
			t.Fatalf("%s requires three complete immutable records", fixture.name)
		}
		fixture.repeats = [3]string(records)
		previous := selected.Prior.Cohorts[fixture.name]
		if fixture.name == "MiniCPM 1B" {
			if len(previous) != 0 {
				t.Fatal("corrected MiniCPM had no accepted prior cohort")
			}
			continue
		}
		if len(previous) != len(fixture.prior) {
			t.Fatalf("%s lost prior accepted records", fixture.name)
		}
		if fixture.retired {
			if !slices.Equal(previous, records) {
				t.Fatal("retired MiniCPM records were replaced")
			}
			continue
		}
		fixture.prior, fixture.priorProducer = [3]string(previous), selected.Prior.Producer
	}
	return fixtures
}

// Accept completed cohorts without granting the pending model live credit.
func TestAcceptedCompletedGuardCohorts(t *testing.T) {
	selected := readGuardCatalog(t)
	fixtures := selectedGuardCohorts(t, selected)
	if len(selected.Pending) == 0 {
		requireGuardCohorts(t, fixtures, selected.Producer, true)
		return
	}
	const pendingModel = "Qwen 0.5B"
	pending, ok := selected.Pending[pendingModel]
	if !ok || len(selected.Pending) != 1 || pending.Record != "evidence:sha256:ae452b31fbf6b55365f91881746c35e2f5ad0a275a6063b8b59be8195b5bf2f9" ||
		pending.Finding != "evidence:sha256:7ff7c0b6406545604fa20777966f7baf30e043b66ebf92180f1043c6fb1c2d92" ||
		!slices.Contains(selected.Cohorts[pendingModel], pending.Record) {
		t.Fatal("partial admission must retain the exact pending Qwen record and repair finding")
	}
	roots, err := dataroot.Resolve(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(roots.Store)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	id, err := artifact.ParseID(pending.Record)
	if err != nil {
		t.Fatal(err)
	}
	failed, err := longform.ReadBaseline(t.Context(), store, id)
	if err != nil {
		t.Fatal(err)
	}
	if err := validateGuard(failed.Result); err != nil {
		t.Fatal(err)
	}
	if err := checkGuardShortBenchmark(t.Context(), store, failed.Result); err != nil {
		t.Fatal(err)
	}
	refused := false
	for _, text := range selected.Prior.Cohorts[pendingModel] {
		id, err := artifact.ParseID(text)
		if err != nil {
			t.Fatal(err)
		}
		prior, err := longform.ReadBaseline(t.Context(), store, id)
		if err != nil {
			t.Fatal(err)
		}
		verdict := longform.Compare(prior.Result, failed.Result, prior.Result.Floors, prior.Result.Floors.CheckRungCeiling)
		for _, reason := range verdict.Reasons {
			refused = refused || strings.HasPrefix(reason, "short: prompt ")
		}
	}
	if !refused {
		t.Fatal("pending record no longer reproduces the short-prefill refusal")
	}
	fixtures = slices.DeleteFunc(fixtures, func(fixture guardCohort) bool { return fixture.name == pendingModel })
	requireGuardCohorts(t, fixtures, selected.Producer, false)
	t.Log("seven live model cohorts accepted; retired history retained; Qwen 0.5B and full-catalog admission remain pending; no models executed")
}
