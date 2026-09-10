package main

import (
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/longform"
	"overgo/internal/overgodb"
)

// TestGuardProducerComparison preserves the complete counterbalanced experiment,
// including earlier refusals. Passing repeats do not close the latency finding.
func TestGuardProducerComparison(t *testing.T) {
	selected := readGuardCatalog(t)
	prior := selected.Prior.Cohorts["Qwen 0.5B"]
	if len(prior) != len(guardCohort{}.prior) {
		t.Fatal("comparison lost the prior cohort")
	}
	var candidates []string
	for _, cohort := range []struct {
		producer string
		records  [3]string
	}{
		{"2ef953fab346b607d003f06734a70ea73e41a4c9", [3]string{
			"evidence:sha256:880d6ca81ed6115eff88f6366659cb61e437c951884fcb77d9d2ae3da1cff356",
			"evidence:sha256:b09c03224fe2af1e55f6530161aa782061831608352b1fe662934b908310f5a6",
			"evidence:sha256:a1a26504dd00c8c44f9ca928ea14d07f58309545c62d0a0803dcac08e60cc12d",
		}},
		{"1b20da456002d9ebe83d27b900a2263444b2f7ec", [3]string{
			"evidence:sha256:b44df9d4f8f2d436c946dd022e295564c96e1456e20be3921f199e7fa021f3f0",
			"evidence:sha256:5400886a0e20e5137add5372543a723979354c33f8d79c48acf5b3e871d148ea",
			"evidence:sha256:2d05a2445092abdc8556c709328ba911e62ebd6589c62f6c6b0751d06645ba21",
		}},
	} {
		candidates = append(candidates, cohort.records[:]...)
		t.Run(cohort.producer, func(t *testing.T) {
			requireGuardCohorts(t, []guardCohort{{
				name: "Qwen 0.5B", repeats: cohort.records,
				historicalSurface: "22be8fb72e5fc316d414215317d7f566691517d9ec8d28eb88e7ee2d72b079e2",
				historical:        "evidence:sha256:0a6076ca812d7b2d0eb8bc17eef3fda6913ecec8dfbfd0e366e14f42eba12e51",
				prior:             [3]string(prior), priorProducer: selected.Prior.Producer,
			}}, cohort.producer, false)
		})
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
	for _, text := range []string{
		"evidence:sha256:e3efb9b54dacddd5f6bf8e56827f16587333fcfef3ad6013b67d823eb83311ab",
		"evidence:sha256:307fa45e07c43130c0027ee117c80c1e4f65098172ca4f08087a60c509d87085",
	} {
		before := measuredGuardRecord(t, store, text)
		for _, text := range candidates {
			fresh := measuredGuardRecord(t, store, text)
			if verdict := longform.Compare(before.Result, fresh.Result, before.Result.Floors, before.Result.Floors.CheckRungCeiling); !verdict.Passed {
				t.Fatalf("retained decode-repair control: %s", verdict)
			}
		}
	}
	for _, text := range []string{
		"evidence:sha256:ae452b31fbf6b55365f91881746c35e2f5ad0a275a6063b8b59be8195b5bf2f9",
		"evidence:sha256:51d9ed6a8fa60500f1c511ed6a3d61b62d16b45bf2db7ebffc7b9c63b4307af3",
	} {
		id, err := artifact.ParseID(text)
		if err != nil {
			t.Fatal(err)
		}
		failed, err := longform.ReadBaseline(t.Context(), store, id)
		if err != nil {
			t.Fatal(err)
		}
		if err := checkGuardShortBenchmark(t.Context(), store, failed.Result); err != nil {
			t.Fatal(err)
		}
		refused := false
		for _, text := range prior {
			id, err := artifact.ParseID(text)
			if err != nil {
				t.Fatal(err)
			}
			before, err := longform.ReadBaseline(t.Context(), store, id)
			if err != nil {
				t.Fatal(err)
			}
			verdict := longform.Compare(before.Result, failed.Result, before.Result.Floors, before.Result.Floors.CheckRungCeiling)
			for _, reason := range verdict.Reasons {
				refused = refused || strings.HasPrefix(reason, "short: prompt ")
			}
		}
		if !refused {
			t.Fatal("earlier refusal was lost or normalized")
		}
	}
	t.Log("two producers, three complete records each; historical and repeat comparisons pass; two earlier refusals retained. No latency-finding closure or full-catalog admission; no model executions.")
}
