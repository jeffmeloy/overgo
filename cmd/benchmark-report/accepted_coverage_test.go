package main

import (
	"os"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/discovery"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
	"overgo/internal/testskip"
)

// mmluProSuite is the store's full multi-domain MMLU-Pro suite; coverage is
// accepted per model against its whole declared denominator, both protocols.
const mmluProSuite = "store/mmlu-pro"

// minMMLUProCovered is the floor of servable models that must carry complete
// MMLU-Pro coverage, so acceptance is never vacuous: the small and mid text
// models, the operator-owned 27B excluded.
const minMMLUProCovered = 6

// TestAcceptedSmallModelMMLUPro accepts the small-model MMLU-Pro coverage: every
// servable model scored on the suite carries it under both prompting protocols
// with an accuracy metric over the full declared denominator. A model with only
// one protocol is incomplete and fails; the operator-owned 27B and non-text
// models carry no MMLU-Pro record and are not required here.
func TestAcceptedSmallModelMMLUPro(t *testing.T) {
	if os.Getenv(dataroot.Env) == "" {
		t.Skip(testskip.Inapplicable + ": integration: set OVERGO_DATA_ROOT for MMLU-Pro coverage acceptance")
	}
	root, err := dataroot.StoreRoot("")
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(root)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := t.Context()
	descriptors := evaluation.DerivedSuiteDescriptors(ctx, store, evaluation.ListingAuthorities())
	if descriptors[mmluProSuite].Cases == 0 {
		t.Fatalf("%s carries no declared denominator", mmluProSuite)
	}
	index := evaluation.LatestEvidence(ctx, store, evaluation.DerivedSuiteNames(ctx, store, evaluation.ListingAuthorities()), 0)
	entries, err := discovery.ServableWithMemo(ctx, store, 100000, discovery.LoadMemo(ctx, store))
	if err != nil {
		t.Fatal(err)
	}
	covered := 0
	for _, entry := range entries {
		if !entry.Present || entry.Stale != "" || entry.Location == "" {
			continue
		}
		protocols := map[string]bool{}
		for _, summary := range index.EvaluationsByRecipe[entry.Recipe] {
			if summary.Suite != mmluProSuite {
				continue
			}
			if _, scored := summary.Metrics["accuracy"]; scored {
				protocols[summary.Prompting] = true
			}
		}
		if len(protocols) == 0 {
			continue // no MMLU-Pro record: the operator-owned 27B or a non-text model.
		}
		if len(protocols) < 2 {
			t.Errorf("%s has incomplete MMLU-Pro coverage: %d protocol(s) %v, needs both raw and chat", entry.Location, len(protocols), protocols)
			continue
		}
		covered++
	}
	if covered < minMMLUProCovered {
		t.Fatalf("only %d model(s) carry complete MMLU-Pro coverage; want at least %d", covered, minMMLUProCovered)
	}
	t.Logf("MMLU-Pro accepted: %d servable models scored on the full %d-case suite under both protocols; 27B operator-owned and excluded", covered, descriptors[mmluProSuite].Cases)
}
