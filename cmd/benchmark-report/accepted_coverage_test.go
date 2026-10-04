//go:build modeltest

package main

import (
	"encoding/json"
	"os"
	"testing"

	"overgo/internal/artifact"
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

// smokeOraclePath declares, per servable model and active recipe, the
// protocol validate -smoke verifies it under; chat is true for a model served
// through its own chat template. It is the declaration that tells a chat text
// model from a text decoder built for something else.
const smokeOraclePath = "../../docs/verification/smoke-oracles.json"

// smokeProtocol is the part of a smoke oracle declaration this acceptance
// reads: the recipe it binds and whether it serves through the chat template.
type smokeProtocol struct {
	Model  artifact.ID `json:"model"`
	Recipe artifact.ID `json:"recipe"`
	Chat   bool        `json:"chat"`
}

// TestAcceptedSmallModelMMLUPro accepts the small-model MMLU-Pro coverage: every
// servable chat model scored on the suite carries it under both prompting
// protocols with an accuracy metric over the full declared denominator. A chat
// model with only one protocol is incomplete and fails. A model whose smoke
// declaration serves its active recipe without the chat template (a base
// model, Unlimited-OCR) owes the raw-completion anchor alone and is reported,
// not counted, when that is all it carries. The operator-owned 27B and
// non-text models carry no MMLU-Pro record and are not required here.
func TestAcceptedSmallModelMMLUPro(t *testing.T) {
	if os.Getenv(dataroot.Env) == "" {
		testskip.NotApplicable(t, "integration: set OVERGO_DATA_ROOT for MMLU-Pro coverage acceptance")
	}
	data, err := os.ReadFile(smokeOraclePath)
	if err != nil {
		t.Fatal(err)
	}
	var declarations []smokeProtocol
	if err := json.Unmarshal(data, &declarations); err != nil {
		t.Fatal(err)
	}
	declared := map[artifact.ID]smokeProtocol{}
	for _, declaration := range declarations {
		declared[declaration.Model] = declaration
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
	names, descriptors := evaluation.DerivedSuiteCatalog(ctx, store, evaluation.ListingAuthorities())
	if descriptors[mmluProSuite].Cases == 0 {
		t.Fatalf("%s carries no declared denominator", mmluProSuite)
	}
	index := evaluation.LatestEvidence(ctx, store, names, 0)
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
		// Fail closed: a model is exempt from the chat protocol only by a
		// declaration of its active recipe that serves it without the template.
		declaration, found := declared[entry.Model]
		chat := !found || declaration.Recipe != entry.Recipe || declaration.Chat
		switch mmluProCoverage(chat, protocols) {
		case mmluProComplete:
			covered++
		case mmluProAnchorOnly:
			t.Logf("%s carries the raw-completion anchor alone, all it owes: its smoke declaration serves recipe %s without the chat template", entry.Location, entry.Recipe)
		default:
			owed := "both raw and chat"
			if !chat {
				owed = "the raw-completion anchor"
			}
			t.Errorf("%s has incomplete MMLU-Pro coverage: %d protocol(s) %v, needs %s", entry.Location, len(protocols), protocols, owed)
		}
	}
	if covered < minMMLUProCovered {
		t.Fatalf("only %d model(s) carry complete MMLU-Pro coverage; want at least %d", covered, minMMLUProCovered)
	}
	t.Logf("MMLU-Pro accepted: %d servable models scored on the full %d-case suite under both protocols; 27B operator-owned and excluded", covered, descriptors[mmluProSuite].Cases)
}
