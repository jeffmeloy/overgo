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
	"overgo/internal/overgodbtest"
	"overgo/internal/testcheck"
	"overgo/internal/testskip"
)

// mmluProSuite is the store's full multi-domain MMLU-Pro suite; coverage is
// accepted per model against its whole declared denominator, both protocols.
const mmluProSuite = "store/mmlu-pro"

// minMMLUProCovered is the floor of servable models that must carry complete
// MMLU-Pro coverage, so acceptance is never vacuous: the small and mid text
// models, and since 2026-10-13 the 27B, which carries both readings of this part.
const minMMLUProCovered = 6

// smokeOraclePath declares, per servable model, the protocol validate -smoke
// verifies it under (against the recipe the model is active under); chat is true for a model served
// through its own chat template. It is the declaration that tells a chat text
// model from a text decoder built for something else.
const smokeOraclePath = "../../docs/verification/smoke-oracles.json"

// smokeProtocol is the part of a smoke oracle declaration this acceptance
// reads: the model and whether it serves through the chat template. A
// declaration names no recipe; it binds the model's active one.
type smokeProtocol struct {
	Model artifact.ID `json:"model"`
	Chat  bool        `json:"chat"`
}

// TestAcceptedSmallModelMMLUPro accepts the small-model MMLU-Pro coverage: every
// servable chat model scored on the suite carries it under both prompting
// protocols with an accuracy metric over the full declared denominator. A chat
// model with only one protocol is incomplete and fails. A model whose smoke
// declaration serves its active recipe without the chat template (a base
// model, Unlimited-OCR) owes the raw-completion anchor alone and is reported,
// not counted, when that is all it carries. A model with no MMLU-Pro record,
// a non-text model, is not required here.
func TestAcceptedSmallModelMMLUPro(t *testing.T) {
	if os.Getenv(dataroot.Env) == "" {
		testskip.NotApplicable(t, "integration: set OVERGO_DATA_ROOT for MMLU-Pro coverage acceptance")
	}
	data := testcheck.Must(os.ReadFile(smokeOraclePath))(t)
	var declarations []smokeProtocol
	testcheck.NoError(t, json.Unmarshal(data, &declarations))
	declared := map[artifact.ID]smokeProtocol{}
	for _, declaration := range declarations {
		declared[declaration.Model] = declaration
	}
	root := testcheck.Must(dataroot.StoreRoot(""))(t)
	store := overgodbtest.OpenReadOnly(t, root)
	ctx := t.Context()
	names, descriptors := evaluation.DerivedSuiteCatalog(ctx, store, evaluation.ListingAuthorities())
	if descriptors[mmluProSuite].Cases == 0 {
		t.Fatalf("%s carries no declared denominator", mmluProSuite)
	}
	index := evaluation.LatestEvidence(ctx, store, names, 0)
	entries := testcheck.Must(discovery.ServableWithMemo(ctx, store, 100000, discovery.LoadMemo(ctx, store)))(t)
	covered := 0
	for _, entry := range entries {
		if !entry.Present || entry.Stale != "" || entry.Location == "" || entry.Trained {
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
			continue // no MMLU-Pro record: a non-text model.
		}
		// Fail closed: a model is exempt from the chat protocol only by a
		// declaration that serves it without the template, which binds its
		// active recipe.
		declaration, found := declared[entry.Model]
		chat := !found || declaration.Chat
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
	t.Logf("MMLU-Pro accepted: %d servable models scored on the full %d-case suite under both protocols", covered, descriptors[mmluProSuite].Cases)
}
