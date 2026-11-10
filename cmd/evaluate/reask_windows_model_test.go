//go:build windows && modeltest

package main

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/discovery"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// The answers the protocol's eight tokens ended, and how they are asked again (answer-format/reread): the four models
// whose MMLU-Pro readings lost most answers to the budget, each variant held apart, and the share of their
// budget-ended answers a variant must draw a letter for to be put to every model's cell. All declared before the run.
var (
	reaskModels   = []string{"Qwen3.5-9B", "gemma-4-12B-it-fp8", "gemma-4-12B-it-fp8-native", "MiniCPM5-1B"}
	reaskVariants = []evaluation.AnswerVariant{{Opener: "The answer is", Tokens: 32}, {Opener: "The answer is (", Tokens: 8}}
)

const reaskBar = 0.5

// TestTheBudgetEndedAnswersAreAskedAgain measures whether a change to the chat-template protocol recovers the answers
// its eight tokens ended. Each of reaskModels' MMLU-Pro readings on the newest scorecard is asked again, its
// budget-ended answers alone (answer-format/unanswered), under each of reaskVariants: a budget of 32 tokens after the
// same opener, and an opener that ends at the letter's place. A variant that draws a letter for at least reaskBar of
// the four models' budget-ended answers together is the protocol's change; the protocol, as ChoiceProtocol says it, is
// held to that variant where one met the bar and to its own opener and eight tokens where none did. A reading's
// budget-ended cases are published as development evidence of its dataset before the model opens (SQA-361), since the
// protocol is chosen on them, and every variant is asked on one load of the model.
func TestTheBudgetEndedAnswersAreAskedAgain(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	roots := testcheck.Must(dataroot.Resolve(testutil.RepoRoot(t)))(t)
	// The scorecards sit beside the store (scorecard.Directory), named here so the test imports no scorecard code.
	records := testcheck.Must(filepath.Glob(filepath.Join(filepath.Dir(roots.Store), "scorecards", "scorecard-*.json")))(t)
	slices.Sort(records)
	if len(records) == 0 {
		t.Fatal("no scorecard record names the readings to ask again")
	}
	type cardRow struct {
		Model artifact.ID `json:"model"`
		Name  string      `json:"name"`
		Cells map[string]struct {
			Source string `json:"source"`
		} `json:"cells"`
	}
	var card struct {
		Rows []cardRow `json:"rows"`
	}
	data := testcheck.Must(os.ReadFile(records[len(records)-1]))(t)
	testcheck.NoError(t, json.Unmarshal(data, &card))
	store := testcheck.Must(overgodb.OpenReadOnly(roots.Store))(t)
	entries, truncated, err := discovery.CapabilityCatalog(ctx, store, discovery.CatalogBound, discovery.LoadMemo(ctx, store))
	testcheck.NoError(t, errors.Join(err, store.Close()))
	asked, lettered := make([]int, len(reaskVariants)), make([]int, len(reaskVariants))
	for _, name := range reaskModels {
		at := slices.IndexFunc(card.Rows, func(row cardRow) bool { return row.Name == name })
		if at < 0 {
			t.Fatalf("the newest scorecard holds no row %s", name)
		}
		row := card.Rows[at]
		entry, found, err := discovery.MatchReference(entries, truncated, row.Model.String())
		if err != nil || !found {
			t.Fatalf("%s (%s) is not in the catalog: %v", name, row.Model, err)
		}
		all := testcheck.Must(reaskBudgetEnded(ctx, roots.Store, entry.Location, row.Cells["held-out-quality"].Source, reaskVariants))(t)
		for index, variant := range reaskVariants {
			asked[index] += all[index].Asked
			lettered[index] += all[index].Lettered
			t.Logf("%s under %q and %d tokens: %d of %d budget-ended answers name a letter", name, variant.Opener, variant.Tokens, all[index].Lettered, all[index].Asked)
		}
	}
	chosen := -1
	for index, variant := range reaskVariants {
		share := float64(lettered[index]) / float64(asked[index])
		t.Logf("%q and %d tokens: %d of %d, %.3f", variant.Opener, variant.Tokens, lettered[index], asked[index], share)
		if share >= reaskBar && (chosen < 0 || lettered[index] > lettered[chosen]) {
			chosen = index
		}
	}
	// On 2026-11-10 the opener ending at the letter's place drew a letter for 370 of the 380 within the same eight
	// tokens, and 32 tokens after the old opener for 106; applying it to the protocol re-binds every chat-template
	// plan and the retained acceptances on it, which answer-format/paren-opener does.
	if chosen < 0 || reaskVariants[chosen].Opener != "The answer is (" {
		t.Errorf("no variant, or not the opener ending at the letter's place, drew a letter for %.0f%% of the budget-ended answers", reaskBar*100)
	}
	words, _ := evaluation.ChoiceProtocol(evaluation.PromptingChatTemplate.Label())
	t.Logf("the chat-template protocol now: %s", words)
}
