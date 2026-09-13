package main

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/discovery"
	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/jsonfile"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/tokenizer"
)

func TestSmokeOracleRejectsFalseGreen(t *testing.T) {
	c := evaluation.GeneratedAnswerCase{Name: "addition", Prompt: "2 + 2?", MaxTokens: 16, Answers: []string{"4"}}
	good := evaluation.ExactResult{Name: c.Name, PromptTokens: 4, GeneratedTokens: 1, Text: "4"}
	if err := acceptSmokeAnswer(c, good); err != nil {
		t.Fatal(err)
	}
	for _, text := range []string{"", c.Prompt, "Paris", "4 4 4", "{broken", "5"} {
		bad := good
		bad.Text = text
		if acceptSmokeAnswer(c, bad) == nil {
			t.Fatalf("accepted %q", text)
		}
	}
	good.GeneratedTokens = 0
	if acceptSmokeAnswer(c, good) == nil {
		t.Fatal("accepted prompt-only record")
	}
}

func TestSmokeDeclarationDenominator(t *testing.T) {
	model, _ := artifact.IdentifyBytes(artifact.KindModel, []byte("model"))
	recipe, _ := artifact.IdentifyBytes(artifact.KindRecipe, []byte("recipe"))
	oracle := smokeOracle{Model: model, Recipe: recipe, Domain: "text", Chat: true, Claim: "arithmetic", Generation: &evaluation.GeneratedAnswerCase{Name: "addition", Prompt: "2+2?", MaxTokens: 16, Answers: []string{"4"}}}
	entries := []discovery.Entry{{Model: model, Recipe: recipe, Present: true}}
	if _, err := bindSmokeOracles([]smokeOracle{oracle}, entries); err != nil {
		t.Fatal(err)
	}
	if _, err := bindSmokeOracles(nil, entries); err == nil {
		t.Fatal("accepted omitted case")
	}
	if _, err := bindSmokeOracles([]smokeOracle{oracle, oracle}, append(slices.Clone(entries), entries[0])); err == nil {
		t.Fatal("accepted duplicate")
	}
	for _, mutate := range []func(*discovery.Entry){func(e *discovery.Entry) { e.Stale = "changed" }, func(e *discovery.Entry) { e.Present = false }, func(e *discovery.Entry) { e.Recipe = artifact.ID{} }} {
		changed := slices.Clone(entries)
		mutate(&changed[0])
		if _, err := bindSmokeOracles([]smokeOracle{oracle}, changed); err == nil {
			t.Fatal("accepted stale or missing activation")
		}
	}
}

func TestSmokeNativeReferenceRejectsWrongScoring(t *testing.T) {
	reference := smokeReferenceCase{Text: "native", IDs: []tokenizer.TokenID{1, 2, 3}, LogProbabilities: []float64{-1, -2}}
	good := inference.PerplexityResult{TokenCount: 3, EvaluatedTokens: 2, Scores: []inference.TokenScore{{Position: 1, TokenID: 2, NegativeLogLik: 1}, {Position: 2, TokenID: 3, NegativeLogLik: 2}}}
	if err := acceptSmokeScores(reference, good); err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*inference.PerplexityResult){func(r *inference.PerplexityResult) { r.Scores = r.Scores[:1] }, func(r *inference.PerplexityResult) { r.Scores[1].TokenID = 4 }, func(r *inference.PerplexityResult) { r.Scores[1].NegativeLogLik = 0 }, func(r *inference.PerplexityResult) { r.Scores[1].NegativeLogLik = 1 }} {
		bad := good
		bad.Scores = slices.Clone(good.Scores)
		mutate(&bad)
		if acceptSmokeScores(reference, bad) == nil {
			t.Fatal("accepted failed scoring outcome")
		}
	}
}

// Declarations and counterexamples need no device. The lane checks live catalog
// identity and executes every case before the plan can close.
func TestAcceptedSmokeOracles(t *testing.T) {
	root := filepath.Join("..", "..")
	var declarations []smokeOracle
	if err := jsonfile.DecodeStrict(filepath.Join(root, smokeOraclePath), &declarations); err != nil {
		t.Fatal(err)
	}
	entries := make([]discovery.Entry, len(declarations))
	text, dna, chat, referenceCases := 0, 0, 0, 0
	for i, d := range declarations {
		if _, err := artifact.JSONID(artifact.KindProfile, d); err != nil {
			t.Fatal(err)
		}
		entries[i] = discovery.Entry{Model: d.Model, Recipe: d.Recipe, Present: true}
		switch d.Domain {
		case "text":
			text++
		case "dna":
			dna++
		default:
			t.Fatal("undeclared domain")
		}
		if d.Chat {
			chat++
		}
		if d.ReferenceFile != "" {
			d.ReferenceFile = filepath.Join(root, d.ReferenceFile)
			cases, err := readSmokeReference(d)
			if err != nil {
				t.Fatal(err)
			}
			referenceCases += len(cases)
		}
	}
	if text != 8 || dna != 1 || chat != 7 || referenceCases != 4 {
		t.Fatalf("denominator text=%d DNA=%d chat=%d native=%d", text, dna, chat, referenceCases)
	}
	referenceStore := os.Getenv("OVERGO_SMOKE_REFERENCE_STORE")
	selectedReference := referenceStore != ""
	if referenceStore != "" || os.Getenv(dataroot.Env) != "" {
		if referenceStore == "" {
			roots, err := dataroot.Resolve(root)
			if err != nil {
				t.Fatal(err)
			}
			referenceStore = roots.Store
		}
		store, err := overgodb.OpenReadOnly(referenceStore)
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		catalog, err := store.Query(t.Context(), overgodb.Query{Kind: artifact.KindModel, MaxResults: 10_000, Projection: overgodb.ProjectManifests})
		if err != nil || catalog.Truncated {
			t.Fatalf("live catalog: %v; truncated=%t", err, catalog.Truncated)
		}
		entries = nil
		extra := 0
		for _, manifest := range catalog.Manifests {
			active, err := modelrecipe.HasActiveRecipe(t.Context(), store, manifest.ID, recipe.TaskInference)
			if err != nil {
				t.Fatal(err)
			}
			if !active {
				continue
			}
			// An explicit external fixture store can acquire unrelated models
			// after this revision. Verify every declared identity without
			// claiming coverage of those additional activations. The normal
			// data-root path and production lane still require the full catalog.
			if selectedReference && !slices.ContainsFunc(declarations, func(d smokeOracle) bool { return d.Model == manifest.ID }) {
				extra++
				continue
			}
			record, found, err := modelrecipe.ActiveRecord(t.Context(), store, manifest.ID, recipe.TaskInference)
			if err != nil || !found {
				t.Fatalf("live activation %s: %v", manifest.ID, err)
			}
			entries = append(entries, discovery.Entry{Model: manifest.ID, Recipe: record.Definition.ID, Present: true})
		}
		t.Logf("%d declared model/recipe identities checked read-only; %d additional reference activations receive no evidence credit; lane verifies artifact bytes", len(entries), extra)
	}
	if _, err := bindSmokeOracles(declarations, entries); err != nil {
		t.Fatal(err)
	}
	t.Run("false_green", TestSmokeOracleRejectsFalseGreen)
	t.Run("denominator", TestSmokeDeclarationDenominator)
	t.Run("native", TestSmokeNativeReferenceRejectsWrongScoring)
	t.Log("9 inference activations: 7 exact-answer cases and 4 native scoring cases; other task/modality evidence remains in modality-verification rows")
}

func TestSmokePublishesCompleteBoundRecord(t *testing.T) {
	model, _ := artifact.IdentifyBytes(artifact.KindModel, []byte("model"))
	recipe, _ := artifact.IdentifyBytes(artifact.KindRecipe, []byte("recipe"))
	path := filepath.Join(t.TempDir(), "store")
	store, err := overgodb.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	_, err = store.Commit(t.Context(), artifact.Batch{Key: "smoke-test-inputs", Artifacts: []artifact.Descriptor{{ID: model}, {ID: recipe}}})
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	oracle := smokeOracle{Model: model, Recipe: recipe, Domain: "text", Chat: true, Claim: "addition", Generation: &evaluation.GeneratedAnswerCase{Name: "addition", Prompt: "2+2?", MaxTokens: 16, Answers: []string{"4"}}}
	observation := smokeObservation{Claim: oracle.Claim, Generation: &evaluation.ExactResult{Name: "addition", PromptTokens: 4, GeneratedTokens: 1, Text: "4"}}
	entry := discovery.Entry{Model: model, Recipe: recipe, Present: true}
	if err := recordSmokeTransaction(t.Context(), path, entry, oracle, observation, nil, time.Second, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"); err != nil {
		t.Fatal(err)
	}
}
