package main

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/discovery"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/strictjson"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

func TestAcceptedTextVisionEvidence(t *testing.T) {
	if testing.Short() {
		t.Skip(testskip.ShortIntegration)
	}
	root := testutil.RepoRoot(t)
	var coverage acceptedCoverage
	requireAcceptedDocument(t, root, "docs/verification/text-vision-coverage.json", "4da8f01ca386f1943401ec58455a682759fa86aa6b7e661bb1e9c8c04ac0d261", &coverage)
	requireAcceptedCoverage(t, root, coverage, recipe.TaskInference, recipe.TaskProjection, recipe.TaskVQA, recipe.TaskEmbedding, recipe.TaskRerank)
}

func requireAcceptedCoverage(t *testing.T, root string, coverage acceptedCoverage, tasks ...recipe.Task) *overgodb.Store {
	t.Helper()
	if coverage.Version != artifact.InitialDocumentVersion || coverage.Scope == "" {
		t.Fatal("coverage: acceptance scope absent")
	}
	for path, digest := range coverage.Inputs {
		if !filepath.IsLocal(path) {
			t.Fatal("coverage: nonlocal declaration")
		}
		data, err := os.ReadFile(filepath.Join(root, path))
		if err != nil || fmt.Sprintf("%x", sha256.Sum256(data)) != digest {
			t.Fatalf("coverage: accepted declaration changed: %s: %v", path, err)
		}
	}
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(retainedReferenceStore(roots.Store))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := store.Close(); err != nil {
			t.Error(err)
		}
	})
	// Reuse the census bound; refuse truncation rather than accept a partial set.
	entries, truncated, err := discovery.CapabilityCatalogForTasks(t.Context(), store, mediaCatalogLimit, discovery.LoadMemo(t.Context(), store), tasks...)
	if err != nil || truncated {
		t.Fatalf("coverage: live denominator unavailable or truncated: %v", err)
	}
	if err := checkAcceptedDenominator(t.Context(), store, coverage.Models, entries); err != nil {
		t.Fatal(err)
	}
	proofs := map[string]bool{}
	for _, proof := range coverage.Proofs {
		if proofs[proof.Reference+"#"+proof.Check] {
			t.Fatal("coverage: duplicate acceptance proof")
		}
		proofs[proof.Reference+"#"+proof.Check] = true
		requireAcceptedGate(t, store, proof)
	}
	cells := 0
	for _, model := range coverage.Models {
		for _, cell := range model.Cells {
			cells++
			for _, reference := range cell.Proofs {
				if !strings.Contains(reference, "#") {
					reference += "#"
				}
				if !proofs[reference] {
					t.Fatalf("coverage: unbound proof %s", reference)
				}
			}
		}
	}
	for _, mutation := range []string{"omitted-cell", "duplicate-cell", "changed-recipe", "new-task", "inactive-registration"} {
		t.Run(mutation, func(t *testing.T) {
			changed := slices.Clone(entries)
			// Mutate one accepted activation; unrelated entries stay unchanged.
			index := slices.IndexFunc(changed, func(e discovery.CatalogEntry) bool { return e.Model == coverage.Models[0].Model })
			changed[index].Capabilities = slices.Clone(changed[index].Capabilities)
			switch mutation {
			case "omitted-cell":
				changed[index].Capabilities = nil
			case "duplicate-cell":
				changed[index].Capabilities = append(changed[index].Capabilities, changed[index].Capabilities[0])
			case "changed-recipe":
				changed[index].Capabilities[0].Recipe = artifact.ID{}
			case "new-task":
				changed[index].Capabilities = append(changed[index].Capabilities, discovery.Capability{Task: recipe.TaskEmbedding})
			case "inactive-registration":
				changed = append(changed, discovery.CatalogEntry{Model: testutil.ArtifactID(t, artifact.KindModel, "inactive")})
			}
			err := checkAcceptedDenominator(t.Context(), store, coverage.Models, changed)
			if (err == nil) != (mutation == "inactive-registration") {
				t.Fatalf("coverage verdict: %v", err)
			}
		})
	}
	t.Logf("%d models, %d activations, %d exact committed acceptance proofs; original protocol scopes retained; model executions=0", len(coverage.Models), cells, len(coverage.Proofs))
	return store
}

func requireAcceptedGate(t *testing.T, store *overgodb.Store, proof acceptedGateProof) {
	t.Helper()
	gate, err := loadAcceptedGate(t.Context(), store, proof)
	if err != nil {
		t.Fatal(err)
	}
	for _, mutation := range []string{"failed", "skipped", "wrong-source", "wrong-verifier", "wrong-check", "missing"} {
		t.Run(proof.Reference+"/"+mutation, func(t *testing.T) {
			changed, changedProof := gate, proof
			changed.Steps = slices.Clone(gate.Steps)
			switch mutation {
			case "failed":
				changed.Outcome = runrecord.OutcomeFailed
			case "skipped":
				for i := range changed.Steps {
					changed.Steps[i].Outcome = runrecord.StepSkipped
				}
			case "wrong-source":
				changedProof.Commit = "foreign source"
			case "wrong-verifier":
				changedProof.Verify = "go test ./cmd/compatibility -run '^MissingAssertion$'"
			case "wrong-check":
				changedProof.Check = "absent-check"
			case "missing":
				changed.Steps = nil
			}
			if checkAcceptedGate(changedProof, changed) == nil {
				t.Fatal("invalid acceptance received credit")
			}
		})
	}
}

func requireAcceptedDocument(t *testing.T, root, path, digest string, target any) {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(root, path))
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprintf("%x", sha256.Sum256(data)) != digest {
		t.Fatal("reviewed acceptance assignments changed", path)
	}
	if err := strictjson.DecodeBytes(data, target); err != nil {
		t.Fatal(err)
	}
}

func TestAcceptedSpecializedTaskEvidence(t *testing.T) {
	if testing.Short() {
		t.Skip(testskip.ShortIntegration)
	}
	root := testutil.RepoRoot(t)
	var coverage acceptedCoverage
	requireAcceptedDocument(t, root, "docs/verification/specialized-coverage.json", "700b2bd046116527c6c514a6cbcce080d02d685ad1b939a3d51c981e020458a8", &coverage)
	requireAcceptedCoverage(t, root, coverage, recipe.TaskForecast, recipe.TaskTabular, recipe.TaskSeq2Seq, recipe.TaskSpeech, recipe.TaskTranscription, recipe.TaskAlignment, recipe.TaskDiarization, recipe.TaskActivityDetection, recipe.TaskAudioConversion, recipe.TaskAudioGeneration)
}

func checkAcceptedDenominator(ctx context.Context, store artifact.Reader, expected []acceptedModel, live []discovery.CatalogEntry) error {
	type key struct {
		model artifact.ID
		task  recipe.Task
	}
	wanted := map[key]acceptedCell{}
	domains := map[artifact.ID][]string{}
	for _, model := range expected {
		if _, found := domains[model.Model]; found || model.Model.Kind() != artifact.KindModel || len(model.Cells) == 0 {
			return errors.New("coverage: empty or duplicate model")
		}
		domains[model.Model] = model.Domains
		for _, cell := range model.Cells {
			k := key{model.Model, cell.Task}
			if _, found := wanted[k]; found || cell.Recipe.Kind() != artifact.KindRecipe || len(cell.Proofs) == 0 || cell.Scope == "" {
				return errors.New("coverage: invalid or duplicate required cell")
			}
			wanted[k] = cell
		}
	}
	if len(wanted) == 0 {
		return errors.New("coverage: required denominator absent")
	}
	for _, entry := range live {
		if len(entry.Capabilities) == 0 {
			continue // An inactive registration grants no credit and invalidates none.
		}
		declared, _, err := evaluation.EvalDomains(ctx, store, entry.Model)
		if err != nil {
			return err
		}
		for _, capability := range entry.Capabilities {
			if capability.Task == recipe.TaskInference && slices.Equal(declared, []string{"dna"}) {
				continue
			}
			k := key{entry.Model, capability.Task}
			cell, found := wanted[k]
			if !found || !entry.Present || capability.Stale != "" || capability.Recipe != cell.Recipe || !slices.Equal(declared, domains[entry.Model]) {
				return fmt.Errorf("coverage: missing, duplicate or changed activation %s/%s", entry.Model, capability.Task)
			}
			definition, err := recipe.RequireDefinition(ctx, store, capability.Recipe)
			if err != nil {
				return err
			}
			model, found := definition.PrimaryDependency(recipe.DependencyModel)
			if !found || model != entry.Model || definition.Task != capability.Task {
				return errors.New("coverage: recipe input or task changed")
			}
			delete(wanted, k)
		}
	}
	if len(wanted) != 0 {
		return fmt.Errorf("coverage: %d required activations missing", len(wanted))
	}
	return nil
}
