package main

import (
	"cmp"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/discovery"
	"overgo/internal/evaluation"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

// This join reuses committed acceptance, not model execution or a new evaluator.
// Original protocol scopes do not confer current-source or performance credit.
type textVisionCoverage struct {
	Version uint16
	Scope   string
	Models  []textVisionModel
	Proofs  []acceptedGateProof
	Inputs  map[string]string
}

type textVisionModel struct {
	Model   artifact.ID
	Domains []string
	Cells   []textVisionCell
}

type textVisionCell struct {
	Task   recipe.Task
	Recipe artifact.ID
	Proofs []string
	Scope  string
}

type acceptedGateProof struct {
	Reference, Commit, Verify string
	Preparation, Result       artifact.ID
	Check                     string `json:",omitzero"`
}

func checkAcceptedDenominator(ctx context.Context, store artifact.Reader, expected []textVisionModel, live []discovery.CatalogEntry) error {
	type key struct {
		model artifact.ID
		task  recipe.Task
	}
	wanted := map[key]textVisionCell{}
	domains := map[artifact.ID][]string{}
	for _, model := range expected {
		if _, found := domains[model.Model]; found || model.Model.Kind() != artifact.KindModel || len(model.Cells) == 0 {
			return errors.New("text/vision: empty or duplicate model")
		}
		domains[model.Model] = model.Domains
		for _, cell := range model.Cells {
			k := key{model.Model, cell.Task}
			if _, found := wanted[k]; found || cell.Recipe.Kind() != artifact.KindRecipe || len(cell.Proofs) == 0 || cell.Scope == "" {
				return errors.New("text/vision: invalid or duplicate required cell")
			}
			wanted[k] = cell
		}
	}
	if len(wanted) == 0 {
		return errors.New("text/vision: required denominator absent")
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
				return fmt.Errorf("text/vision: missing, duplicate or changed activation %s/%s", entry.Model, capability.Task)
			}
			definition, err := recipe.RequireDefinition(ctx, store, capability.Recipe)
			if err != nil {
				return err
			}
			model, found := definition.PrimaryDependency(recipe.DependencyModel)
			if !found || model != entry.Model || definition.Task != capability.Task {
				return errors.New("text/vision: recipe input or task changed")
			}
			delete(wanted, k)
		}
	}
	if len(wanted) != 0 {
		return fmt.Errorf("text/vision: %d required activations missing", len(wanted))
	}
	return nil
}

func checkAcceptedGate(proof acceptedGateProof, gate runrecord.GateResult) error {
	if gate.ID != proof.Result || gate.CodeCommit != proof.Commit || gate.Outcome != runrecord.OutcomeSucceeded {
		return errors.New("text/vision: acceptance result failed or source changed")
	}
	for _, step := range gate.Steps {
		if step.Name == cmp.Or(proof.Check, "acceptance") && (step.Outcome == runrecord.StepSucceeded || step.Outcome == runrecord.StepReused) {
			return runrecord.VerifyCompletionAcceptanceEvidence(step.Evidence, proof.Reference, proof.Verify)
		}
	}
	return errors.New("text/vision: required acceptance was not completed")
}

func TestAcceptedTextVisionEvidence(t *testing.T) {
	if testing.Short() {
		t.Skip(testskip.ShortIntegration)
	}
	root := testutil.RepoRoot(t)
	path := filepath.Join(root, "docs/verification/text-vision-coverage.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	// Freeze the reviewed protocol-to-proof assignments, not just a row count.
	if fmt.Sprintf("%x", sha256.Sum256(data)) != "4da8f01ca386f1943401ec58455a682759fa86aa6b7e661bb1e9c8c04ac0d261" {
		t.Fatal("text/vision: reviewed acceptance assignments changed")
	}
	var coverage textVisionCoverage
	if err := jsonfile.DecodeStrict(path, &coverage); err != nil {
		t.Fatal(err)
	}
	if coverage.Version != artifact.InitialDocumentVersion || coverage.Scope == "" {
		t.Fatal("text/vision: acceptance scope absent")
	}
	for path, digest := range coverage.Inputs {
		if !filepath.IsLocal(path) {
			t.Fatal("text/vision: nonlocal declaration")
		}
		data, err := os.ReadFile(filepath.Join(root, path))
		if err != nil || fmt.Sprintf("%x", sha256.Sum256(data)) != digest {
			t.Fatalf("text/vision: accepted declaration changed: %s: %v", path, err)
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
	defer store.Close()
	// Reuse the census bound; refuse truncation rather than accept a partial set.
	entries, truncated, err := discovery.CapabilityCatalogForTasks(t.Context(), store, mediaCatalogLimit, discovery.LoadMemo(t.Context(), store), recipe.TaskInference, recipe.TaskProjection, recipe.TaskVQA, recipe.TaskEmbedding, recipe.TaskRerank)
	if err != nil || truncated {
		t.Fatalf("text/vision: live denominator unavailable or truncated: %v", err)
	}
	if err := checkAcceptedDenominator(t.Context(), store, coverage.Models, entries); err != nil {
		t.Fatal(err)
	}
	proofs := map[string]bool{}
	for _, proof := range coverage.Proofs {
		if proofs[proof.Reference] {
			t.Fatal("text/vision: duplicate acceptance proof")
		}
		proofs[proof.Reference] = true
		requireAcceptedGate(t, store, proof)
	}
	cells := 0
	for _, model := range coverage.Models {
		for _, cell := range model.Cells {
			cells++
			for _, reference := range cell.Proofs {
				if !proofs[reference] {
					t.Fatalf("text/vision: unbound proof %s", reference)
				}
			}
		}
	}
	for _, mutation := range []string{"omitted-cell", "duplicate-cell", "changed-recipe", "new-embedding", "inactive-registration"} {
		t.Run(mutation, func(t *testing.T) {
			changed := slices.Clone(entries)
			// First text model; the DNA-only row remains unchanged.
			index := slices.IndexFunc(changed, func(e discovery.CatalogEntry) bool { return e.Model == coverage.Models[0].Model })
			changed[index].Capabilities = slices.Clone(changed[index].Capabilities)
			switch mutation {
			case "omitted-cell":
				changed[index].Capabilities = nil
			case "duplicate-cell":
				changed[index].Capabilities = append(changed[index].Capabilities, changed[index].Capabilities[0])
			case "changed-recipe":
				changed[index].Capabilities[0].Recipe = coverage.Models[1].Cells[0].Recipe
			case "new-embedding":
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
}

func requireAcceptedGate(t *testing.T, store *overgodb.Store, proof acceptedGateProof) {
	t.Helper()
	finalized, found, err := runrecord.GateFinalizationForPreparation(t.Context(), store, proof.Preparation)
	if err != nil || !found || finalized.CodeCommit != proof.Commit || finalized.Outcome != runrecord.OutcomeSucceeded || finalized.Result == nil || *finalized.Result != proof.Result {
		t.Fatalf("text/vision: %s lacks unique successful finalization: %v", proof.Reference, err)
	}
	gate, err := runrecord.RequireGateResult(t.Context(), store, proof.Result)
	if err != nil {
		t.Fatal(err)
	}
	if err := checkAcceptedGate(proof, gate); err != nil {
		t.Fatalf("%s: %v", proof.Reference, err)
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
