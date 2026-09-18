package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
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
	"overgo/internal/jsonfile"
	"overgo/internal/modelartifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/strictjson"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

func TestAcceptedModalityCoverage(t *testing.T) {
	if testing.Short() {
		t.Skip(testskip.ShortIntegration)
	}
	if os.Getenv(dataroot.Env) == "" {
		t.Skip("integration: set OVERGO_DATA_ROOT for retained modality coverage")
	}
	t.Run("case-denominators", TestProtocolCaseProjection)
	t.Run("resource-denominators", TestResourceCaseProjection)
	output := acceptedProtocolCoverageProjection(t)
	// Current resource admission stays separate; this checks the original source.
	checkE4BResourceRefresh(t, false)
	root := testutil.RepoRoot(t)
	for path, want := range map[string][]byte{mediaReportPath: output.Markdown, mediaProjectionPath: output.JSON} {
		got, err := os.ReadFile(filepath.Join(root, path))
		if err != nil {
			t.Fatal(err)
		}
		got = bytes.ReplaceAll(got, []byte("\r\n"), []byte("\n"))
		if !bytes.Equal(got, want) {
			index := 0
			for index < len(got) && index < len(want) && got[index] == want[index] {
				index++
			}
			before, _, _ := bytes.Cut(got[index:], []byte("\n"))
			after, _, _ := bytes.Cut(want[index:], []byte("\n"))
			t.Fatalf("%s is stale at byte %d: stored line suffix=%q derived=%q; regenerate with compatibility -update-media", path, index, before, after)
		}
	}
	var projection mediaProjection
	if err := json.Unmarshal(output.JSON, &projection); err != nil {
		t.Fatal(err)
	}
	var generation imageVideoProtocol
	if err := jsonfile.DecodeStrict(filepath.Join(root, imageVideoProtocolPath), &generation); err != nil {
		t.Fatal(err)
	}
	for _, declared := range generation.Cases {
		matches := 0
		for _, model := range projection.Coverage.Models {
			for _, p := range model.Protocols {
				if p.Name == declared.ID && model.Model == declared.Model && p.Recipe == declared.Recipe && p.Unit == "generation case" && p.Required != nil && *p.Required == 1 && p.Recorded != nil && *p.Recorded == 1 {
					matches++
				}
			}
		}
		if matches != 1 {
			t.Fatalf("generation case %s: projected %d times", declared.ID, matches)
		}
	}
	var source modelValidationSpecification
	if err := jsonfile.DecodeStrict(filepath.Join(root, "docs/verification/e4b-validation.json"), &source); err != nil {
		t.Fatal(err)
	}
	index := slices.IndexFunc(projection.Coverage.Models, func(row modalityModelProjection) bool { return row.Model == source.Model })
	if index < 0 {
		t.Fatal("multimodal inference model omitted")
	}
	protocols := projection.Coverage.Models[index].Protocols
	required := e4bValidationCells()
	checkNames := func(values []modalityProtocolProjection) bool {
		if len(values) != len(required) {
			return false
		}
		seen := map[string]bool{}
		for _, value := range values {
			if seen[value.Name] || !slices.Contains(required, value.Name) {
				return false
			}
			seen[value.Name] = true
		}
		return true
	}
	if !checkNames(protocols) || checkNames(protocols[1:]) {
		t.Fatal("modality protocol denominator differs")
	}
	for _, value := range protocols {
		if value.Required == nil || value.Recorded == nil || *value.Required != *value.Recorded {
			t.Fatalf("case denominator unbound: %s", value.Name)
		}
		if strings.HasPrefix(value.Name, "resources/") && value.RepairOwner != "device-memory-retention/do" {
			t.Fatal("current resource gap lost its owner")
		}
	}
	for name, mutate := range map[string]func(*modalityProtocolProjection){
		"missing-mode":        func(p *modalityProtocolProjection) { p.InputMode = "" },
		"missing-source":      func(p *modalityProtocolProjection) { p.Source = "foreign" },
		"missing-denominator": func(p *modalityProtocolProjection) { p.Required = nil },
		"missing-repair":      func(p *modalityProtocolProjection) { p.RepairOwner = "" },
		"contradictory-count": func(p *modalityProtocolProjection) { n := *p.Recorded + 1; p.Recorded = &n },
	} {
		t.Run(name, func(t *testing.T) {
			changed := protocols[0]
			mutate(&changed)
			if checkModalityProtocols([]modalityProtocolProjection{changed}) == nil {
				t.Fatal("invalid protocol received case credit")
			}
		})
	}
	t.Log("Current catalog and original protocol scopes reconciled; required examples, recorded examples, generation cases and recovery actions remain separate. No model acquisitions.")
}

func TestProtocolCaseProjection(t *testing.T) {
	declared := []json.RawMessage{json.RawMessage(`{"name":"first"}`), json.RawMessage(`{"name":"second"}`)}
	for name, observed := range map[string][]json.RawMessage{
		"complete": {declared[1], declared[0]}, "omitted": {declared[0]}, "duplicate": {declared[0], declared[0]}, "foreign": {declared[0], json.RawMessage(`{"name":"third"}`)},
	} {
		t.Run(name, func(t *testing.T) {
			if (checkProtocolCaseNames(declared, observed) == nil) != (name == "complete") {
				t.Fatal("case membership differs")
			}
		})
	}
}

func TestResourceCaseProjection(t *testing.T) {
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "resource-recipe")
	observation := testutil.ArtifactID(t, artifact.KindEvidence, "resource-observation")
	gate := runrecord.GateResult{CodeCommit: "source", Recipe: recipeID, Outcome: runrecord.OutcomeSucceeded, Steps: []runrecord.GateStep{
		{Name: "media-resource-contract", Outcome: runrecord.StepSucceeded, Evidence: "reloads=1;cycles=1;actions=1"},
		{Name: "media-r0", Outcome: runrecord.StepSucceeded, Evidence: fmt.Sprintf(`{"Mode":"mixed","Observation":%q}`, observation)},
	}}
	for name, mutate := range map[string]func(*runrecord.GateResult){
		"complete":        func(*runrecord.GateResult) {},
		"omitted":         func(g *runrecord.GateResult) { g.Steps = g.Steps[:1] },
		"duplicate":       func(g *runrecord.GateResult) { g.Steps = append(g.Steps, g.Steps[1]) },
		"failed":          func(g *runrecord.GateResult) { g.Steps[1].Outcome = runrecord.StepFailed },
		"failed-contract": func(g *runrecord.GateResult) { g.Steps[0].Outcome = runrecord.StepFailed },
		"zero":            func(g *runrecord.GateResult) { g.Steps[0].Evidence = "reloads=0;cycles=1;actions=1" },
		"overflow":        func(g *runrecord.GateResult) { g.Steps[0].Evidence = "reloads=18446744073709551615;cycles=2;actions=1" },
		"duplicate-axis":  func(g *runrecord.GateResult) { g.Steps[0].Evidence += ";reloads=1" },
		"changed-source":  func(g *runrecord.GateResult) { g.CodeCommit = "other" },
	} {
		t.Run(name, func(t *testing.T) {
			changed := gate
			changed.Steps = slices.Clone(gate.Steps)
			mutate(&changed)
			value := modalityProtocolProjection{Source: gate.CodeCommit, Recipe: recipeID, InputMode: "mixed-history"}
			if (projectResourceCaseCounts(&value, changed) == nil) != (name == "complete") {
				t.Fatal("resource denominator verdict differs")
			}
		})
	}
}

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
		declared, _, err := modelartifact.EvalDomains(ctx, store, entry.Model)
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
