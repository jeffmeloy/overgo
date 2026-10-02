package main

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/hfrepo"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/safetensors"
	"overgo/internal/testutil"
)

// TestReacquireCommand proves the bound guard command is fully determined by the
// cell and the selected store, and that an unbound kind reports rather than
// fabricates a command. The guard publishes into the store the command read,
// never the default its own working directory would resolve.
func TestReacquireCommand(t *testing.T) {
	t.Parallel()
	guard := ModelValidation{Kind: recipe.TaskInference, Validation: "guard", Location: "C:/models/qwen.gguf"}
	name, args, err := reacquireCommand(guard, "C:/stores/selected")
	if err != nil || name != "go" {
		t.Fatalf("guard command = (%q, %v)", name, err)
	}
	if !slices.Contains(args, "-guard") || !slices.Contains(args, "-publish") || args[len(args)-1] != "C:/models/qwen.gguf" {
		t.Fatalf("guard args = %v", args)
	}
	if at := slices.Index(args, "-repo"); at < 0 || args[at+1] != "C:/stores/selected" {
		t.Fatalf("guard args = %v, want -repo C:/stores/selected", args)
	}

	if _, _, err := reacquireCommand(ModelValidation{Kind: recipe.TaskInference, Validation: "guard"}, "C:/stores/selected"); err == nil {
		t.Fatal("guard without a location bound a command")
	}
	if _, _, err := reacquireCommand(ModelValidation{Kind: recipe.TaskImageGen, Validation: "image-proof"}, "C:/stores/selected"); err == nil {
		t.Fatal("an image proof without a registered directory bound a command")
	}
}

// TestDNAGuardBindsTheSmokeLane holds a DNA model's stale guard cell to its
// owner. Its smoke run predates an inference surface move, so the
// current-evidence decision refuses it and the cell runs; -run then binds
// validate -smoke on that model into the selected store, the very command the
// refusal names, never the text guard and never no command at all.
func TestDNAGuardBindsTheSmokeLane(t *testing.T) {
	t.Parallel()
	root, beforeSurface, _, _ := surfaceHistory(t)
	const store = "C:/stores/selected"
	carbon := testutil.ArtifactID(t, artifact.KindModel, "carbon")
	cell := ModelValidation{
		Model: carbon, ModelName: "Carbon", Location: "C:/models/carbon.gguf",
		Kind: recipe.TaskInference, Modality: dnaModality, Validation: guardValidation, Surface: "inference",
		Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "carbon-inference"),
	}
	stale := runrecord.Run{Outcome: runrecord.OutcomeSucceeded, CodeCommit: beforeSurface}
	read := acceptedRecords{smoke: map[artifact.ID]error{carbon: smokeCurrent(t.Context(), root, store, carbon, stale, true)}}
	plan := SelectValidation(Inputs{AffectedSurfaces: map[SurfaceID]AffectRecord{"inference": {}}, Registered: []ModelValidation{cell}, Current: read.current})
	if len(plan.Run) != 1 || plan.Run[0].Model != carbon || !strings.Contains(plan.Run[0].Reason, "inference surface moved") {
		t.Fatalf("a stale DNA smoke run was not selected to run: run=%+v reuse=%+v", plan.Run, plan.Reuse)
	}

	name, args, err := reacquireCommand(plan.Run[0].ModelValidation, store)
	want := []string{"run", "./cmd/validate", "-smoke", "-budget", "15m", "-model", carbon.String(), "-repo", store}
	if err != nil || name != "go" || !slices.Equal(args, want) {
		t.Fatalf("a stale DNA cell bound (%q, %v, %v), want go %v", name, args, err, want)
	}
	if bound := "run: go " + strings.Join(want, " "); !strings.HasSuffix(plan.Run[0].Reason, bound) {
		t.Fatalf("the refusal %q does not name the bound command %q", plan.Run[0].Reason, bound)
	}
}

// TestRunBindsMediaMeasurement holds each media and specialized validation
// to one measurement of the model through its active capability, into the
// selected store, so the loop's empty-plan run fills those models' scorecard
// rows in the store it read. Each cell is built as the cycle builds it, from
// a store holding the model as its registration recorded it, where the
// catalog locates the model by one weights file: a shard, the root weights,
// or a head's weights beneath the directory. The capability resolves the
// directory its registration inventoried, or, for a composite inventoried
// at a parent it shares, its own directory recorded inside that parent, so
// the command's path must re-inventory to the registered model the
// activation names and, for a generation model, select its registered
// source profile from the configuration beside its weights. An inference
// model's serving cell measures the weights file it serves.
func TestRunBindsMediaMeasurement(t *testing.T) {
	t.Parallel()
	ctx, models := t.Context(), t.TempDir()
	const selected = "C:/stores/selected"
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	write := func(path, text string) {
		t.Helper()
		err := os.MkdirAll(filepath.Dir(path), 0o755)
		switch {
		case err != nil:
		case strings.HasSuffix(path, ".safetensors"):
			err = safetensors.Save(path, map[string][]float32{text: {1}}, map[string][]int{text: {1}}, nil)
		default:
			err = os.WriteFile(path, []byte(text), 0o600)
		}
		if err != nil {
			t.Fatal(err)
		}
	}
	// The inventories the capabilities resolve: a Hugging Face repository,
	// a root configuration beside one weights file, one subdirectory per
	// tabular head, and an edit composite rooted at the parent it shares
	// with the base model beside it, which records its own directory there.
	composite := func(path string) (modelartifact.Inventory, error) {
		base := filepath.Join(filepath.Dir(path), "Wan2.1-T2V-1.3B")
		inventory, err := modelartifact.FromFiles(filepath.Dir(path), []modelartifact.FileSpec{
			{Path: filepath.Join(path, "model.safetensors"), Name: "liveedit/weights", Role: artifact.ComponentWeights},
			{Path: filepath.Join(base, "config.json"), Name: "wan/config", Role: artifact.ComponentConfig},
			{Path: filepath.Join(base, "model.safetensors"), Name: "wan/weights", Role: artifact.ComponentWeights},
		})
		if err != nil {
			return inventory, err
		}
		own, err := artifact.CanonicalLocalLocation(inventory.Manifest.ID, artifact.LocationDirectory, path)
		inventory.Locations = append(inventory.Locations, own)
		return inventory, err
	}
	rootWeights := func(root string) (modelartifact.Inventory, error) {
		return modelartifact.FromFiles(root, []modelartifact.FileSpec{
			{Path: filepath.Join(root, "config.json"), Name: "config", Role: artifact.ComponentConfig},
			{Path: filepath.Join(root, "model.safetensors"), Name: "weights", Role: artifact.ComponentWeights},
		})
	}
	heads := func(root string) (modelartifact.Inventory, error) {
		var specs []modelartifact.FileSpec
		for _, head := range []string{"classification", "regression"} {
			specs = append(specs,
				modelartifact.FileSpec{Path: filepath.Join(root, head, "config.json"), Name: head + "/config", Role: artifact.ComponentConfig},
				modelartifact.FileSpec{Path: filepath.Join(root, head, "model.safetensors"), Name: head + "/weights", Role: artifact.ComponentWeights})
		}
		return modelartifact.FromFiles(root, specs)
	}
	register := func(name string, inventory func(string) (modelartifact.Inventory, error), files map[string]string) artifact.ID {
		t.Helper()
		root := filepath.Join(models, name)
		for path, text := range files {
			write(filepath.Join(root, filepath.FromSlash(path)), text)
		}
		resolved, err := inventory(root)
		if err != nil {
			t.Fatal(err)
		}
		batch, err := resolved.Batch("fixture/model/" + name)
		if err == nil {
			err = modelrecipe.PublishFacts(ctx, store, batch)
		}
		if err != nil {
			t.Fatal(err)
		}
		return resolved.Manifest.ID
	}
	activate := func(definition recipe.Definition, err error) {
		t.Helper()
		if err == nil {
			err = modelrecipetest.PublishActivation(ctx, store, "fixture/activation/"+definition.ID.String(), definition)
		}
		if err != nil {
			t.Fatal(err)
		}
	}
	sharded := register("SenseNova", modelartifact.FromHFPath, map[string]string{
		"config.json":                      `{"model_type":"neo_chat","architectures":["NEOChatModel"]}`,
		"model-00001-of-00002.safetensors": "first",
		"model-00002-of-00002.safetensors": "second",
		"model.safetensors.index.json":     `{"metadata":{},"weight_map":{"first":"model-00001-of-00002.safetensors","second":"model-00002-of-00002.safetensors"}}`,
	})
	single := register("Un-0", rootWeights, map[string]string{"config.json": `{"model_type":"un0"}`, "model.safetensors": "weights"})
	headed := register("tabfm", heads, map[string]string{
		"config.json": `{}`, "classification/config.json": `{}`, "classification/model.safetensors": "classification",
		"regression/config.json": `{}`, "regression/model.safetensors": "regression",
	})
	repository := register("needle", modelartifact.FromHFPath, map[string]string{"config.json": `{"model_type":"t5"}`, "model.safetensors": "needle"})
	edit := register("LiveEdit", composite, map[string]string{
		"model.safetensors": "edit", "../Wan2.1-T2V-1.3B/config.json": `{}`, "../Wan2.1-T2V-1.3B/model.safetensors": "base",
	})
	flow := testutil.ArtifactID(t, artifact.KindProfile, "flow profile")
	testutil.PublishArtifact(t, store, flow)
	activate(modelrecipe.ReferenceVideoEditDefinition(edit, flow))
	activate(modelrecipe.GenerationDefinition(modelrecipe.ModuleRoutedImagePrepare, sharded, flow))
	activate(modelrecipe.GenerationDefinition(modelrecipe.ModuleOscillatorImagePrepare, single, artifact.ID{}))
	activate(modelrecipe.GenerationDefinition(modelrecipe.ModuleOscillatorVideoPrepare, single, artifact.ID{}))
	activate(modelrecipe.CapabilityDefinition(recipe.TaskTabular, headed))
	activate(modelrecipe.CapabilityDefinition(recipe.TaskSeq2Seq, repository))

	cells, err := registeredCells(ctx, store)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []struct {
		model      artifact.ID
		task       recipe.Task
		validation string
		inventory  func(string) (modelartifact.Inventory, error)
		prepare    recipe.ModuleID
	}{
		{sharded, recipe.TaskImageGen, "image-proof", modelartifact.FromHFPath, modelrecipe.ModuleRoutedImagePrepare},
		{single, recipe.TaskImageGen, "image-proof", rootWeights, modelrecipe.ModuleOscillatorImagePrepare},
		{single, recipe.TaskVideoGen, "video-proof", rootWeights, modelrecipe.ModuleOscillatorVideoPrepare},
		{headed, recipe.TaskTabular, "specialized-proof", heads, ""},
		{repository, recipe.TaskSeq2Seq, "specialized-proof", modelartifact.FromHFPath, ""},
		{edit, recipe.TaskVideoGen, "video-proof", composite, ""},
	} {
		at := slices.IndexFunc(cells, func(cell ModelValidation) bool { return cell.Model == want.model && cell.Kind == want.task })
		if at < 0 || cells[at].Validation != want.validation {
			t.Fatalf("no %s cell for the %s activation of %s: %+v", want.validation, want.task, want.model, cells)
		}
		cell := cells[at]
		if info, err := os.Stat(cell.Location); err != nil || info.IsDir() {
			t.Fatalf("the catalog locates %s at %q, not by one of its weights files (%v)", want.model, cell.Location, err)
		}
		name, args, err := reacquireCommand(cell, selected)
		prefix := []string{"run", "./cmd/recipe", "measure", "-repo", selected, "-task", string(want.task)}
		if err != nil || name != "go" || len(args) != len(prefix)+1 || !slices.Equal(args[:len(prefix)], prefix) {
			t.Fatalf("%s %s bound (%q, %v, %v), want go %v <model>", want.task, want.validation, name, args, err, prefix)
		}
		path := args[len(args)-1]
		if resolved, err := want.inventory(path); err != nil || resolved.Manifest.ID != want.model {
			t.Errorf("%s %s measures %s, which re-inventories to %s (%v), not the registered model %s located at %s",
				want.task, want.validation, path, resolved.Manifest.ID, err, want.model, cell.Location)
		}
		if want.prepare == "" {
			continue
		}
		identity, err := hfrepo.InspectIdentity(path)
		profile, profileErr := modelrecipe.ResolveGenerationSource(want.task, identity)
		if err != nil || profileErr != nil || profile.Prepare != want.prepare {
			t.Errorf("%s %s measures %s, which selects source profile %q (%v), want %q",
				want.task, want.validation, path, profile.Prepare, errors.Join(err, profileErr), want.prepare)
		}
	}

	serving := ModelValidation{Kind: recipe.TaskInference, Validation: servingValidation, Location: "C:/models/qwen.gguf"}
	name, args, err := reacquireCommand(serving, selected)
	if want := []string{"run", "./cmd/recipe", "measure", "-repo", selected, "-task", string(recipe.TaskInference), serving.Location}; err != nil || name != "go" || !slices.Equal(args, want) {
		t.Fatalf("serving measurement bound (%q, %v, %v), want go %v", name, args, err, want)
	}
}

// TestValidateRunEmpty proves an empty run set is the clean no-op the dry run
// predicts for a commit that moves no surface: no command executes.
func TestValidateRunEmpty(t *testing.T) {
	t.Parallel()
	var out bytes.Buffer
	plan := Plan{Reuse: []Selected{{}, {}}}
	if err := executeRun(t.Context(), &out, ".", "C:/stores/selected", plan, false); err != nil {
		t.Fatalf("empty run set errored: %v", err)
	}
	if !strings.Contains(out.String(), "nothing to re-acquire") {
		t.Fatalf("empty run output = %q", out.String())
	}
}

// TestValidateRunCycleLanes runs each selected cycle lane's go command with
// its environment and only those: a lane that passes is counted, one that
// fails fails the run by name, so the loop withholds promotion, and a reused
// lane never starts. A command other than go test answers by its exit; a go
// test lane is acquired only by the outcomes it requires, read from its own
// run's events: each package it names completes with a passing test its
// selector matches, a declared exclusion beside them visible but uncredited.
// A failed package, a zero-exit skip that a same-named test passing in
// another package does not excuse, and a selector matching nothing each
// withhold by name; a lane whose selected tests only declared themselves
// inapplicable acquires nothing and is reported apart from the failures.
func TestValidateRunCycleLanes(t *testing.T) {
	t.Parallel()
	const fixtures = "./internal/testevidence/testdata/"
	var out bytes.Buffer
	plan := Plan{Suites: []SelectedSuite{
		{Lane: "passes", Run: true, Arguments: []string{"env", "GOFLAGS"}, Environment: []string{"GOFLAGS=-tags=lane-environment"}},
		{Lane: "fails", Run: true, Arguments: []string{"vet", "./no-such-package"}},
		{Lane: "reused", Arguments: []string{"env", "GOFLAGS"}},
		{Lane: "tested", Run: true, Arguments: []string{"test", "-tags", "", fixtures + "probefixture"}},
		{Lane: "failed", Run: true, Arguments: []string{"test", "./no-such-package"}},
		{Lane: "skipped", Run: true, Arguments: []string{"test", fixtures + "probefixture", fixtures + "skipfixture", "-run", "^TestUnavailableCapability$"}},
		{Lane: "absent", Run: true, Arguments: []string{"test", fixtures + "probefixture", "-run", "^TestAbsent$"}},
		{Lane: "inapplicable", Run: true, Arguments: []string{"test", fixtures + "probefixture", "-run", "^TestElsewhere$"}},
	}}
	err := executeRun(t.Context(), &out, testutil.RepoRoot(t), "C:/stores/selected", plan, false)
	if err == nil {
		t.Fatalf("cycle lanes run passed; output:\n%s", out.String())
	}
	failed, inapplicable, _ := strings.Cut(err.Error(), "declared inapplicable here")
	for lane, want := range map[string]string{
		"fails":   "exit=",
		"failed":  "./no-such-package failed",
		"skipped": "testdata/skipfixture: TestUnavailableCapability skipped",
		"absent":  "testdata/probefixture is absent",
	} {
		if !strings.Contains(failed, "cycle lane "+lane+": ") || !strings.Contains(failed, want) {
			t.Errorf("cycle lane %s is not withheld as a failure saying %q: %v", lane, want, err)
		}
	}
	if !strings.Contains(inapplicable, "cycle lane inapplicable: ") || !strings.Contains(inapplicable, "TestElsewhere") || strings.Contains(failed, "cycle lane inapplicable") {
		t.Errorf("a lane whose required test declared itself inapplicable is not reported apart from the failures: %v", err)
	}
	for _, lane := range []string{"passes", "tested", "reused"} {
		if strings.Contains(err.Error(), "cycle lane "+lane+":") {
			t.Errorf("cycle lane %s was withheld: %v", lane, err)
		}
	}
	if !strings.Contains(err.Error(), "re-acquired 2 cell(s)") {
		t.Errorf("cycle lanes run = %v, want the passing and the tested lanes acquired", err)
	}
	if !strings.Contains(out.String(), "lane-environment") || strings.Contains(out.String(), "cycle lane reused") {
		t.Fatalf("cycle lane output = %q, want the passing lane run with its environment and the reused one not started", out.String())
	}
}

// TestValidateRunUnbound proves an unbound cell is reported as incomplete
// coverage without executing anything, never silently skipped.
func TestValidateRunUnbound(t *testing.T) {
	t.Parallel()
	var out bytes.Buffer
	model := testutil.ArtifactID(t, artifact.KindModel, "wan")
	plan := Plan{Run: []Selected{{ModelValidation: ModelValidation{Model: model, ModelName: "Wan", Kind: recipe.TaskVideoGen, Validation: "video-proof"}}}}
	err := executeRun(t.Context(), &out, ".", "C:/stores/selected", plan, false)
	if err == nil || !strings.Contains(err.Error(), "no bound command") {
		t.Fatalf("unbound run = %v", err)
	}
}
