package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// TestCellCreditSurvivesAKilledRun checks persisted credit against declared content and execution.
//   - The same commit, store, model, recipe and inputs retain completed work
//   - Unknown authority and changed bytes cannot inherit completed work
//   - Metadata changes alone preserve identical content
//   - File-link coverage is reported unavailable when the host refuses its fixture
func TestCellCreditSurvivesAKilledRun(t *testing.T) {
	t.Parallel()
	path := filepath.Join(t.TempDir(), "cycle_credit.json")
	model, recipeID := testutil.ArtifactID(t, artifact.KindModel, "credit"), testutil.ArtifactID(t, artifact.KindRecipe, "credit")
	cell := ModelValidation{Model: model, Kind: recipe.TaskTraining, Validation: trainingProof, Recipe: recipeID, Directory: "d:/models/a", Grounding: "dataset:1"}

	first := openCellCredit(path, "commit-a", "store-a")
	if first.has(cell) {
		t.Fatal("a cell is credited before any run acquired it")
	}
	if err := first.add(cell); err != nil {
		t.Fatal(err)
	}
	// The next process reads the same file.
	second := openCellCredit(path, "commit-a", "store-a")
	if !second.has(cell) {
		t.Fatal("a cell a killed run acquired is not credited to the run that continues it")
	}
	measured := cell
	measured.Cost = 9e9
	if !second.has(measured) {
		t.Error("what a run changes of a cell (its cost) made it another cell")
	}

	for name, change := range map[string]func(*ModelValidation){
		"recipe":       func(c *ModelValidation) { c.Recipe = testutil.ArtifactID(t, artifact.KindRecipe, "other") },
		"protocol":     func(c *ModelValidation) { c.Protocol = "chat-template" },
		"directory":    func(c *ModelValidation) { c.Directory = "d:/models/b" },
		"grounding":    func(c *ModelValidation) { c.Grounding = "dataset:2" },
		"objective":    func(c *ModelValidation) { c.Objective = "objective.registered.other" },
		"conditioning": func(c *ModelValidation) { c.Conditioning = "d:/video-stack" },
		"projector":    func(c *ModelValidation) { c.Projector = "d:/mmproj.gguf" },
	} {
		other := cell
		change(&other)
		if second.has(other) {
			t.Errorf("another %s retained credit", name)
		}
	}

	for name, credit := range map[string]*cellCredit{
		"another commit": openCellCredit(path, "commit-b", "store-a"),
		"another store":  openCellCredit(path, "commit-a", "store-b"),
		"no file":        openCellCredit("", "commit-a", "store-a"),
	} {
		if credit.has(cell) {
			t.Errorf("%s credited a cell it did not acquire", name)
		}
	}
	// A run of another commit keeps its own credit, replacing the old.
	other := openCellCredit(path, "commit-b", "store-a")
	if err := other.add(cell); err != nil {
		t.Fatal(err)
	}
	if openCellCredit(path, "commit-a", "store-a").has(cell) {
		t.Error("the credit of the earlier commit survived a run of the next")
	}

	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	directory, kept := t.TempDir(), filepath.Join(t.TempDir(), "cycle_credit.json")
	weights, beside := filepath.Join(directory, "model.bin"), filepath.Join(directory, "processor", "config.json")
	write := func(path, text string) {
		t.Helper()
		if err := errors.Join(os.MkdirAll(filepath.Dir(path), 0700), os.WriteFile(path, []byte(text), 0600)); err != nil {
			t.Fatal(err)
		}
	}
	replace := func(path, text string) {
		t.Helper()
		info, err := os.Stat(path)
		if err != nil {
			t.Fatal(err)
		}
		write(path, text)
		if err = os.Chtimes(path, info.ModTime(), info.ModTime()); err != nil {
			t.Fatal(err)
		}
	}
	write(weights, "weights")
	write(beside, "{}")
	model = creditModel(t, store, weights)
	directoryModel := creditModel(t, store, weights, beside)
	execution := "fixture-execution"
	bound := func(cell ModelValidation) ModelValidation {
		t.Helper()
		cells := []ModelValidation{cell}
		if err := bindCells(t.Context(), testutil.RepoRoot(t), store, cells, execution); err != nil {
			t.Fatal(err)
		}
		return cells[0]
	}
	serving := func() ModelValidation {
		return bound(ModelValidation{Model: model, Recipe: recipeID, Kind: recipe.TaskInference, Validation: servingValidation, Location: weights})
	}
	proof := func() ModelValidation {
		return bound(ModelValidation{Model: directoryModel, Recipe: recipeID, Kind: recipe.TaskImageGen, Validation: "image-proof", Location: weights, Directory: directory})
	}
	smoke := func() ModelValidation {
		c := serving()
		c.Modality = dnaModality
		c.Validation = guardValidation
		return bound(c)
	}
	acquired := openCellCredit(kept, "commit-a", "store-a")
	acquire := func(cells ...ModelValidation) {
		t.Helper()
		for _, c := range cells {
			if err := acquired.add(c); err != nil {
				t.Fatal(err)
			}
		}
	}
	resumed := func(c ModelValidation) bool { return openCellCredit(kept, "commit-a", "store-a").has(c) }
	acquire(serving(), proof(), smoke())
	if !resumed(serving()) || !resumed(proof()) || !resumed(smoke()) {
		t.Fatal("unchanged declared files lost credit")
	}
	data, err := json.Marshal(proof())
	if err != nil || !bytes.Contains(data, []byte("\"consumed\":")) {
		t.Fatalf("phase input identity missing: %s %v", data, err)
	}
	if got := bound(ModelValidation{Kind: recipe.TaskImageGen, Validation: "image-proof", Location: weights}).Consumed; got != "" {
		t.Fatalf("unbound command read inputs: %s", got)
	}
	replace(beside, "{\"do_resize\":true}")
	if resumed(proof()) || !resumed(serving()) || !resumed(smoke()) {
		t.Fatal("a companion change did not respect the declared input boundary")
	}
	directoryModel = creditModel(t, store, weights, beside)
	acquire(proof())
	for _, path := range []string{weights, filepath.Dir(beside)} {
		if err := os.Chtimes(path, time.Now(), time.Now()); err != nil {
			t.Fatal(err)
		}
	}
	if !resumed(serving()) || !resumed(proof()) {
		t.Fatal("metadata alone invalidated identical content")
	}
	replace(weights, "changed")
	if resumed(serving()) || resumed(smoke()) || resumed(proof()) {
		t.Fatal("same-size same-time replacement retained credit")
	}
	model = creditModel(t, store, weights)
	directoryModel = creditModel(t, store, weights, beside)
	acquire(serving(), proof(), smoke())
	if !resumed(serving()) || !resumed(proof()) {
		t.Fatal("new declared model content could not earn credit")
	}
	for _, identity := range []string{"other-execution", ""} {
		execution = identity
		acquire(serving())
		if identity == "" && resumed(serving()) {
			t.Fatal("unknown execution retained credit")
		}
		if identity != "" && resumed(proof()) {
			t.Fatal("changed execution inherited credit")
		}
	}
	execution = "fixture-execution"
	if err := os.Rename(beside, beside+".moved"); err != nil {
		t.Fatal(err)
	}
	if resumed(proof()) {
		t.Fatal("missing declared companion retained credit")
	}
	beside += ".moved"
	directoryModel = creditModel(t, store, weights, beside)
	acquire(proof())
	write(filepath.Join(directory, "unlisted.bin"), "unknown")
	acquire(proof())
	if resumed(proof()) {
		t.Fatal("undeclared directory content retained credit")
	}
	if err := os.Remove(filepath.Join(directory, "unlisted.bin")); err != nil {
		t.Fatal(err)
	}
	linked := filepath.Join(t.TempDir(), "tokenizer.json")
	write(linked, "{}")
	link := filepath.Join(directory, "tokenizer.json")
	if err := os.Symlink(linked, link); err != nil {
		t.Logf("file-link coverage unavailable: %v", err)
	} else {
		directoryModel = creditModel(t, store, weights, beside, link)
		acquire(proof())
		if !resumed(proof()) {
			t.Fatal("declared file link could not earn credit")
		}
		replace(linked, "[]")
		if resumed(proof()) {
			t.Fatal("changed file-link target retained credit")
		}
		if err := os.Remove(link); err != nil {
			t.Fatal(err)
		}
	}
	directoryModel = creditModel(t, store, weights, beside)
	acquire(proof())
	link = filepath.Join(directory, "linked-directory")
	target := t.TempDir()
	linkErr := os.Symlink(target, link)
	if linkErr != nil && runtime.GOOS == "windows" {
		output, err := exec.Command("cmd", "/c", "mklink", "/J", link, target).CombinedOutput()
		linkErr = err
		if err != nil {
			t.Logf("junction fixture: %s", output)
		}
	}
	if linkErr != nil {
		t.Logf("directory-link coverage unavailable: %v", linkErr)
	} else {
		acquire(proof())
		if resumed(proof()) {
			t.Fatal("unresolved directory link retained credit")
		}
		write(filepath.Join(target, "new.bin"), "changed")
		if resumed(proof()) {
			t.Fatal("changed directory-link target retained credit")
		}
	}
	for _, location := range []string{filepath.Join(directory, "absent.gguf"), "https://provider.example/v1/models/remote"} {
		c := serving()
		c.Location = location
		acquire(bound(c))
		if resumed(bound(c)) {
			t.Fatalf("unknown input retained credit: %s", location)
		}
	}
	apart := filepath.Join(t.TempDir(), "config.json")
	write(apart, "{}")
	apartModel := creditModel(t, store, apart)
	together := func() []ModelValidation {
		t.Helper()
		cells := []ModelValidation{{Model: apartModel, Recipe: recipeID, Kind: recipe.TaskImageGen, Validation: "image-proof", Location: apart, Directory: filepath.Dir(apart)}, {Model: model, Recipe: recipeID, Kind: recipe.TaskInference, Validation: servingValidation, Location: weights}}
		if err := bindCells(t.Context(), testutil.RepoRoot(t), store, cells, execution); err != nil {
			t.Fatal(err)
		}
		return cells
	}
	acquire(together()...)
	replace(weights, "another")
	again := together()
	if !resumed(again[0]) || resumed(again[1]) {
		t.Fatal("one changed input invalidated the wrong binding")
	}

}

// TestResumedRunStartsAgainACellWhoseWeightsChanged holds the run that continues a cycle to
// what it starts: a guard cell an earlier run of the commit and store acquired is not started
// while its weights file is as it was, and is started again once the file has changed. The
// fixture repository holds no Go module, so the started command ends at once, failed, which
// is how the test sees that it was started.
func TestResumedRunStartsAgainACellWhoseWeightsChanged(t *testing.T) {
	// Serial: it names the cycle's credit file in the process environment.
	root, head := fixtureRepository(t)
	storeRoot, err := filepath.Abs(filepath.Join(t.TempDir(), "overgodb-store"))
	if err != nil {
		t.Fatal(err)
	}
	kept, weights := filepath.Join(t.TempDir(), "cycle_credit.json"), filepath.Join(t.TempDir(), "model.gguf")
	t.Setenv(clioptions.CycleCreditEnvironment, kept)
	store, err := overgodb.Open(storeRoot)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if err := os.WriteFile(weights, []byte("weights"), 0o600); err != nil {
		t.Fatal(err)
	}
	model := creditModel(t, store, weights)
	bound := func() Plan {
		t.Helper()
		cells := []ModelValidation{{
			Model: model, ModelName: "kept", Kind: recipe.TaskInference, Validation: guardValidation,
			Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "kept"), Location: weights,
		}}
		if err := bindCells(t.Context(), testutil.RepoRoot(t), store, cells, "fixture-execution"); err != nil {
			t.Fatal(err)
		}
		return Plan{Run: []Selected{{ModelValidation: cells[0]}}}
	}
	if err := openCellCredit(kept, head, storeRoot).add(bound().Run[0].ModelValidation); err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if err := executeRun(t.Context(), &out, root, storeRoot, nil, bound(), scopeCells); err != nil || !strings.Contains(out.String(), "kept credited") {
		t.Fatalf("a cell whose weights are as they were was started again: %v\n%s", err, out.String())
	}
	before, err := os.Stat(weights)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(weights, []byte("changed"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(weights, before.ModTime(), before.ModTime()); err != nil {
		t.Fatal(err)
	}
	out.Reset()
	if err := executeRun(t.Context(), &out, root, storeRoot, nil, bound(), scopeCells); err == nil || !strings.Contains(err.Error(), "1 failed") || strings.Contains(out.String(), "credited") {
		t.Fatalf("a cell whose weights changed was not started again: %v\n%s", err, out.String())
	}
}

// creditModel declares the exact files two recovery fixtures consume.
func creditModel(t *testing.T, store *overgodb.Store, paths ...string) artifact.ID {
	t.Helper()
	var batch artifact.Batch
	var components []artifact.Component
	for index, path := range paths {
		id, err := artifact.IdentifyFile(artifact.KindFile, path)
		if err != nil {
			t.Fatal(err)
		}
		info, err := os.Stat(path)
		if err != nil {
			t.Fatal(err)
		}
		location, err := artifact.CanonicalLocalLocation(id, artifact.LocationFile, path)
		if err != nil {
			t.Fatal(err)
		}
		components = append(components, artifact.Component{Role: artifact.ComponentConfig, Ordinal: uint32(index), Name: filepath.Base(path), Artifact: id})
		batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: id, Size: uint64(info.Size())})
		batch.Locations = append(batch.Locations, artifact.LocationEvent{Location: location, Action: artifact.LocationAdd})
	}
	manifest, err := artifact.NewManifest(artifact.KindModel, components)
	if err != nil {
		t.Fatal(err)
	}
	batch.Manifests = []artifact.Manifest{manifest}
	batch.Key = "test/credit/" + manifest.ID.String()
	if _, err := artifact.Publish(t.Context(), store, batch); err != nil {
		t.Fatal(err)
	}
	return manifest.ID
}
