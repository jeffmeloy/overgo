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
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
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
	testcheck.NoError(t, first.add(cell))
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
	testcheck.NoError(t, other.add(cell))
	if openCellCredit(path, "commit-a", "store-a").has(cell) {
		t.Error("the credit of the earlier commit survived a run of the next")
	}

	store := overgodbtest.Open(t, t.TempDir())
	directory, kept := t.TempDir(), filepath.Join(t.TempDir(), "cycle_credit.json")
	weights, beside := filepath.Join(directory, "model.bin"), filepath.Join(directory, "processor", "config.json")
	write := func(path, text string) {
		t.Helper()
		testcheck.NoError(t, errors.Join(os.MkdirAll(filepath.Dir(path), 0700), os.WriteFile(path, []byte(text), 0600)))
	}
	replace := func(path, text string) {
		t.Helper()
		info, err := os.Stat(path)
		testcheck.NoError(t, err)
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
		testcheck.NoError(t, bindCells(t.Context(), testutil.RepoRoot(t), store, cells, execution))
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
			testcheck.NoError(t, acquired.add(c))
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
		testcheck.NoError(t, os.Chtimes(path, time.Now(), time.Now()))
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
	testcheck.NoError(t, os.Rename(beside, beside+".moved"))
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
	testcheck.NoError(t, os.Remove(filepath.Join(directory, "unlisted.bin")))
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
		testcheck.NoError(t, os.Remove(link))
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
		testcheck.NoError(t, bindCells(t.Context(), testutil.RepoRoot(t), store, cells, execution))
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
	storeRoot := testcheck.Must(filepath.Abs(filepath.Join(t.TempDir(), "overgodb-store")))(t)
	kept, weights := filepath.Join(t.TempDir(), "cycle_credit.json"), filepath.Join(t.TempDir(), "model.gguf")
	t.Setenv(clioptions.CycleCreditEnvironment, kept)
	store := overgodbtest.Create(t, storeRoot)
	testcheck.NoError(t, os.WriteFile(weights, []byte("weights"), 0o600))
	model := creditModel(t, store, weights)
	bound := func() Plan {
		t.Helper()
		cells := []ModelValidation{{
			Model: model, ModelName: "kept", Kind: recipe.TaskInference, Validation: guardValidation,
			Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "kept"), Location: weights,
		}}
		testcheck.NoError(t, bindCells(t.Context(), testutil.RepoRoot(t), store, cells, "fixture-execution"))
		return Plan{Run: []Selected{{ModelValidation: cells[0]}}}
	}
	testcheck.NoError(t, openCellCredit(kept, head, storeRoot).add(bound().Run[0].ModelValidation))
	var out bytes.Buffer
	if err := executeRun(t.Context(), &out, root, storeRoot, nil, bound(), scopeCells); err != nil || !strings.Contains(out.String(), "kept credited") {
		t.Fatalf("a cell whose weights are as they were was started again: %v\n%s", err, out.String())
	}
	before := testcheck.Must(os.Stat(weights))(t)
	testcheck.NoError(t, os.WriteFile(weights, []byte("changed"), 0o600))
	testcheck.NoError(t, os.Chtimes(weights, before.ModTime(), before.ModTime()))
	out.Reset()
	if err := executeRun(t.Context(), &out, root, storeRoot, nil, bound(), scopeCells); err == nil || !strings.Contains(err.Error(), "1 failed") || strings.Contains(out.String(), "credited") {
		t.Fatalf("a cell whose weights changed was not started again: %v\n%s", err, out.String())
	}
}

// creditModel declares the exact files two recovery fixtures consume.
// creditView registers a model as a native directory view is: one image no file
// holds, located at the directory it reads from.
func creditView(t *testing.T, store *overgodb.Store, directory string) artifact.ID {
	t.Helper()
	image := testutil.ArtifactID(t, artifact.KindTensorSet, "view image")
	manifest := testcheck.Must(artifact.NewManifest(artifact.KindModel, []artifact.Component{{Role: artifact.ComponentWeights, Name: "weights", Artifact: image}}))(t)
	location := testcheck.Must(artifact.CanonicalLocalLocation(image, artifact.LocationDirectory, directory))(t)
	if _, err := artifact.Publish(t.Context(), store, artifact.Batch{
		Key: "test/credit/view/" + manifest.ID.String(), Artifacts: []artifact.Descriptor{{ID: image, Size: 1}}, Manifests: []artifact.Manifest{manifest},
		Locations: []artifact.LocationEvent{{Location: location, Action: artifact.LocationAdd}},
	}); err != nil {
		t.Fatal(err)
	}
	return manifest.ID
}

// TestCellCreditCoversADirectoryView holds resume credit for a model registered as
// a native view: its image is no file, so the credit rests on the directory's
// files. Unchanged files earn the same credit; a changed or an added one does not,
// and a view whose directory is gone earns none.
func TestCellCreditCoversADirectoryView(t *testing.T) {
	t.Parallel()
	store := overgodbtest.Open(t, t.TempDir())
	directory := t.TempDir()
	weights, config := filepath.Join(directory, "model.safetensors"), filepath.Join(directory, "config.json")
	write := func(path, text string) {
		testutil.WriteTextFile(t, filepath.Dir(path), filepath.Base(path), text, 0o600)
	}
	write(weights, "weights")
	write(config, "{}")
	model := creditView(t, store, directory)
	cell := ModelValidation{Model: model, Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "view"), Kind: recipe.TaskImageGen, Validation: "image-proof", Location: directory, Directory: directory}
	consumed := func() string {
		t.Helper()
		return consumedFiles(t.Context(), store, cell, map[string]string{}, "fixture-execution")
	}
	first := consumed()
	if first == "" || first != consumed() {
		t.Fatalf("an unchanged view earned %q, then %q", first, consumed())
	}
	write(weights, "changed")
	changed := consumed()
	if changed == first {
		t.Fatal("changed weights kept the view's credit")
	}
	write(weights, "weights")
	if consumed() != first {
		t.Fatal("restored weights did not earn the view's credit back")
	}
	write(filepath.Join(directory, "added.bin"), "new")
	if consumed() == first {
		t.Fatal("an added file kept the view's credit")
	}
	testcheck.NoError(t, os.RemoveAll(directory))
	if consumed() == consumed() {
		t.Fatal("a view whose directory is gone earned a stable credit")
	}
}

func creditModel(t *testing.T, store *overgodb.Store, paths ...string) artifact.ID {
	t.Helper()
	var batch artifact.Batch
	var components []artifact.Component
	for index, path := range paths {
		id := testcheck.Must(artifact.IdentifyFile(artifact.KindFile, path))(t)
		info := testcheck.Must(os.Stat(path))(t)
		location := testcheck.Must(artifact.CanonicalLocalLocation(id, artifact.LocationFile, path))(t)
		components = append(components, artifact.Component{Role: artifact.ComponentConfig, Ordinal: uint32(index), Name: filepath.Base(path), Artifact: id})
		batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: id, Size: uint64(info.Size())})
		batch.Locations = append(batch.Locations, artifact.LocationEvent{Location: location, Action: artifact.LocationAdd})
	}
	manifest := testcheck.Must(artifact.NewManifest(artifact.KindModel, components))(t)
	batch.Manifests = []artifact.Manifest{manifest}
	batch.Key = "test/credit/" + manifest.ID.String()
	if _, err := artifact.Publish(t.Context(), store, batch); err != nil {
		t.Fatal(err)
	}
	return manifest.ID
}
