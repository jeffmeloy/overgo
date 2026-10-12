package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// TestCellCreditSurvivesAKilledRun holds the credit a cycle keeps to what a killed run needs of
// it: a cell a run acquired is credited to the next run of the same commit and store, which is
// a new process reading the file; a cell is the same cell whatever a run changed of it (its
// cost) and another cell when a recipe, a protocol or any committed
// input it reads differs; another commit or store, or no file, credits nothing. A cell the
// continuing run binds again is the same cell while the files its command is given, and its
// weights, are as they were, and another once one of them changed in size, time or name; a
// file its command is not given costs it nothing; and weights nothing can read, an absent file
// or a remote model's address, are never credited. Only where the machine lets a test make a
// symbolic link does it also hold that a file reached through one is the file the link leads
// to. Windows refuses the link to an account without that privilege: there the test logs the
// refusal and holds nothing about links.
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
	for name, other := range map[string]ModelValidation{
		"recipe":       withRecipe(cell, testutil.ArtifactID(t, artifact.KindRecipe, "other")),
		"protocol":     withProtocol(cell, "chat-template"),
		"directory":    withDirectory(cell, "d:/models/b"),
		"grounding":    withGrounding(cell, "dataset:2"),
		"objective":    withObjective(cell, "objective.registered.other"),
		"conditioning": withConditioning(cell, "d:/video-stack"),
		"projector":    withProjector(cell, "d:/mmproj.gguf"),
	} {
		if second.has(other) {
			t.Errorf("a cell with another %s was credited by the one that was acquired", name)
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

	// What no manifest checks. Each process binds its own cells, so every question below is put
	// to a cell bound afresh, and the credit file is kept outside the directory a command is given.
	directory, kept := t.TempDir(), filepath.Join(t.TempDir(), "cycle_credit.json")
	weights, beside := filepath.Join(directory, "model.safetensors"), filepath.Join(directory, "processor", "preprocessor_config.json")
	write := func(path, text string) {
		t.Helper()
		if err := errors.Join(os.MkdirAll(filepath.Dir(path), 0o700), os.WriteFile(path, []byte(text), 0o600)); err != nil {
			t.Fatal(err)
		}
	}
	modified := func(path string) time.Time {
		t.Helper()
		info, err := os.Stat(path)
		if err != nil {
			t.Fatal(err)
		}
		return info.ModTime()
	}
	// stamp sets a file's time, which a write also moves, so each step below changes one fact of one file.
	stamp := func(path string, at time.Time) {
		t.Helper()
		if err := os.Chtimes(path, at, at); err != nil {
			t.Fatal(err)
		}
	}
	write(weights, "weights")
	write(beside, "{}")
	bound := func(cell ModelValidation) ModelValidation {
		t.Helper()
		cell.Model, cell.Recipe = model, recipeID
		cells := []ModelValidation{cell}
		if err := bindCells(t.Context(), testutil.RepoRoot(t), nil, cells); err != nil {
			t.Fatal(err)
		}
		return cells[0]
	}
	// Three cells, by what each one's command is given: the weights file; the directory; and no
	// path at all, a DNA guard, which names its model by identity.
	serving := func() ModelValidation {
		return bound(ModelValidation{Kind: recipe.TaskInference, Validation: servingValidation, Location: weights})
	}
	proof := func() ModelValidation {
		return bound(ModelValidation{Kind: recipe.TaskImageGen, Validation: "image-proof", Location: weights, Directory: directory})
	}
	smoke := func() ModelValidation {
		return bound(ModelValidation{Kind: recipe.TaskInference, Modality: dnaModality, Validation: guardValidation, Location: weights})
	}
	// The cycle keys its phases by this field of the plan, under this name (cmd/loop obligationsDigest).
	consumed := func(cell ModelValidation) any {
		t.Helper()
		var planned map[string]any
		data, err := json.Marshal(cell)
		if err == nil {
			err = json.Unmarshal(data, &planned)
		}
		if err != nil {
			t.Fatal(err)
		}
		return planned["consumed"]
	}
	if named, _ := consumed(proof()).(string); named == "" {
		t.Error("a bound cell's plan entry does not say what its files were")
	}
	if named := consumed(bound(ModelValidation{Kind: recipe.TaskImageGen, Validation: "image-proof", Location: weights})); named != nil {
		t.Errorf("a cell with no command, which never runs, had its files read: %v", named)
	}
	acquired := openCellCredit(kept, "commit-a", "store-a")
	acquire := func(cells ...ModelValidation) {
		t.Helper()
		for _, cell := range cells {
			if err := acquired.add(cell); err != nil {
				t.Fatal(err)
			}
		}
	}
	resumed := func(cell ModelValidation) bool { return openCellCredit(kept, "commit-a", "store-a").has(cell) }
	acquire(serving(), proof(), smoke())
	if !resumed(serving()) || !resumed(proof()) || !resumed(smoke()) {
		t.Fatal("a cell whose files are as they were is not credited to the run that continues it")
	}
	// A file under the directory changes size alone.
	written := modified(beside)
	write(beside, `{"do_resize":true}`)
	stamp(beside, written)
	if resumed(proof()) {
		t.Error("a cell given a directory is credited after a file under it changed size")
	}
	if !resumed(serving()) || !resumed(smoke()) {
		t.Error("a cell that is not given the directory lost its credit to a file under it")
	}
	// The weights file changes time alone.
	stamp(weights, modified(weights).Add(time.Hour))
	if resumed(serving()) {
		t.Error("a cell given its weights file is credited after the file's time moved")
	}
	if resumed(smoke()) {
		t.Error("a cell whose command names its model by identity is credited after its weights file's time moved")
	}
	// A file under the directory changes name alone.
	acquire(proof())
	if !resumed(proof()) {
		t.Fatal("a cell acquired again over the changed files is not credited")
	}
	// A directory's own time is no file's: a directory under the one given moves in time alone.
	stamp(filepath.Dir(beside), modified(beside).Add(time.Hour))
	if !resumed(proof()) {
		t.Error("a cell given a directory lost its credit to the time of a directory under it, which is no file a run reads")
	}
	if err := os.Rename(beside, beside+".moved"); err != nil {
		t.Fatal(err)
	}
	if resumed(proof()) {
		t.Error("a cell given a directory is credited after a file under it was renamed")
	}
	// Where a link can be made, a file reached through one is the file it leads to, not the link.
	// Where none can, the log says so, since nothing about links is held there.
	linked := filepath.Join(t.TempDir(), "tokenizer.json")
	write(linked, "{}")
	if err := os.Symlink(linked, filepath.Join(directory, "tokenizer.json")); err != nil {
		t.Logf("no symbolic link can be made here, so a file reached through a link is not held: %v", err)
	} else {
		acquire(proof())
		write(linked, `{"model":"other"}`)
		if resumed(proof()) {
			t.Error("a cell given a directory is credited after the file a link under it leads to changed")
		}
	}
	// A path that cannot be read, and a remote model's address, are unknown bytes: never credited.
	for name, location := range map[string]string{"an absent file": filepath.Join(directory, "absent.gguf"), "a remote address": "https://provider.example/v1/models/remote"} {
		unknown := func() ModelValidation {
			return bound(ModelValidation{Kind: recipe.TaskInference, Validation: servingValidation, Location: location})
		}
		acquire(unknown())
		if resumed(unknown()) {
			t.Errorf("a cell whose weights are at %s, which nothing can read, is credited", name)
		}
	}
	// One binding of several cells reads each path once and keeps each cell's paths its own: of
	// two cells bound together, the one given the weights file loses its credit to that file's
	// time, and the one given another directory keeps its own.
	apart := t.TempDir()
	write(filepath.Join(apart, "config.json"), "{}")
	together := func() []ModelValidation {
		t.Helper()
		cells := []ModelValidation{
			{Model: model, Recipe: recipeID, Kind: recipe.TaskImageGen, Validation: "image-proof", Location: filepath.Join(apart, "config.json"), Directory: apart},
			{Model: model, Recipe: recipeID, Kind: recipe.TaskInference, Validation: servingValidation, Location: weights},
		}
		if err := bindCells(t.Context(), testutil.RepoRoot(t), nil, cells); err != nil {
			t.Fatal(err)
		}
		return cells
	}
	acquire(together()...)
	stamp(weights, modified(weights).Add(time.Hour))
	if again := together(); !resumed(again[0]) || resumed(again[1]) {
		t.Errorf("of two cells bound together, after the second one's weights file moved in time, the first is credited %t and the second %t; want the first credited and the second not", resumed(again[0]), resumed(again[1]))
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
	store, err := filepath.Abs(filepath.Join(t.TempDir(), "overgodb-store"))
	if err != nil {
		t.Fatal(err)
	}
	kept, weights := filepath.Join(t.TempDir(), "cycle_credit.json"), filepath.Join(t.TempDir(), "model.gguf")
	t.Setenv(clioptions.CycleCreditEnvironment, kept)
	if err := os.WriteFile(weights, []byte("weights"), 0o600); err != nil {
		t.Fatal(err)
	}
	bound := func() Plan {
		t.Helper()
		cells := []ModelValidation{{
			Model: testutil.ArtifactID(t, artifact.KindModel, "kept"), ModelName: "kept", Kind: recipe.TaskInference, Validation: guardValidation,
			Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "kept"), Location: weights,
		}}
		if err := bindCells(t.Context(), testutil.RepoRoot(t), nil, cells); err != nil {
			t.Fatal(err)
		}
		return Plan{Run: []Selected{{ModelValidation: cells[0]}}}
	}
	if err := openCellCredit(kept, head, store).add(bound().Run[0].ModelValidation); err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if err := executeRun(t.Context(), &out, root, store, nil, bound(), scopeCells); err != nil || !strings.Contains(out.String(), "kept credited") {
		t.Fatalf("a cell whose weights are as they were was started again: %v\n%s", err, out.String())
	}
	if err := os.WriteFile(weights, []byte("other weights"), 0o600); err != nil {
		t.Fatal(err)
	}
	out.Reset()
	if err := executeRun(t.Context(), &out, root, store, nil, bound(), scopeCells); err == nil || !strings.Contains(err.Error(), "1 failed") || strings.Contains(out.String(), "credited") {
		t.Fatalf("a cell whose weights changed was not started again: %v\n%s", err, out.String())
	}
}

func withRecipe(cell ModelValidation, id artifact.ID) ModelValidation {
	cell.Recipe = id
	return cell
}

func withProtocol(cell ModelValidation, value string) ModelValidation {
	cell.Protocol = value
	return cell
}

func withDirectory(cell ModelValidation, value string) ModelValidation {
	cell.Directory = value
	return cell
}

func withGrounding(cell ModelValidation, value string) ModelValidation {
	cell.Grounding = value
	return cell
}

func withObjective(cell ModelValidation, value string) ModelValidation {
	cell.Objective = value
	return cell
}

func withConditioning(cell ModelValidation, value string) ModelValidation {
	cell.Conditioning = value
	return cell
}

func withProjector(cell ModelValidation, value string) ModelValidation {
	cell.Projector = value
	return cell
}
