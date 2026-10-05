package main

import (
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// TestCellCreditSurvivesAKilledRun holds the credit a cycle keeps to what a killed run needs of
// it: a cell a run acquired is credited to the next run of the same commit and store, which is
// a new process reading the file; a cell is the same cell whatever a run changed of it (its
// cost, the record it replays) and another cell when a recipe, a protocol or any committed
// input it reads differs; another commit or store, or no file, credits nothing.
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
	measured.Cost, measured.Replay = 9e9, testutil.ArtifactID(t, artifact.KindEvaluation, "replay")
	if !second.has(measured) {
		t.Error("what a run changes of a cell (its cost, the record it replays) made it another cell")
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
