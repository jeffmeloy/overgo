package main

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestVideoCellsTimeAUserSizeRequest: a video model's cells hold a request at the size a person asks for beside the small
// one, a timed cell the scorecard waits on, on the directory the small one reads. Its measure asks for it by name.
func TestVideoCellsTimeAUserSizeRequest(t *testing.T) {
	t.Parallel()
	if got := kindValidation(recipe.TaskVideoGen); !slices.Equal(got, []string{"video-proof", videoUserSize}) {
		t.Fatalf("a video model's cells = %v", got)
	}
	if kinds := kindValidation(recipe.TaskImageGen); slices.Contains(kinds, videoUserSize) {
		t.Fatalf("an image model's cells = %v", kinds)
	}
	if !timedValidation(videoUserSize) {
		t.Fatal("the reading is a time, so it runs apart from other work")
	}
	cell := ModelValidation{
		Model: testutil.ArtifactID(t, artifact.KindModel, "wan"), Kind: recipe.TaskVideoGen, Validation: videoUserSize,
		Directory: "C:/models/wan", Location: "C:/models/wan/weights.safetensors",
	}
	name, args, err := reacquireCommand(cell, "C:/stores/selected")
	testcheck.NoError(t, err)
	want := []string{"run", "./cmd/recipe", "measure", "-repo", "C:/stores/selected", "-task", "video-gen", "-user-size", "C:/models/wan"}
	if name != "go" || !slices.Equal(args, want) {
		t.Fatalf("the command = %s %v, want go %v", name, args, want)
	}
	cell.Directory = ""
	if _, _, err := reacquireCommand(cell, "C:/stores/selected"); err == nil {
		t.Fatal("a model with no registered directory has a command")
	}
	plan := SelectValidation(Inputs{Registered: []ModelValidation{{
		Model: cell.Model, Kind: recipe.TaskVideoGen, Validation: videoUserSize, Directory: "C:/models/wan",
	}}})
	if len(plan.Run) != 1 || plan.Run[0].Validation != videoUserSize {
		t.Fatalf("the plan runs %+v, want the reading among the cells the scorecard waits on", plan.Run)
	}
}
