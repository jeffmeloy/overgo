package main

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/scorecard"
	"overgo/internal/testutil"
)

// TestARowOfAModelTheListLeavesOutIsUnchosen holds the scorecard's mark of the models a full run leaves out to the
// names its list gives, read as -model reads them (carried-unselected, SQA-527): a model listed by the name its row
// shows, and one listed by the name of the file the catalog finds it by, are chosen; Qwen3.5-9B, which the list leaves
// to Qwen3.5-4B for the path the two share, is marked Unchosen in each of its rows, so its carried readings are owed
// none.
func TestARowOfAModelTheListLeavesOutIsUnchosen(t *testing.T) {
	t.Parallel()
	model := func(name string) artifact.ID { return testutil.ArtifactID(t, artifact.KindModel, name) }
	small, large, vae := model("Qwen3.5-4B"), model("Qwen3.5-9B"), model("Wan VAE")
	rows := []scorecard.Row{
		{Model: small, Name: "Qwen3.5-4B", Task: recipe.TaskInference},
		{Model: large, Name: "Qwen3.5-9B", Task: recipe.TaskInference},
		{Model: large, Name: "Qwen3.5-9B", Task: recipe.TaskTraining},
		{Model: vae, Name: "Wan2.1 VAE", Task: recipe.TaskReconstruction},
	}
	registered := []ModelValidation{{Model: vae, ModelName: "Wan2.1_VAE.pth"}, {Model: large, ModelName: "Qwen3.5-9B-bf16.safetensors"}}
	marked := unchosen(rows, registered, []string{"Qwen3.5-4B", "Wan2.1_VAE.pth"})
	for index, row := range marked {
		if want := row.Model == large; row.Unchosen != want {
			t.Errorf("%s %s is marked unchosen %t, want %t", row.Name, row.Task, row.Unchosen, want)
		}
		if rows[index].Unchosen {
			t.Fatal("marking the rows changed the rows it was given")
		}
	}
}
