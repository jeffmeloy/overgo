package main

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/scorecard"
	"overgo/internal/testutil"
)

// TestDeclarationAloneShare holds the scorecard's declaration share to what
// selects serving code: a served model whose architecture a serving file
// names has model-specific code, and its models are listed with those files;
// one whose architecture no serving file names is supported by its
// declaration alone. The share counts each served model once, across its
// tasks, and leaves out a model whose recipe declares no architecture.
func TestDeclarationAloneShare(t *testing.T) {
	t.Parallel()
	llama := testutil.ArtifactID(t, artifact.KindModel, "declaration-llama")
	gemma := testutil.ArtifactID(t, artifact.KindModel, "declaration-gemma")
	forecast := testutil.ArtifactID(t, artifact.KindModel, "declaration-forecast")
	rows := []scorecard.Row{
		{Model: llama, Name: "Small-Llama", Task: recipe.TaskInference, Architecture: "llama"},
		{Model: gemma, Name: "Gemma", Task: recipe.TaskInference, Architecture: "gemma4"},
		{Model: gemma, Name: "Gemma", Task: recipe.TaskTraining, Architecture: "gemma4"},
		{Model: forecast, Name: "Forecast", Task: recipe.TaskForecast},
	}
	share := declarationShare(rows, map[string][]string{"llama": {"overgo/internal/serving/load.go"}})
	want := "Supported by declaration alone: 1 of 2 served models; the architecture of each other selects serving code by name:\n" +
		"- llama (Small-Llama): overgo/internal/serving/load.go\n"
	if share != want {
		t.Fatalf("declaration share:\n%s\nwant:\n%s", share, want)
	}
	if share := declarationShare(rows, nil); share != "Supported by declaration alone: 2 of 2 served models.\n" {
		t.Fatalf("declaration share with no serving name = %q", share)
	}
}
