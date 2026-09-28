package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/inferencesurface"
	"overgo/internal/recipe"
	"overgo/internal/scorecard"
	"overgo/internal/testutil"
)

// TestDeclarationAloneShare holds the scorecard's declaration share to what
// selects serving code: an architecture named by a string literal in a
// serving package's source has model-specific code, and its models are
// listed with the files that name it; one named only in a test file, or only
// in the tokenizer, whose names are tokenizer types that can coincide with an
// architecture's, is supported by its declaration alone. The share counts
// each served model once, across its tasks, and leaves out a model whose
// recipe declares no architecture.
func TestDeclarationAloneShare(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	write := func(name, content string) {
		t.Helper()
		file := filepath.Join(root, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(file), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(file, []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	write("serving/load.go", "package serving\n\nfunc special(name string) bool { return name == \"llama\" }\n")
	write("serving/load_test.go", "package serving\n\nconst fixture = \"gemma4\"\n")
	write("tokenizer/vocab.go", "package tokenizer\n\nconst sentencePiece = `gemma4`\n")
	named, err := servingNames([]inferencesurface.Package{
		{Dir: filepath.Join(root, "serving"), ImportPath: "overgo/internal/serving"},
		{Dir: filepath.Join(root, "tokenizer"), ImportPath: "overgo" + tokenizerPackage},
	})
	if err != nil {
		t.Fatal(err)
	}
	llama := testutil.ArtifactID(t, artifact.KindModel, "declaration-llama")
	gemma := testutil.ArtifactID(t, artifact.KindModel, "declaration-gemma")
	forecast := testutil.ArtifactID(t, artifact.KindModel, "declaration-forecast")
	rows := []scorecard.Row{
		{Model: llama, Name: "Small-Llama", Task: recipe.TaskInference, Architecture: "llama"},
		{Model: gemma, Name: "Gemma", Task: recipe.TaskInference, Architecture: "gemma4"},
		{Model: gemma, Name: "Gemma", Task: recipe.TaskTraining, Architecture: "gemma4"},
		{Model: forecast, Name: "Forecast", Task: recipe.TaskForecast},
	}
	share := declarationShare(rows, named)
	want := "Supported by declaration alone: 1 of 2 served models; the architecture of each other selects serving code by name:\n" +
		"- llama (Small-Llama): overgo/internal/serving/load.go\n"
	if share != want {
		t.Fatalf("declaration share:\n%s\nwant:\n%s", share, want)
	}
	if strings.Contains(share, "gemma4") {
		t.Fatalf("a name only a test or the tokenizer holds was counted as serving code: %s", share)
	}
}
