package main

import (
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/loop"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// strategyStore publishes a strategy and its prompt into a new store.
func strategyStore(t *testing.T) (string, loop.Strategy) {
	t.Helper()
	repository := t.TempDir()
	store := testcheck.Must(overgodb.Create(repository))(t)
	data := []byte("strategy prompt")
	prompt := testcheck.Must(artifact.IdentifyBytes(artifact.KindFile, data))(t)
	text := artifact.Content{Descriptor: artifact.Descriptor{ID: prompt, Size: uint64(len(data)), MediaType: "text/plain"}, Data: data}
	worker := testcheck.Must(recipe.NewAgentDefinition(recipe.AgentDefinition{
		Name: "configured-strategy", Prompt: prompt,
		ModelRecipe: testutil.ArtifactID(t, artifact.KindRecipe, "model"),
		Policies:    []artifact.ID{testutil.ArtifactID(t, artifact.KindProfile, "policy")},
	}))(t)
	want := testcheck.Must(loop.NewStrategy(worker, testutil.ArtifactID(t, artifact.KindProfile, "catalog"), loop.Config{MaxAttemptsPerStep: 3, MaxInvocations: 11}))(t)
	content := testcheck.Must(want.Content())(t)
	batch := testcheck.Must(artifact.NewDocumentBatch("fixture/configured-strategy", []artifact.Content{content, text}, nil, nil))(t)
	if _, err := store.CommitAs(t.Context(), gateDoor, batch); err != nil {
		t.Fatal(err)
	}
	testcheck.NoError(t, store.Close())
	return repository, want
}

func TestLoopConfigResolvesStrategyAuthority(t *testing.T) {
	if _, _, err := resolveConfiguredStrategy(config{}, ""); err == nil {
		t.Fatal("automation ran without exact strategy authority")
	}
	repository, want := strategyStore(t)
	got, _, err := resolveConfiguredStrategy(config{StrategyID: want.ID, Repository: repository}, filepath.Join(t.TempDir(), "absent"))
	testcheck.NoError(t, err)
	if got.ID != want.ID || got.Loop.MaxAttemptsPerStep != 3 || got.Loop.MaxInvocations != 11 {
		t.Fatalf("resolved strategy = %+v", got)
	}
}

// TestTheDriverOpensTheStoreItResolved runs from a directory holding no store:
// a configuration that names none reads the strategy and its prompt from the
// store the command resolved, never from a relative folder.
func TestTheDriverOpensTheStoreItResolved(t *testing.T) {
	repository, want := strategyStore(t)
	t.Chdir(t.TempDir())
	got, prompt, err := resolveConfiguredStrategy(config{StrategyID: want.ID}, repository)
	testcheck.NoError(t, err)
	if got.ID != want.ID || prompt != "strategy prompt" {
		t.Fatalf("resolved %+v with prompt %q", got, prompt)
	}
}
