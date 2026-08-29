package main

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/loop"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func TestLoopConfigResolvesStrategyAuthority(t *testing.T) {
	if _, err := resolveConfiguredStrategy(config{}); err == nil {
		t.Fatal("automation ran without exact strategy authority")
	}

	repository := t.TempDir()
	store, err := overgodb.Open(repository)
	if err != nil {
		t.Fatal(err)
	}
	worker, err := recipe.NewAgentDefinition(recipe.AgentDefinition{
		Name: "configured-strategy", Prompt: testutil.ArtifactID(t, artifact.KindFile, "prompt"),
		ModelRecipe: testutil.ArtifactID(t, artifact.KindRecipe, "model"),
		Policies:    []artifact.ID{testutil.ArtifactID(t, artifact.KindProfile, "policy")},
	})
	if err != nil {
		t.Fatal(err)
	}
	want, err := loop.NewStrategy(worker, testutil.ArtifactID(t, artifact.KindProfile, "catalog"), loop.Config{MaxAttemptsPerStep: 3, MaxInvocations: 11, SaturationLimit: 2})
	if err != nil {
		t.Fatal(err)
	}
	content, err := want.Content()
	if err != nil {
		t.Fatal(err)
	}
	batch, err := artifact.NewDocumentBatch("fixture/configured-strategy", []artifact.Content{content}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	got, err := resolveConfiguredStrategy(config{StrategyID: want.ID, Repository: repository})
	if err != nil {
		t.Fatal(err)
	}
	if got.ID != want.ID || got.Loop.MaxAttemptsPerStep != 3 || got.Loop.MaxInvocations != 11 || got.Loop.SaturationLimit != 2 {
		t.Fatalf("resolved strategy = %+v", got)
	}
}
