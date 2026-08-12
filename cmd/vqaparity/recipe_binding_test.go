package main

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/testutil"
)

func TestExecuteVQABindsDescriptorInputsAndOutput(t *testing.T) {
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	modelID := testutil.ArtifactID(t, artifact.KindModel, "vqa-model")
	testutil.PublishArtifact(t, store, modelID)
	definition, err := modelrecipe.CapabilityDefinition(recipe.TaskVQA, modelID)
	if err != nil {
		t.Fatal(err)
	}
	program, err := modelrecipe.CompileCapability(definition)
	if err != nil {
		t.Fatal(err)
	}
	image := []byte("image")
	answer, err := executeVQA(
		context.Background(), store, modelID, program,
		"vqa/runtime", image, "where?",
		func(_ context.Context, gotImage []byte, question string) (string, error) {
			if string(gotImage) != string(image) || question != "where?" {
				t.Fatalf("input = (%q, %q)", gotImage, question)
			}
			return "there", nil
		},
	)
	if err != nil || answer != "there" {
		t.Fatalf("answer = (%q, %v)", answer, err)
	}
}
