package vqaserve

import (
	"context"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func TestRunProgramBindsDescriptorInputsAndOutput(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
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
	image, err := ImageContract.ContentBytes([]byte("image"))
	if err != nil {
		t.Fatal(err)
	}
	answer, _, err := RunProgram(
		t.Context(), store, modelID, program,
		"vqa/runtime", image, "where?",
		func(gotImage []byte, question string) (string, error) {
			if string(gotImage) != "image" || question != "where?" {
				t.Fatalf("input = (%q, %q)", gotImage, question)
			}
			return question + " prepared", nil
		},
		func(_ context.Context, prepared string) (string, error) {
			if prepared != "where? prepared" {
				t.Fatalf("prepared = %q", prepared)
			}
			return "there", nil
		},
	)
	if err != nil || answer != "there" {
		t.Fatalf("answer = (%q, %v)", answer, err)
	}
}

// TestRequestFormAndBudget pins the page form: a request names a stored
// image and a question, a blank budget is the serving policy's, and an
// image the store does not hold is refused by name.
func TestRequestFormAndBudget(t *testing.T) {
	imageID := testutil.ArtifactID(t, artifact.KindFile, "vqa-image")
	if err := ValidateRequest(Request{Question: "what?"}); err == nil || !strings.Contains(err.Error(), "image") {
		t.Fatalf("a request without an image validated: %v", err)
	}
	if err := ValidateRequest(Request{Image: imageID, Question: " "}); err == nil || !strings.Contains(err.Error(), "question") {
		t.Fatalf("a request without a question validated: %v", err)
	}
	if err := ValidateRequest(Request{Image: imageID, Question: "what?", MaxTokens: -1}); err == nil {
		t.Fatal("a negative budget validated")
	}
	if err := ValidateRequest(Request{Image: imageID, Question: "what?"}); err != nil {
		t.Fatal(err)
	}
	if budget := (Request{Image: imageID, Question: "what?"}).Budget(); budget != Policy.DecodeMaxSteps || budget <= 0 {
		t.Fatalf("blank budget = %d, policy %d", budget, Policy.DecodeMaxSteps)
	}
	if budget := (Request{Image: imageID, Question: "what?", MaxTokens: 3}).Budget(); budget != 3 {
		t.Fatalf("typed budget = %d", budget)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	if _, err := ReadImage(t.Context(), store, imageID); err == nil || !strings.Contains(err.Error(), "absent") {
		t.Fatalf("an absent image read: %v", err)
	}
	question, err := artifact.JSONContent(QuestionContract, "what?")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), artifact.Batch{Key: "test/vqa/question", Contents: []artifact.Content{question}}); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadImage(t.Context(), store, question.Descriptor.ID); err == nil || !strings.Contains(err.Error(), "not an image") {
		t.Fatalf("a JSON document read as an image: %v", err)
	}
}
