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
	head, sequence := store.Head()
	retained, err := ReadImage(t.Context(), store, image.Descriptor.ID)
	if err != nil || retained.Descriptor != image.Descriptor || string(retained.Data) != string(image.Data) {
		t.Fatalf("workflow image cannot enter the next request unchanged: %v", err)
	}
	if after, afterSequence := store.Head(); after != head || afterSequence != sequence {
		t.Fatal("reading the workflow image changed the store")
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
	for _, test := range []struct {
		name, media, schema string
		accepted            bool
	}{
		{"declared image", "image/jpeg", "overgo.image-input.v1", true},
		{"untyped binary", ImageContract.MediaType, "overgo.binary-fixture.v1", false},
		{"wrong media", artifact.JSONMediaType, ImageContract.Schema, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			content, err := (artifact.DocumentContract{Kind: artifact.KindFile, MediaType: test.media, Schema: test.schema}).ContentBytes([]byte(test.name))
			if err != nil {
				t.Fatal(err)
			}
			if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{Key: "test/vqa/" + test.name, Contents: []artifact.Content{content}}); err != nil {
				t.Fatal(err)
			}
			read, err := ReadImage(t.Context(), store, content.Descriptor.ID)
			if (err == nil) != test.accepted || test.accepted && (read.Descriptor != content.Descriptor || string(read.Data) != string(content.Data)) {
				t.Fatalf("image declaration admission = %+v, %v", read.Descriptor, err)
			}
		})
	}
}
