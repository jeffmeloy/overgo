package server

import (
	"encoding/json"
	"net/http"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestArtifactLineageListsProducersAndConsumers pins the lineage route: an
// output lists the run that produced it with that run's request document
// and media input described, and the run that consumed it with its
// outputs; a generator without a workspace derives no next step.
func TestArtifactLineageListsProducersAndConsumers(t *testing.T) {
	root := t.TempDir()
	store, err := overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	content := func(kind artifact.Kind, mediaType, schema string, data []byte) artifact.Content {
		t.Helper()
		id, err := artifact.IdentifyBytes(kind, data)
		if err != nil {
			t.Fatal(err)
		}
		return artifact.Content{Descriptor: artifact.Descriptor{ID: id, Size: uint64(len(data)), MediaType: mediaType, Schema: schema}, Data: data}
	}
	request := content(artifact.KindFile, artifact.JSONMediaType, "overgo.video-edit-input.v1", []byte(`{"seed":3}`))
	source := content(artifact.KindFile, "video/gif", "", []byte("a source clip"))
	output := content(artifact.KindOutput, "image/png", "", []byte("the produced image"))
	answer := content(artifact.KindOutput, "text/plain", "", []byte("an answer about it"))
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "lineage-recipe")
	if _, err := store.Commit(t.Context(), artifact.Batch{Key: "fixture/lineage/facts", Artifacts: []artifact.Descriptor{{ID: recipeID}}, Contents: []artifact.Content{request, source, output, answer}}); err != nil {
		t.Fatal(err)
	}
	commitRun := func(key string, inputs, outputs []artifact.ID) artifact.ID {
		t.Helper()
		run, err := runrecord.NewRun(recipeID, runrecord.OutcomeSucceeded, inputs, outputs, "")
		if err != nil {
			t.Fatal(err)
		}
		batch, err := run.Batch(key)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.Commit(t.Context(), batch); err != nil {
			t.Fatal(err)
		}
		return run.ID
	}
	producer := commitRun("fixture/lineage/producer", []artifact.ID{request.Descriptor.ID, source.Descriptor.ID}, []artifact.ID{output.Descriptor.ID})
	consumer := commitRun("fixture/lineage/consumer", []artifact.ID{output.Descriptor.ID}, []artifact.ID{answer.Descriptor.ID})
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	handler, err := New(Config{OvergoDBPath: root, MaxStoredResponses: 8}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = handler.Close() })
	page := serveTestRequest(handler, http.MethodGet, "/artifacts/lineage?id="+output.Descriptor.ID.String(), "")
	var lineage artifactLineageResponse
	if page.Code != http.StatusOK || json.Unmarshal(page.Body.Bytes(), &lineage) != nil {
		t.Fatalf("status=%d body=%s", page.Code, page.Body.String())
	}
	if lineage.ID != output.Descriptor.ID || lineage.MediaType != "image/png" || len(lineage.Producers) != 1 || len(lineage.Consumers) != 1 || len(lineage.Next) != 0 {
		t.Fatalf("lineage = %+v", lineage)
	}
	// A run record keeps its inputs in identity order: find each by id.
	made := lineage.Producers[0]
	described := map[artifact.ID]artifactLineageInput{}
	for _, input := range made.Inputs {
		described[input.ID] = input
	}
	if made.Run != producer || made.Recipe != recipeID || made.Outcome != runrecord.OutcomeSucceeded || len(made.Inputs) != 2 ||
		described[request.Descriptor.ID].Schema != "overgo.video-edit-input.v1" || described[request.Descriptor.ID].MediaType != artifact.JSONMediaType ||
		described[source.Descriptor.ID].MediaType != "video/gif" || len(made.Outputs) != 1 || made.Outputs[0] != output.Descriptor.ID {
		t.Fatalf("producer = %+v", made)
	}
	used := lineage.Consumers[0]
	if used.Run != consumer || len(used.Inputs) != 1 || used.Inputs[0].ID != output.Descriptor.ID || len(used.Outputs) != 1 || used.Outputs[0] != answer.Descriptor.ID {
		t.Fatalf("consumer = %+v", used)
	}
	if missing := serveTestRequest(handler, http.MethodGet, "/artifacts/lineage?id="+testutil.ArtifactID(t, artifact.KindOutput, "absent").String(), ""); missing.Code != http.StatusNotFound {
		t.Fatalf("an absent artifact answered %d: %s", missing.Code, missing.Body.String())
	}
}

// TestNextStepsFollowDeclaredSlots pins the next-step derivation: a
// capability is a next step once per artifact slot whose declared media
// takes the artifact's type; a refused capability and a slot of another
// kind are not.
func TestNextStepsFollowDeclaredSlots(t *testing.T) {
	vqa := testutil.ArtifactID(t, artifact.KindRecipe, "vqa")
	edit := testutil.ArtifactID(t, artifact.KindRecipe, "edit")
	refused := testutil.ArtifactID(t, artifact.KindRecipe, "refused")
	capabilities := []WorkflowCapability{
		{Task: recipe.TaskVQA, Recipe: vqa, Name: "vqa.safetensors", Controls: []WorkflowControl{
			{Name: "image", Type: WorkflowControlArtifact, Label: "image", Media: "image"}, {Name: "question", Type: WorkflowControlText}}},
		{Task: recipe.TaskVideoGen, Recipe: edit, Name: "edit.safetensors", Controls: []WorkflowControl{
			{Name: "source_artifact", Type: WorkflowControlArtifact, Label: "source clip", Media: "video"}}},
		{Task: recipe.TaskVQA, Recipe: refused, Name: "refused.safetensors", Refusal: "its request carries a tensor", Controls: []WorkflowControl{
			{Name: "image", Type: WorkflowControlArtifact, Label: "image", Media: "image"}}},
	}
	images := nextSteps(capabilities, "image/png")
	if len(images) != 1 || images[0].Recipe != vqa || images[0].Task != recipe.TaskVQA || images[0].Control != "image" || images[0].Label != "image" || images[0].Name != "vqa.safetensors" {
		t.Fatalf("next steps for an image = %+v", images)
	}
	clips := nextSteps(capabilities, "video/gif")
	if len(clips) != 1 || clips[0].Recipe != edit || clips[0].Control != "source_artifact" || clips[0].Label != "source clip" {
		t.Fatalf("next steps for a clip = %+v", clips)
	}
	if none := nextSteps(capabilities, "audio/wav"); len(none) != 0 {
		t.Fatalf("next steps for a clip of audio = %+v", none)
	}
}
