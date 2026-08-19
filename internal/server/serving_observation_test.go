package server

import (
	"context"
	"net/http"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/operation"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestServingObservationPublication(t *testing.T) {
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelID := testutil.ArtifactID(t, artifact.KindModel, "serving-publication-model")
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "serving-publication-recipe")
	runID := testutil.ArtifactID(t, artifact.KindRun, "serving-publication-run")
	if _, err := store.Commit(context.Background(), artifact.Batch{
		Key:       "serving/publication/authorities",
		Artifacts: []artifact.Descriptor{{ID: modelID}, {ID: recipeID}, {ID: runID}},
	}); err != nil {
		t.Fatal(err)
	}
	generator := &recipeInspectorGenerator{
		fakeGenerator: &fakeGenerator{},
		description: modelrecipe.RuntimeDescription{
			Task:     recipe.TaskInference,
			Identity: modelrecipe.ProgramIdentity{Model: modelID, Recipe: recipeID},
		},
	}
	handler, err := New(Config{
		ModelID: testModelID, MaxTokens: testMaxTokens,
		DefaultTemperature: testNeutralTemperature, DefaultTopP: testFullTopP,
		Repository: store, Analysis: testAnalysisPolicy,
	}, generator)
	if err != nil {
		t.Fatal(err)
	}
	defer handler.Close()
	const prompt = "private-serving-prompt"
	response := serveTestRequest(handler, http.MethodPost, "/v1/completions", `{"prompt":"`+prompt+`","max_tokens":1}`)
	if response.Code != http.StatusOK {
		t.Fatalf("completion status=%d body=%s", response.Code, response.Body.String())
	}

	operationID, err := handler.operations.Submit(context.Background(), operation.Request{
		Task: recipe.TaskInference, Recipe: recipeID,
	}, func(ctx context.Context, reporter operation.Reporter) (operation.Completion, error) {
		return handler.executeObservedOperation(ctx, reporter, recipe.TaskInference, recipeID,
			func(context.Context, operation.Reporter) (operation.Completion, error) {
				return operation.Completion{Run: runID}, nil
			})
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := handler.operations.Wait(context.Background(), operationID); err != nil {
		t.Fatal(err)
	}
	result, err := store.Query(context.Background(), repodb.Query{
		Artifact: &modelID, Follow: repodb.FollowChildren, MaxDepth: 1,
		MediaType: runrecord.ServingObservationMediaType, Schema: runrecord.ServingObservationSchema,
		MaxResults: 8,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Artifacts) != 2 || handler.observationErrors.Load() != 0 {
		t.Fatalf("observations=%d publication_errors=%d", len(result.Artifacts), handler.observationErrors.Load())
	}
	for _, descriptor := range result.Artifacts {
		content, found, err := store.Content(context.Background(), descriptor.ID)
		if err != nil || !found {
			t.Fatalf("observation content=(%v,%v)", found, err)
		}
		if strings.Contains(string(content.Data), prompt) {
			t.Fatal("serving observation captured request text")
		}
	}
}
