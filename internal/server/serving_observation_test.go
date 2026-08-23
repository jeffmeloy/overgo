package server

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/operation"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/runrecord"
	"overgo/internal/strictjson"
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
	status, err := handler.operations.Wait(context.Background(), operationID)
	if err != nil {
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
	if len(status.Attempts) != 1 || status.Attempts[0].Kind() != artifact.KindEvidence {
		t.Fatalf("recipe-bound attempts = %v", status.Attempts)
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

func TestWebUIRuntimeActivityUsesSessionAndRepoDBAPIs(t *testing.T) {
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelID := testutil.ArtifactID(t, artifact.KindModel, "runtime-view-model")
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "runtime-view-recipe")
	if _, err := store.Commit(context.Background(), artifact.Batch{
		Key:       "runtime/view/authorities",
		Artifacts: []artifact.Descriptor{{ID: modelID}, {ID: recipeID}},
	}); err != nil {
		t.Fatal(err)
	}
	handler, err := New(Config{
		ModelID: testModelID, MaxTokens: testMaxTokens, Repository: store,
	}, &recipeInspectorGenerator{
		fakeGenerator: &fakeGenerator{},
		description: modelrecipe.RuntimeDescription{
			Task:     recipe.TaskInference,
			Identity: modelrecipe.ProgramIdentity{Model: modelID, Recipe: recipeID},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer handler.Close()
	completion := serveTestRequest(handler, http.MethodPost, "/v1/completions", `{"prompt":"runtime","max_tokens":1}`)
	if completion.Code != http.StatusOK {
		t.Fatalf("completion status=%d body=%s", completion.Code, completion.Body.String())
	}
	var sessions runtimeSessionsResponse
	response := serveTestRequest(handler, http.MethodGet, "/runtime/sessions", "")
	if err := strictjson.DecodeBytes(response.Body.Bytes(), &sessions); err != nil {
		t.Fatal(err)
	}
	if response.Code != http.StatusOK || sessions.Session.Capacity != 1 || len(sessions.Slots) != 1 {
		t.Fatalf("sessions status=%d response=%+v", response.Code, sessions)
	}
	if sessions.Authority == nil || sessions.Authority.Model != modelID || sessions.Authority.Recipe != recipeID {
		t.Fatalf("runtime authority = %+v", sessions.Authority)
	}
	var activity runtimeActivityResponse
	response = serveTestRequest(handler, http.MethodGet, "/runtime/activity", "")
	if err := strictjson.DecodeBytes(response.Body.Bytes(), &activity); err != nil {
		t.Fatal(err)
	}
	if response.Code != http.StatusOK || activity.Count != 1 || len(activity.Activity) != 1 ||
		activity.Activity[0].Model != modelID || activity.Activity[0].Recipe != recipeID {
		t.Fatalf("activity status=%d response=%+v", response.Code, activity)
	}
	module := serveTestRequest(handler, http.MethodGet, "/mod/runtime.js", "").Body.String()
	for _, endpoint := range []string{"/runtime/sessions", "/runtime/activity", "/runtime/activity/stream"} {
		if !strings.Contains(module, endpoint) {
			t.Fatalf("runtime module lacks %q", endpoint)
		}
	}
	if strings.Contains(module, `api.get("/slots"`) {
		t.Fatal("runtime module bypasses session authority")
	}
}

func TestOperationEventSSEUsesSharedEmitter(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	defer handler.Close()
	ctx, cancel := context.WithCancel(t.Context())
	recorder := &signalingRecorder{ResponseRecorder: httptest.NewRecorder(), flushed: make(chan struct{})}
	done := make(chan struct{})
	go func() {
		handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/runtime/activity/stream", nil).WithContext(ctx))
		close(done)
	}()
	<-recorder.flushed
	cancel()
	<-done
	if recorder.Header().Get("Content-Type") != "text/event-stream" ||
		!strings.Contains(recorder.Body.String(), "event: operation.snapshot") {
		t.Fatalf("operation SSE headers=%v body=%s", recorder.Header(), recorder.Body.String())
	}
}

func TestServingHardwareEvidenceUsesLifecycleBounds(t *testing.T) {
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelID := testutil.ArtifactID(t, artifact.KindModel, "hardware-model")
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "hardware-recipe")
	if _, err := store.Commit(context.Background(), artifact.Batch{
		Key:       "serving/hardware/authorities",
		Artifacts: []artifact.Descriptor{{ID: modelID}, {ID: recipeID}},
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
	handler, err := New(Config{Repository: store}, generator)
	if err != nil {
		t.Fatal(err)
	}
	defer handler.Close()
	response := serveTestRequest(handler, http.MethodPost, "/v1/completions", `{"prompt":"hardware","max_tokens":1}`)
	if response.Code != http.StatusOK {
		t.Fatalf("completion status=%d body=%s", response.Code, response.Body.String())
	}
	result, err := store.Query(context.Background(), repodb.Query{
		MediaType:  runrecord.ServingObservationMediaType,
		Schema:     runrecord.ServingObservationSchema,
		MaxResults: repodb.MaxQueryResults,
	})
	if err != nil || len(result.Artifacts) != 1 {
		t.Fatalf("observations=%d err=%v", len(result.Artifacts), err)
	}
	content, found, err := store.Content(context.Background(), result.Artifacts[0].ID)
	if err != nil || !found {
		t.Fatalf("content found=%v err=%v", found, err)
	}
	observation, err := runrecord.ParseServingObservation(content.Data)
	if err != nil {
		t.Fatal(err)
	}
	wantStages := []runrecord.ServingHardwareStage{
		runrecord.ServingHardwareStart,
		runrecord.ServingHardwarePrefill,
		runrecord.ServingHardwareFinish,
	}
	expected, err := generator.DeviceMemoryStats(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if observation.Resources.PeakDeviceBytes != expected.PeakBytes || len(observation.Hardware) != len(wantStages) {
		t.Fatalf("hardware=%+v resources=%+v", observation.Hardware, observation.Resources)
	}
	for index, sample := range observation.Hardware {
		if sample.Stage != wantStages[index] || sample.DeviceCurrentBytes != expected.CurrentBytes ||
			sample.DevicePeakBytes != expected.PeakBytes || sample.DeviceAllocations != expected.Allocations {
			t.Fatalf("sample[%d]=%+v", index, sample)
		}
	}
}
