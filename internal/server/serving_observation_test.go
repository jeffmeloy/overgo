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
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/strictjson"
	"overgo/internal/testutil"
)

func TestServingObservationPublication(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelID := testutil.ArtifactID(t, artifact.KindModel, "serving-publication-model")
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "serving-publication-recipe")
	runID := testutil.ArtifactID(t, artifact.KindRun, "serving-publication-run")
	if _, err := store.Commit(t.Context(), artifact.Batch{
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

	operationID, err := handler.operations.Submit(t.Context(), operation.Request{
		Task: recipe.TaskInference, Recipe: recipeID,
	}, func(ctx context.Context, reporter operation.Reporter) (operation.Completion, error) {
		return handler.executeObservedOperation(ctx, reporter, recipe.TaskInference, recipeID, modelID,
			func(context.Context, operation.Reporter) (operation.Completion, error) {
				return operation.Completion{Run: runID}, nil
			})
	})
	if err != nil {
		t.Fatal(err)
	}
	status, err := handler.operations.Wait(t.Context(), operationID)
	if err != nil {
		t.Fatal(err)
	}
	result, err := store.Query(t.Context(), overgodb.Query{
		Artifact: &modelID, Follow: overgodb.FollowChildren, MaxDepth: 1,
		MediaType: runrecord.ServingObservationMediaType, Schema: runrecord.ServingObservationSchema,
		MaxResults: 8,
		Projection: overgodb.ProjectArtifacts,
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
		content, found, err := artifact.ReadContent(t.Context(), store, descriptor.ID)
		if err != nil || !found {
			t.Fatalf("observation content=(%v,%v)", found, err)
		}
		if strings.Contains(string(content.Data), prompt) {
			t.Fatal("serving observation captured request text")
		}
	}
}

func TestRuntimeActivitySessionLedgerGUI(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelID := testutil.ArtifactID(t, artifact.KindModel, "runtime-view-model")
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "runtime-view-recipe")
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key:       "runtime/view/authorities",
		Artifacts: []artifact.Descriptor{{ID: modelID}, {ID: recipeID}},
	}); err != nil {
		t.Fatal(err)
	}
	handler, err := New(Config{
		ModelID: testModelID, MaxTokens: testMaxTokens, Repository: store, MaxStoredResponses: 1,
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
	completion = serveTestRequest(handler, http.MethodPost, "/v1/completions", `{"prompt":"runtime-next","max_tokens":1}`)
	if completion.Code != http.StatusOK {
		t.Fatalf("second completion status=%d body=%s", completion.Code, completion.Body.String())
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
	if response.Code != http.StatusOK || activity.Count != 1 || len(activity.Activity) != 1 || !activity.Truncated ||
		activity.Activity[0].Model != modelID || activity.Activity[0].Recipe != recipeID {
		t.Fatalf("activity status=%d response=%+v", response.Code, activity)
	}
	module := serveTestRequest(handler, http.MethodGet, "/mod/runtime.js", "").Body.String()
	workflow := serveTestRequest(handler, http.MethodGet, "/workflow.js", "").Body.String()
	if !strings.Contains(workflow, "/runtime/activity/stream") || !strings.Contains(module, "runtime.sessions") {
		t.Fatal("runtime GUI lacks shared session stream")
	}
	if strings.Contains(module, `api.get("/slots"`) {
		t.Fatal("runtime module bypasses session authority")
	}
}

func TestOperationEventSSEUsesSharedEmitter(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	defer handler.Close()
	ctx, cancel := context.WithCancel(t.Context())
	recorder := &countingRecorder{ResponseRecorder: httptest.NewRecorder(), flushes: make(chan struct{}, 8)}
	done := make(chan struct{})
	go func() {
		handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/runtime/activity/stream", nil).WithContext(ctx))
		close(done)
	}()
	// The stream opens with three flushed snapshot events -- sessions,
	// activity, operations -- and cancelling earlier races the emitter.
	for range 3 {
		<-recorder.flushes
	}
	cancel()
	<-done
	if recorder.Header().Get("Content-Type") != "text/event-stream" ||
		!strings.Contains(recorder.Body.String(), "event: operation.snapshot") {
		t.Fatalf("operation SSE headers=%v body=%s", recorder.Header(), recorder.Body.String())
	}
}

func TestServingHardwareEvidenceUsesLifecycleBounds(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelID := testutil.ArtifactID(t, artifact.KindModel, "hardware-model")
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "hardware-recipe")
	if _, err := store.Commit(t.Context(), artifact.Batch{
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
	var observations []runrecord.ServingObservation
	_, err = overgodb.VisitDecodedDocuments(t.Context(), store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{
			Kind: artifact.KindEvidence, MediaType: runrecord.ServingObservationMediaType, Schema: runrecord.ServingObservationSchema,
		}}, Order: overgodb.DocumentOldestFirst,
	}, runrecord.ParseServingObservation, func(_ overgodb.DocumentView, observation runrecord.ServingObservation) error {
		observations = append(observations, observation)
		return nil
	})
	if err != nil || len(observations) != 1 {
		t.Fatalf("observations=%d err=%v", len(observations), err)
	}
	observation := observations[0]
	wantStages := []runrecord.ServingHardwareStage{
		runrecord.ServingHardwareStart,
		runrecord.ServingHardwarePrefill,
		runrecord.ServingHardwareFinish,
	}
	expected, err := generator.DeviceMemoryStats(t.Context())
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

type countingRecorder struct {
	*httptest.ResponseRecorder
	flushes chan struct{}
}

func (recorder *countingRecorder) Flush() {
	recorder.ResponseRecorder.Flush()
	select {
	case recorder.flushes <- struct{}{}:
	default:
	}
}
