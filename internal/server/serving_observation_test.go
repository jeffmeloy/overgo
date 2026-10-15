package server

import (
	"context"
	"net/http"
	"net/http/httptest"
	"overgo/internal/processmeasure"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/strictjson"
	"overgo/internal/testutil"
)

type realizedServingGenerator struct{ *recipeInspectorGenerator }

func (*realizedServingGenerator) RealizedResidency() recipe.RealizedResidency {
	return recipe.RealizedOOMStreamed
}

func TestServingObservationPublication(t *testing.T) {
	t.Parallel()
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
	}, &realizedServingGenerator{generator})
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
	if authority := handler.runtimeSessionsSnapshot().Authority; authority == nil || authority.RealizedResidency != recipe.RealizedOOMStreamed {
		t.Fatalf("runtime session lost actual residency: %+v", authority)
	}
	for _, descriptor := range result.Artifacts {
		content, found, err := artifact.ReadContent(t.Context(), store, descriptor.ID)
		if err != nil || !found {
			t.Fatalf("observation content=(%v,%v)", found, err)
		}
		if strings.Contains(string(content.Data), prompt) {
			t.Fatal("serving observation captured request text")
		}
		observation, err := runrecord.ParseServingObservation(content.Data)
		if err != nil || observation.RealizedResidency != recipe.RealizedOOMStreamed {
			t.Fatalf("serving residency was not observed: %+v, %v", observation, err)
		}
	}
}

func TestRuntimeActivitySessionLedgerGUI(t *testing.T) {
	t.Parallel()
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
}

func TestServingHardwareEvidenceUsesLifecycleBounds(t *testing.T) {
	t.Parallel()
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
		fakeGenerator: &fakeGenerator{started: make(chan struct{}), release: make(chan struct{}), promptDuration: time.Nanosecond},
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
	responses := make(chan *httptest.ResponseRecorder, 1)
	go func() {
		responses <- serveTestRequest(handler, http.MethodPost, "/v1/completions", `{"prompt":"hardware","max_tokens":1}`)
	}()
	<-generator.started
	wait := processmeasure.NewStopwatch()
	waited, err := wait.Elapsed()
	for waited == 0 && err == nil {
		waited, err = wait.Elapsed()
	}
	close(generator.release)
	response := <-responses
	if err != nil {
		t.Fatal(err)
	}
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
	for _, phase := range observation.Phases {
		if phase.Phase == runrecord.PhaseDecode && phase.DurationNS > observation.MeasuredNS-waited {
			t.Errorf("decode %d includes the %d ns wait before prompt evaluation (request %d)", phase.DurationNS, waited, observation.MeasuredNS)
		}
	}
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
	for _, test := range []struct {
		prompt, evaluated, total uint64
		want                     []runrecord.PhaseMetric
	}{
		{0, 0, 0, nil},
		{0, 0, 10, []runrecord.PhaseMetric{{Phase: runrecord.PhaseDecode, DurationNS: 10}}},
		{5, 0, 10, []runrecord.PhaseMetric{{Phase: runrecord.PhasePrefill, DurationNS: 5}, {Phase: runrecord.PhaseDecode, DurationNS: 5}}},
		{5, 8, 10, []runrecord.PhaseMetric{{Phase: runrecord.PhasePrefill, DurationNS: 5}, {Phase: runrecord.PhaseDecode, DurationNS: 2}}},
		{12, 15, 10, []runrecord.PhaseMetric{{Phase: runrecord.PhasePrefill, DurationNS: 10}}},
	} {
		if got := servingPhases(test.prompt, test.evaluated, test.total); !slices.Equal(got, test.want) {
			t.Errorf("phases(%d, %d, %d) = %+v, want %+v", test.prompt, test.evaluated, test.total, got, test.want)
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
