package server

import (
	"bytes"
	"context"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/speechrecognitiontest"
)

func testAudioTranscriptionsLifecycle(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	candidate := speechrecognitiontest.Publish(t, store, "../speechrecognition/testdata/encoder.json")
	policy := TranscriptionPolicy{Recipe: candidate.Definition.ID, MemoryBytes: 1 << 20, Inspection: candidate.Policy}
	if workspace, err := NewTranscriptionWorkspace(t.Context(), store, policy, transcriptionHTTPCommit); err == nil || workspace != nil {
		t.Fatalf("unverified candidate served: workspace=%v err=%v", workspace, err)
	}
	fixture := newTranscriptionHTTPFixture(t, nil)
	definition := fixture.workspace.program.Definition()
	failed, err := runrecord.NewGateRecord(definition.ID, fixture.workspace.environment, transcriptionHTTPCommit,
		runrecord.OutcomeFailed, "fixture-retirement", 1, []runrecord.GateStep{{Name: "fixture-retirement",
			Phase: runrecord.PhaseValidate, Outcome: runrecord.StepFailed, DurationNS: 1}})
	if err != nil {
		t.Fatal(err)
	}
	batch, err := failed.Batch("transcription/fixture-retirement")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(t.Context(), fixture.store, batch); err != nil {
		t.Fatal(err)
	}
	if err := modelrecipe.RetireActiveCapability(t.Context(), fixture.store, definition,
		modelrecipe.Verification{Gate: failed.Result.ID, Run: failed.Run.ID}, "synthetic fixture retirement, not model-quality evidence"); err != nil {
		t.Fatal(err)
	}
	response := httptest.NewRecorder()
	fixture.handler.ServeHTTP(response, fixture.request(t, fixture.wave, nil))
	statuses := fixture.handler.operations.List()
	if response.Code != http.StatusInternalServerError || len(statuses) != 1 || statuses[0].State != operation.StateFailed || statuses[0].Run == nil || len(statuses[0].Outputs) != 0 {
		t.Fatalf("retired recipe executed: HTTP=%d operations=%+v", response.Code, statuses)
	}
	if err := fixture.workspace.Close(context.WithoutCancel(t.Context())); err != nil {
		t.Fatal(err)
	}
	if snapshot := fixture.workspace.sessions.Snapshot(); snapshot.Active != 0 || snapshot.Waiting != 0 {
		t.Fatalf("retirement leaked session: %+v", snapshot)
	}
	var zero TranscriptionWorkspace
	if err := zero.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
	if capabilities, err := zero.WorkflowCapabilities(t.Context(), WorkflowGeneration); err != nil || len(capabilities) != 0 {
		t.Fatalf("zero capabilities=%+v err=%v", capabilities, err)
	}
	if _, err := zero.ExecuteWorkflow(t.Context(), WorkflowGeneration, recipe.TaskTranscription, definition.ID, nil, nil); err == nil {
		t.Fatal("zero workspace accepted execution")
	}
}

func testAudioTranscriptionsMultipart(t *testing.T) {
	fixture := newTranscriptionHTTPFixture(t, nil)
	for _, name := range []string{"file", "model", "response_format", "stream"} {
		t.Run("duplicate-"+name, func(t *testing.T) {
			var body bytes.Buffer
			writer := multipart.NewWriter(&body)
			values := map[string]string{"model": fixture.workspace.policy.Recipe.String(), "response_format": "json", "stream": "false"}
			for _, field := range []string{"file", "model", "response_format", "stream", name} {
				if field == "file" {
					part, err := writer.CreateFormFile(field, "clip.wav")
					if err != nil {
						t.Fatal(err)
					}
					if _, err := part.Write(fixture.wave); err != nil {
						t.Fatal(err)
					}
				} else if err := writer.WriteField(field, values[field]); err != nil {
					t.Fatal(err)
				}
			}
			if err := writer.Close(); err != nil {
				t.Fatal(err)
			}
			request := httptest.NewRequest(http.MethodPost, "/v1/audio/transcriptions", &body)
			request.Header.Set("Content-Type", writer.FormDataContentType())
			request.Header.Set("Authorization", "Bearer "+testAPIKey)
			response := httptest.NewRecorder()
			fixture.handler.ServeHTTP(response, request)
			if response.Code != http.StatusBadRequest {
				t.Fatalf("duplicate accepted: %d %s", response.Code, response.Body)
			}
		})
	}
	request := fixture.request(t, fixture.wave, nil)
	request.Header.Set("Content-Type", "multipart/form-data; boundary=incorrect")
	response := httptest.NewRecorder()
	fixture.handler.ServeHTTP(response, request)
	if response.Code != http.StatusBadRequest || len(fixture.handler.operations.List()) != 0 {
		t.Fatalf("malformed multipart executed: %d %s", response.Code, response.Body)
	}
}
