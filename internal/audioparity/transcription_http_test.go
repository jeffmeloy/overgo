package audioparity

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/server"
	"overgo/internal/tokenizer"
)

const audioHTTPFixtureKey = "audio-integration-test-only-key"

type audioHTTPRuntime struct {
	*server.TranscriptionWorkspace
	observe func(context.Context, operation.Reporter)
}

// Generate refuses text generation; the fixture executes only the real audio
// workflow exposed by its embedded production workspace.
func (*audioHTTPRuntime) Generate(context.Context, string, inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	return nil, "", errors.New("text generation is not part of this audio fixture")
}

// ExecuteWorkflow exposes a test observation of admission, then invokes the
// unchanged production operation. It never replaces inference or publication.
func (runtime *audioHTTPRuntime) ExecuteWorkflow(ctx context.Context, kind server.WorkflowKind, task recipe.Task, id artifact.ID, raw json.RawMessage, reporter operation.Reporter) (operation.Completion, error) {
	if runtime.observe != nil {
		runtime.observe(ctx, reporter)
	}
	return runtime.TranscriptionWorkspace.ExecuteWorkflow(ctx, kind, task, id, raw, reporter)
}

func newAudioHTTPFixture(t *testing.T, store *overgodb.Store, definition recipe.Definition, policy dataset.AudioInspectionPolicy, commit string) (*server.Handler, *audioHTTPRuntime) {
	t.Helper()
	// Actual inference/evaluation precedes this isolated routing fixture. This
	// scaffolding does not claim a quality win or production promotion.
	verification, err := modelrecipetest.PublishVerification(t.Context(), store, "audio/http-verification/"+definition.ID.String(), definition.ID)
	if err != nil {
		t.Fatal(err)
	}
	if err := modelrecipe.ActivateCapability(t.Context(), store, definition, verification, recipe.EvidenceParity, "isolated audio HTTP fixture; not production promotion"); err != nil {
		t.Fatal(err)
	}
	workspace, err := server.NewTranscriptionWorkspace(t.Context(), store, server.TranscriptionPolicy{Recipe: definition.ID, MemoryBytes: adapterAcceptanceMemory, Inspection: policy}, commit)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := workspace.Close(context.WithoutCancel(t.Context())); err != nil {
			t.Error(err)
		}
	})
	runtimePolicy, found, err := modelrecipe.CatalogRuntimePolicy(recipe.TaskInference)
	if err != nil || !found {
		t.Fatalf("HTTP host policy absent: %v", err)
	}
	runtime := &audioHTTPRuntime{TranscriptionWorkspace: workspace}
	handler, err := server.New(server.Config{Repository: store, APIKey: audioHTTPFixtureKey, RuntimePolicy: runtimePolicy}, runtime)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := handler.Close(); err != nil {
			t.Error(err)
		}
	})
	return handler, runtime
}

func audioHTTPOperations(t *testing.T, handler *server.Handler) []operation.Status {
	t.Helper()
	request := httptest.NewRequest(http.MethodGet, "/operations", nil)
	request.Header.Set("Authorization", "Bearer "+audioHTTPFixtureKey)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	var statuses []operation.Status
	if response.Code != http.StatusOK || json.Unmarshal(response.Body.Bytes(), &statuses) != nil {
		t.Fatalf("operation status %d: %s", response.Code, response.Body)
	}
	return statuses
}

func audioHTTPRequest(t *testing.T, definition recipe.Definition, data []byte, authorized bool) *http.Request {
	t.Helper()
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormFile("file", "source.flac")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := part.Write(data); err != nil {
		t.Fatal(err)
	}
	if err := writer.WriteField("model", definition.ID.String()); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodPost, "/v1/audio/transcriptions", &body)
	request.Header.Set("Content-Type", writer.FormDataContentType())
	if authorized {
		request.Header.Set("Authorization", "Bearer "+audioHTTPFixtureKey)
	}
	return request
}
