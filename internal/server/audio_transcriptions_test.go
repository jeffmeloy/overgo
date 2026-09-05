package server

import (
	"bytes"
	"context"
	"encoding/hex"
	"encoding/json"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/speechrecognition"
	"overgo/internal/speechrecognitiontest"
	"overgo/internal/testutil"
)

const transcriptionHTTPCommit = "0123456789abcdef0123456789abcdef01234567"

type transcriptionHTTPRuntime struct {
	*fakeGenerator
	WorkflowWorkspaceAPI
}

type transcriptionHTTPFixture struct {
	handler   *Handler
	workspace *TranscriptionWorkspace
	store     *overgodb.Store
	wave      []byte
}

func newTranscriptionHTTPFixture(t *testing.T, configure func(*TranscriptionPolicy)) transcriptionHTTPFixture {
	t.Helper()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	fixture := speechrecognitiontest.Publish(t, store, "../speechrecognition/testdata/encoder.json")
	verification, err := modelrecipetest.PublishVerification(t.Context(), store, "transcription/fixture-verification", fixture.Definition.ID)
	if err != nil {
		t.Fatal(err)
	}
	if err := modelrecipe.ActivateCapability(t.Context(), store, fixture.Definition, verification, recipe.EvidenceParity, "isolated HTTP fixture activation; not production model-quality evidence"); err != nil {
		t.Fatal(err)
	}
	policy := TranscriptionPolicy{Recipe: fixture.Definition.ID, MemoryBytes: 1 << 20, Inspection: fixture.Policy}
	if configure != nil {
		configure(&policy)
	}
	workspace, err := NewTranscriptionWorkspace(t.Context(), store, policy, transcriptionHTTPCommit)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := workspace.Close(context.WithoutCancel(t.Context())); err != nil {
			t.Error(err)
		}
	})
	runtime := &transcriptionHTTPRuntime{fakeGenerator: &fakeGenerator{}, WorkflowWorkspaceAPI: workspace}
	handler, err := New(Config{Repository: store, APIKey: testAPIKey}, runtime)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = handler.Close() })
	// A co-hosted text model must never become the ASR observation's model.
	textModel := testutil.ArtifactID(t, artifact.KindModel, "unrelated text model")
	if _, err := store.Commit(t.Context(), artifact.Batch{Key: "transcription/text-model", Artifacts: []artifact.Descriptor{{ID: textModel}}}); err != nil {
		t.Fatal(err)
	}
	handler.modelArtifact = textModel
	wave, _ := speechrecognitiontest.Wave(t, false)
	return transcriptionHTTPFixture{handler: handler, workspace: workspace, store: store, wave: wave}
}

func (fixture transcriptionHTTPFixture) request(t *testing.T, data []byte, fields map[string]string) *http.Request {
	t.Helper()
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormFile("file", "clip.wav")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := part.Write(data); err != nil {
		t.Fatal(err)
	}
	if fields == nil {
		if err := writer.WriteField("model", fixture.workspace.policy.Recipe.String()); err != nil {
			t.Fatal(err)
		}
	}
	for name, value := range fields {
		if err := writer.WriteField(name, value); err != nil {
			t.Fatal(err)
		}
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodPost, "/v1/audio/transcriptions", &body)
	request.Header.Set("Content-Type", writer.FormDataContentType())
	request.Header.Set("Authorization", "Bearer "+testAPIKey)
	return request
}

func TestAudioTranscriptionsAcceptance(t *testing.T) {
	t.Run("native-cpu-and-repeat-lineage", testAudioTranscriptionsNative)
	t.Run("authentication-before-upload", testAudioTranscriptionsAuthentication)
	t.Run("invalid-input-and-controls", testAudioTranscriptionsInvalid)
	t.Run("upload-and-sample-bounds", testAudioTranscriptionsLimits)
	t.Run("cancelled-operation-releases-session", testAudioTranscriptionsCancellation)
	t.Run("flac-native-cpu", testAudioTranscriptionsFLAC)
	t.Run("recipe-admission-and-retirement", testAudioTranscriptionsLifecycle)
	t.Run("multipart-contract", testAudioTranscriptionsMultipart)
}

func testAudioTranscriptionsFLAC(t *testing.T) {
	fixture := newTranscriptionHTTPFixture(t, nil)
	// Independently encoded and round-trip checked against Wave's 128 PCM samples:
	// SoundFile 0.14.0, libsndfile 1.2.2; SHA256
	// e5d8371b075ab406db365306e2ad28ddb5d3c11803fb2fe049cd1a19d6f33347.
	data, err := hex.DecodeString("664c6143000000221000100000006300006303e800f0000000804da079258be0ec51abb4c859ae07606e84000028200000007265666572656e6365206c6962464c414320312e342e3320323032333036323300000000fff86508007f8f4980000c3e2d407641000b4223c30382e5685280776b4bfeefb9cc295dd76b4bfeefb9cc295dd76b4bfeefb9cc295dd76b4bfeefb9cc295dd76b4bfeefb9cc295dd76b4bfeefb9cc295dd76b4bfeefb9cc295dd76b4bfeefb9c02971")
	if err != nil {
		t.Fatal(err)
	}
	response := httptest.NewRecorder()
	fixture.handler.ServeHTTP(response, fixture.request(t, data, map[string]string{
		"model": fixture.workspace.policy.Recipe.String(), "response_format": "json", "stream": "false",
	}))
	var output map[string]string
	if err := json.Unmarshal(response.Body.Bytes(), &output); err != nil || response.Code != http.StatusOK || output["text"] != "x" || len(output) != 1 {
		t.Fatalf("FLAC status=%d output=%s decode=%v", response.Code, response.Body, err)
	}
}

func testAudioTranscriptionsNative(t *testing.T) {
	fixture := newTranscriptionHTTPFixture(t, nil)
	runs := make(map[artifact.ID]bool)
	for range 2 {
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, fixture.request(t, fixture.wave, nil))
		var output map[string]string
		if err := json.Unmarshal(response.Body.Bytes(), &output); err != nil || response.Code != http.StatusOK || output["text"] != "x" || len(output) != 1 {
			t.Fatalf("status=%d output=%s decode=%v", response.Code, response.Body, err)
		}
	}
	statuses := fixture.handler.operations.List()
	if len(statuses) != 2 {
		t.Fatalf("operations=%d", len(statuses))
	}
	for _, status := range statuses {
		if status.State != operation.StateCompleted || status.Run == nil || len(status.Outputs) != 1 || len(status.Attempts) != 1 || runs[*status.Run] {
			t.Fatalf("incomplete or reused operation: %+v", status)
		}
		runs[*status.Run] = true
		run, err := runrecord.RequireExactRun(t.Context(), fixture.store, *status.Run)
		if err != nil || run.CodeCommit != transcriptionHTTPCommit || run.Recipe != fixture.workspace.policy.Recipe || run.Environment != fixture.workspace.environment {
			t.Fatalf("bound run=%+v err=%v", run, err)
		}
		transcription, err := speechrecognition.RequireTranscription(t.Context(), fixture.store, status.Outputs[0])
		if err != nil || transcription.Text != "x" {
			t.Fatalf("stored transcription=%+v err=%v", transcription, err)
		}
		content, found, err := artifact.ReadContent(t.Context(), fixture.store, status.Attempts[0])
		if err != nil || !found {
			t.Fatalf("observation content: found=%t err=%v", found, err)
		}
		observation, err := runrecord.ParseServingObservation(content.Data)
		if err != nil || observation.Model != fixture.workspace.component.Model || observation.Recipe != run.Recipe || observation.Run != run.ID {
			t.Fatalf("wrong serving model/run: %+v err=%v", observation, err)
		}
	}
	if snapshot := fixture.workspace.sessions.Snapshot(); snapshot.Loads != 1 || snapshot.Active != 0 {
		t.Fatalf("resident model not reused or lease leaked: %+v", snapshot)
	}
}

func testAudioTranscriptionsAuthentication(t *testing.T) {
	fixture := newTranscriptionHTTPFixture(t, nil)
	request := fixture.request(t, fixture.wave, nil)
	request.Header.Del("Authorization")
	response := httptest.NewRecorder()
	fixture.handler.ServeHTTP(response, request)
	id := testutil.ArtifactBytesID(t, artifact.KindFile, fixture.wave)
	_, found, err := fixture.store.Artifact(t.Context(), id)
	if response.Code != http.StatusUnauthorized || found || err != nil || len(fixture.handler.operations.List()) != 0 {
		t.Fatalf("unauthenticated side effect: status=%d found=%t err=%v", response.Code, found, err)
	}
}

func testAudioTranscriptionsInvalid(t *testing.T) {
	fixture := newTranscriptionHTTPFixture(t, nil)
	for _, name := range []string{"language", "prompt", "timestamp_granularities[]", "temperature"} {
		t.Run(name, func(t *testing.T) {
			response := httptest.NewRecorder()
			fixture.handler.ServeHTTP(response, fixture.request(t, fixture.wave, map[string]string{
				"model": fixture.workspace.policy.Recipe.String(), name: "unsupported",
			}))
			if response.Code != http.StatusBadRequest {
				t.Fatalf("unsupported control: %d %s", response.Code, response.Body)
			}
		})
	}
	for _, fields := range []map[string]string{
		{}, {"model": "unconfigured"},
		{"model": fixture.workspace.policy.Recipe.String(), "stream": "true"},
		{"model": fixture.workspace.policy.Recipe.String(), "response_format": "verbose_json"},
	} {
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, fixture.request(t, fixture.wave, fields))
		if response.Code != http.StatusBadRequest {
			t.Fatalf("invalid control: %d %s", response.Code, response.Body)
		}
	}
	if len(fixture.handler.operations.List()) != 0 {
		t.Fatal("invalid protocol controls reached execution")
	}
	for _, data := range [][]byte{nil, []byte("unsupported encoding"), fixture.wave[:len(fixture.wave)/2]} {
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, fixture.request(t, data, nil))
		if response.Code != http.StatusBadRequest {
			t.Fatalf("invalid audio: %d %s", response.Code, response.Body)
		}
	}
}

func testAudioTranscriptionsLimits(t *testing.T) {
	fixture := newTranscriptionHTTPFixture(t, func(policy *TranscriptionPolicy) { policy.Inspection.MaximumSamples = 1 })
	for _, test := range []struct {
		data []byte
		code int
	}{
		{fixture.wave, http.StatusBadRequest},
		{make([]byte, int(fixture.workspace.policy.Inspection.MaximumEncodedBytes)+1), http.StatusRequestEntityTooLarge},
		{make([]byte, maxMediaBytes+1), http.StatusRequestEntityTooLarge},
	} {
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, fixture.request(t, test.data, nil))
		if response.Code != test.code {
			t.Fatalf("limit status=%d want=%d body=%s", response.Code, test.code, response.Body)
		}
	}
}

type observedTranscriptionWorkspace struct {
	*TranscriptionWorkspace
	entered chan artifact.ID
}

func (workspace observedTranscriptionWorkspace) ExecuteWorkflow(ctx context.Context, kind WorkflowKind, task recipe.Task, recipeID artifact.ID, raw json.RawMessage, reporter operation.Reporter) (operation.Completion, error) {
	workspace.entered <- reporter.OperationID()
	return workspace.TranscriptionWorkspace.ExecuteWorkflow(ctx, kind, task, recipeID, raw, reporter)
}

func testAudioTranscriptionsCancellation(t *testing.T) {
	fixture := newTranscriptionHTTPFixture(t, nil)
	lease, err := fixture.workspace.sessions.LeaseComponent(t.Context(), fixture.workspace.component, fixture.workspace.load)
	if err != nil {
		t.Fatal(err)
	}
	defer lease.Release()
	entered := make(chan artifact.ID, 1)
	fixture.handler.generator.(*transcriptionHTTPRuntime).WorkflowWorkspaceAPI = observedTranscriptionWorkspace{fixture.workspace, entered}
	ctx, cancel := context.WithCancelCause(t.Context())
	defer cancel(nil)
	request := fixture.request(t, fixture.wave, nil).WithContext(ctx)
	response := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		fixture.handler.ServeHTTP(response, request)
		close(done)
	}()
	var id artifact.ID
	select {
	case id = <-entered:
	case <-t.Context().Done():
		t.Fatal("transcription operation never started")
	}
	cancel(context.Canceled)
	<-done
	status, err := fixture.handler.operations.Wait(t.Context(), id)
	if err != nil || status.State != operation.StateCancelled || status.Run == nil || response.Code == http.StatusOK {
		t.Fatalf("cancelled operation=%+v HTTP=%d err=%v", status, response.Code, err)
	}
	run, err := runrecord.RequireExactRun(t.Context(), fixture.store, *status.Run)
	if err != nil || run.Outcome != runrecord.OutcomeCancelled {
		t.Fatalf("cancelled run=%+v err=%v", run, err)
	}
	if err := lease.Release(); err != nil {
		t.Fatal(err)
	}
	if snapshot := fixture.workspace.sessions.Snapshot(); snapshot.Active != 0 || snapshot.Waiting != 0 {
		t.Fatalf("cancellation leaked admission: %+v", snapshot)
	}
}
