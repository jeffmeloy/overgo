package server

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/speechrecognition"
	"overgo/internal/speechrecognitiontest"
	"overgo/internal/testevidence"
	"overgo/internal/workflowruntime"
)

func TestStreamingTranscriptionAcceptance(t *testing.T) {
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip + ": authenticated native streaming requires registered waveform and model captures")
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })
	native := speechrecognitiontest.PublishNative(t, store)
	verification, err := modelrecipetest.PublishVerification(t.Context(), store, "streaming/fixture", native.Definition.ID)
	if err != nil {
		t.Fatal(err)
	}
	if err := modelrecipe.ActivateCapability(t.Context(), store, native.Definition, verification, recipe.EvidenceParity, "isolated native HTTP acceptance; not production promotion"); err != nil {
		t.Fatal(err)
	}
	clip := native.Clips[0]
	wave := speechrecognitiontest.NativeWave(t, clip.PCM, native.Profile.Frontend.SampleRate)
	policy := TranscriptionPolicy{Recipe: native.Definition.ID, MemoryBytes: 4 << 30,
		Inspection: dataset.AudioInspectionPolicy{MaximumEncodedBytes: uint64(len(wave)), MaximumSamples: uint64(len(clip.PCM)), ClipThreshold: .999,
			Admission: recipecontract.AudioAdmissionPolicy{MinimumChannels: 1, MaximumChannels: 1, MaximumAbsoluteDCOffset: .1}}}
	workspace, err := NewTranscriptionWorkspace(t.Context(), store, policy, transcriptionHTTPCommit)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { workspace.Close(context.WithoutCancel(t.Context())) })
	runtime := &transcriptionHTTPRuntime{fakeGenerator: &fakeGenerator{}, WorkflowWorkspaceAPI: WorkflowWorkspaceSet{workspace}}
	handler, err := New(Config{Repository: store, APIKey: testAPIKey}, runtime)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { handler.Close() })
	server := httptest.NewUnstartedServer(handler)
	server.Config.ErrorLog = log.New(transcriptionHTTPLog{t}, "", 0)
	server.Start()
	t.Cleanup(server.Close)
	source, chunks := native.PublishStream(t, store, clip, native.Profile.Frontend.SampleRate+1)
	t.Run("authentication-before-input", func(t *testing.T) {
		request, err := http.NewRequestWithContext(t.Context(), http.MethodPost, server.URL+"/v1/audio/transcriptions", strings.NewReader("invalid input"))
		if err != nil {
			t.Fatal(err)
		}
		request.Header.Set("Content-Type", "application/x-ndjson")
		response, err := server.Client().Do(request)
		if err != nil {
			t.Fatal(err)
		}
		defer response.Body.Close()
		if response.StatusCode != http.StatusUnauthorized || workspace.sessions.Snapshot().Active != 0 {
			t.Fatal("unauthenticated stream was admitted")
		}
	})
	initial := transcriptionStreamRequest{Model: native.Definition.ID.String(), StreamRequest: speechrecognition.StreamRequest{Source: source}}
	client := openTranscriptionClient(t, server, initial)
	started := time.Now()
	var text strings.Builder
	var partial, final time.Duration
	for i, chunk := range chunks {
		name, data := client.send(t, chunk)
		var event transcriptionStreamEvent
		if err := json.Unmarshal(data, &event); err != nil {
			t.Fatal(err)
		}
		if event.Transcript.Sequence != chunk.Sequence || event.Transcript.Source != source || event.Transcript.Final != chunk.Final || !event.Output.Valid() || !event.Checkpoint.Valid() || !event.Admission.Valid() {
			t.Fatalf("invalid HTTP event %s: %s", name, data)
		}
		if !chunk.Final && name != streamEventToken || chunk.Final && name != streamEventDone {
			t.Fatalf("unexpected event type %s", name)
		}
		text.WriteString(event.Transcript.Text)
		if text.Len() != 0 && partial == 0 {
			partial = time.Since(started)
		}
		if chunk.Final {
			final = time.Since(started)
		}
		if i == 0 {
			client.close()
			// Acquiring the existing component waits for disconnect cleanup;
			// no polling delay or parallel serving owner is introduced.
			lease, err := workspace.sessions.Lease(t.Context())
			if err != nil {
				t.Fatal(err)
			}
			if err := lease.Release(); err != nil {
				t.Fatal(err)
			}
			initial.Resume = event.Checkpoint
			client = openTranscriptionClient(t, server, initial)
		}
	}
	client.close()
	if text.String() != clip.Streaming || partial == 0 || final < partial {
		t.Fatalf("native HTTP transcript=%q want=%q partial=%s final=%s", text.String(), clip.Streaming, partial, final)
	}
	lease, err := workspace.sessions.Lease(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if err := lease.Release(); err != nil {
		t.Fatal(err)
	}
	t.Run("out-of-order-refusal", func(t *testing.T) {
		initial.Resume = artifact.ID{}
		client := openTranscriptionClient(t, server, initial)
		defer client.close()
		name, _ := client.send(t, chunks[1])
		if name != streamEventError {
			t.Fatal("HTTP stream accepted out-of-order input")
		}
	})
	t.Logf("native authenticated HTTP: %d waveform chunks; first text=%s final event=%s end-to-end RTF=%.3f; one disconnect/restart; exact native transcript and durable output/state/admission IDs; CPU only, no word timestamps", len(chunks), partial, final, final.Seconds()/(float64(len(clip.PCM))/float64(native.Profile.Frontend.SampleRate)))
}

type transcriptionClient struct {
	writer  *io.PipeWriter
	encoder *json.Encoder
	reader  *bufio.Scanner
	body    io.ReadCloser
	cancel  context.CancelCauseFunc
}

type transcriptionHTTPLog struct{ t *testing.T }

func (writer transcriptionHTTPLog) Write(data []byte) (int, error) {
	writer.t.Errorf("unexpected HTTP server error: %s", data)
	return len(data), nil
}

func openTranscriptionClient(t *testing.T, server *httptest.Server, initial transcriptionStreamRequest) *transcriptionClient {
	t.Helper()
	ctx, cancel := context.WithCancelCause(t.Context())
	reader, writer := io.Pipe()
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, server.URL+"/v1/audio/transcriptions", reader)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Content-Type", "application/x-ndjson")
	request.Header.Set("Authorization", "Bearer "+testAPIKey)
	type received struct {
		response *http.Response
		err      error
	}
	ready := make(chan received, 1)
	go func() {
		response, err := server.Client().Do(request)
		ready <- received{response, err}
	}()
	client := &transcriptionClient{writer: writer, encoder: json.NewEncoder(writer), cancel: cancel}
	t.Cleanup(client.close)
	if err := client.encoder.Encode(initial); err != nil {
		t.Fatal(err)
	}
	result := <-ready
	if result.err != nil {
		t.Fatal(result.err)
	}
	client.body = result.response.Body
	if result.response.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(client.body)
		t.Fatalf("stream HTTP=%d: %s", result.response.StatusCode, body)
	}
	client.reader = bufio.NewScanner(client.body)
	client.reader.Buffer(nil, maxRequestBytes)
	if name, _ := client.event(t); name != streamEventCreated {
		t.Fatal("stream did not acknowledge source before requesting chunks")
	}
	return client
}

func (client *transcriptionClient) event(t *testing.T) (string, []byte) {
	t.Helper()
	var name string
	for client.reader.Scan() {
		line := client.reader.Text()
		if value, ok := strings.CutPrefix(line, "event: "); ok {
			name = value
		}
		if value, ok := strings.CutPrefix(line, "data: "); ok {
			return name, []byte(value)
		}
	}
	t.Fatalf("stream ended without event: %v", client.reader.Err())
	return "", nil
}

func (client *transcriptionClient) send(t *testing.T, chunk workflowruntime.AudioStreamChunk) (string, []byte) {
	t.Helper()
	if err := client.encoder.Encode(chunk); err != nil {
		t.Fatal(err)
	}
	return client.event(t)
}

func (client *transcriptionClient) close() {
	client.cancel(context.Canceled)
	client.writer.Close()
	if client.body != nil {
		client.body.Close()
	}
}
