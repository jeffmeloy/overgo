package agenttool

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"overgo/internal/artifact"
)

func TestHTTPJSONStreamProducesSequencedEventsAndTerminal(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.Header().Set("Content-Type", HTTPJSONStreamMediaType)
		fmt.Fprintln(response, `{"token":"first"}`)
		fmt.Fprintln(response, `{"token":"second"}`)
	}))
	defer server.Close()
	manual := streamManual(t, server.URL)
	stream, err := NewOperatorExecutor().OpenStream(t.Context(), manual, json.RawMessage(`{"pattern":"x"}`), StreamPolicy{
		MaxEvents: 2, MaxEventBytes: 256, MaxBytes: 512,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	for sequence := uint64(1); sequence <= 2; sequence++ {
		event, err := stream.Receive(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		if event.Sequence != sequence || event.Invocation.Kind() != artifact.KindRun {
			t.Fatalf("event = %+v", event)
		}
		if _, err := event.ArtifactContent(); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := stream.Receive(t.Context()); !errors.Is(err, io.EOF) {
		t.Fatalf("end error = %v", err)
	}
	terminal, err := stream.Terminal()
	if err != nil {
		t.Fatal(err)
	}
	if terminal.State != StreamCompleted || terminal.Events != 2 || terminal.Bytes == 0 {
		t.Fatalf("terminal = %+v", terminal)
	}
	if _, err := terminal.ArtifactContent(); err != nil {
		t.Fatal(err)
	}
}

func TestHTTPJSONStreamFailsAtAdmissionBound(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.Header().Set("Content-Type", HTTPJSONStreamMediaType)
		fmt.Fprintln(response, `{"value":1}`)
		fmt.Fprintln(response, `{"value":2}`)
	}))
	defer server.Close()
	stream, err := NewOperatorExecutor().OpenStream(t.Context(), streamManual(t, server.URL), json.RawMessage(`{"pattern":"x"}`), StreamPolicy{
		MaxEvents: 1, MaxEventBytes: 128, MaxBytes: 128,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	if _, err := stream.Receive(t.Context()); err != nil {
		t.Fatal(err)
	}
	if _, err := stream.Receive(t.Context()); !errors.Is(err, io.EOF) {
		t.Fatalf("end error = %v", err)
	}
	terminal, err := stream.Terminal()
	if err != nil {
		t.Fatal(err)
	}
	if terminal.State != StreamFailed || terminal.Events != 1 || terminal.Error == "" {
		t.Fatalf("terminal = %+v", terminal)
	}
}

func TestHTTPJSONStreamCancellationProducesTerminal(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		response.Header().Set("Content-Type", HTTPJSONStreamMediaType)
		response.(http.Flusher).Flush()
		<-request.Context().Done()
	}))
	defer server.Close()
	stream, err := NewOperatorExecutor().OpenStream(t.Context(), streamManual(t, server.URL), json.RawMessage(`{"pattern":"x"}`), StreamPolicy{
		MaxEvents: 1, MaxEventBytes: 128, MaxBytes: 128,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := stream.Close(); err != nil {
		t.Fatal(err)
	}
	terminal, err := stream.Terminal()
	if err != nil {
		t.Fatal(err)
	}
	if terminal.State != StreamCancelled || terminal.Events != 0 {
		t.Fatalf("terminal = %+v", terminal)
	}
}

func streamManual(t *testing.T, endpoint string) Manual {
	t.Helper()
	return inspectionManual(t, "tokens.stream", Transport{Kind: TransportHTTPJSONStream, URL: endpoint})
}
