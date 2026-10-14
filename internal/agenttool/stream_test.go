package agenttool

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

func TestHTTPJSONStreamProducesSequencedEventsAndTerminal(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.Header().Set("Content-Type", HTTPJSONStreamMediaType)
		fmt.Fprintln(response, `{"token":"first"}`)
		fmt.Fprintln(response, `{"token":"second"}`)
	}))
	defer server.Close()
	manual := streamManual(t, server.URL)
	stream, err := operatorStreamAdapter().openStream(t.Context(), manual, json.RawMessage(`{"pattern":"x"}`), StreamPolicy{
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
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.Header().Set("Content-Type", HTTPJSONStreamMediaType)
		fmt.Fprintln(response, `{"value":1}`)
		fmt.Fprintln(response, `{"value":2}`)
	}))
	defer server.Close()
	stream, err := operatorStreamAdapter().openStream(t.Context(), streamManual(t, server.URL), json.RawMessage(`{"pattern":"x"}`), StreamPolicy{
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
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		response.Header().Set("Content-Type", HTTPJSONStreamMediaType)
		response.(http.Flusher).Flush()
		<-request.Context().Done()
	}))
	defer server.Close()
	stream, err := operatorStreamAdapter().openStream(t.Context(), streamManual(t, server.URL), json.RawMessage(`{"pattern":"x"}`), StreamPolicy{
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

// TestHTTPJSONStreamManualInvokesCollected holds a registered manual of the streaming transport to one result per call.
//   - a completed stream returns its events as one JSON array in sequence order
//   - the output observer sees each event as one line as it arrives
//   - a stream that breaks, carries a line that is not JSON, passes the output limit or is cancelled is refused with what went wrong
//   - a refused stream returns no result and no effect
func TestHTTPJSONStreamManualInvokesCollected(t *testing.T) {
	t.Parallel()
	const (
		// eventBytes is one line of the oversized stream with its newline.
		eventBytes = 1 << 20
		// oversizedEvents is the fewest such lines whose events pass artifact.MaxContentBytes.
		oversizedEvents = artifact.MaxContentBytes/eventBytes + 1
	)
	line := `"` + strings.Repeat("x", eventBytes-len("\"\"\n")) + "\"\n"
	stopped := errors.New("the caller left the stream")
	rows := []struct {
		name     string
		write    func(http.ResponseWriter, *http.Request)
		result   string
		observed []string
		refusal  string
	}{
		{"stream.completed", func(response http.ResponseWriter, _ *http.Request) {
			fmt.Fprint(response, "{\"token\":\"a\"}\n\n {\"token\":\"b\"} \n")
		}, `[{"token":"a"},{"token":"b"}]`, []string{"{\"token\":\"a\"}\n", "{\"token\":\"b\"}\n"}, ""},
		{"stream.broken", func(response http.ResponseWriter, _ *http.Request) {
			fmt.Fprintln(response, `{"token":"a"}`)
			response.(http.Flusher).Flush()
			panic(http.ErrAbortHandler)
		}, "", nil, "stream failed after 1 events"},
		{"stream.not-json", func(response http.ResponseWriter, _ *http.Request) {
			fmt.Fprint(response, "{\"token\":\"a\"}\n{\"token\":\n")
		}, "", nil, "invalid stream JSON"},
		{"stream.oversized", func(response http.ResponseWriter, _ *http.Request) {
			for range oversizedEvents {
				if _, err := io.WriteString(response, line); err != nil {
					return
				}
			}
		}, "", nil, fmt.Sprintf("stream admission bound exceeded at event %d", oversizedEvents)},
		{"stream.cancelled", func(response http.ResponseWriter, request *http.Request) {
			fmt.Fprintln(response, `{"token":"a"}`)
			response.(http.Flusher).Flush()
			<-request.Context().Done()
		}, "", nil, "stream cancelled after 1 events"},
	}
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	manuals := make([]Manual, len(rows))
	for index, row := range rows {
		server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
			response.Header().Set("Content-Type", HTTPJSONStreamMediaType)
			row.write(response, request)
		}))
		defer server.Close()
		manuals[index] = inspectionManual(t, row.name, Transport{Kind: TransportHTTPJSONStream, URL: server.URL})
	}
	if _, err := PublishManualCatalog(ctx, store, manuals); err != nil {
		t.Fatal(err)
	}
	executor := NewOperatorExecutor()
	for _, row := range rows {
		manual, err := ResolveRegisteredManual(ctx, store, row.name)
		if err != nil {
			t.Fatal(err)
		}
		// The cancelled row's caller leaves once its first event arrived, while the server holds the stream open.
		invocation, cancel := context.WithCancelCause(ctx)
		var observed []string
		invocation = WithOutputObserver(invocation, func(piece []byte) {
			observed = append(observed, string(piece))
			if row.name == "stream.cancelled" {
				cancel(stopped)
			}
		})
		result, effect, err := executor.InvokeWithEffect(invocation, manual, json.RawMessage(`{"pattern":"x"}`))
		cancel(nil)
		if row.refusal == "" {
			if err != nil || string(result) != row.result || effect.ObservedResult == nil || !slices.Equal(observed, row.observed) {
				t.Errorf("%s: result = %s, effect = %+v, observed = %q, err = %v, want %s observed as %q", row.name, result, effect, observed, err, row.result, row.observed)
			}
			continue
		}
		if err == nil || !strings.Contains(err.Error(), row.refusal) || result != nil || effect.ObservedResult != nil {
			t.Errorf("%s: result of %d bytes, effect = %+v, err = %v, want a refusal naming %q", row.name, len(result), effect, err, row.refusal)
		}
		if row.name == "stream.cancelled" && !errors.Is(err, stopped) {
			t.Errorf("%s: err = %v, want the caller's cause", row.name, err)
		}
	}
}

// operatorStreamAdapter returns the stream adapter an operator's executor holds, which reaches a loopback test server.
func operatorStreamAdapter() *httpJSONStreamAdapter {
	return NewOperatorExecutor().adapters[TransportHTTPJSONStream].(*httpJSONStreamAdapter)
}

func streamManual(t *testing.T, endpoint string) Manual {
	t.Helper()
	return inspectionManual(t, "tokens.stream", Transport{Kind: TransportHTTPJSONStream, URL: endpoint})
}
