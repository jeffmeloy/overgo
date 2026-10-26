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
	lines := []string{`{"token":"first"}`, `{"token":"second"}`}
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.Header().Set("Content-Type", HTTPJSONStreamMediaType)
		for _, line := range lines {
			fmt.Fprintln(response, line)
		}
	}))
	defer server.Close()
	manual := streamManual(t, server.URL)
	// The two-event stream completes under a two-event policy and fails at its admission bound under a one-event policy.
	for _, row := range []struct {
		maxEvents uint32
		state     StreamState
		failure   string
	}{{2, StreamCompleted, ""}, {1, StreamFailed, "stream admission bound exceeded at event 2"}} {
		stream, err := operatorStreamAdapter().openStream(t.Context(), manual, json.RawMessage(`{"pattern":"x"}`), StreamPolicy{
			MaxEvents: row.maxEvents, MaxEventBytes: 256, MaxBytes: 512,
		})
		if err != nil {
			t.Fatal(err)
		}
		defer stream.Close()
		var events []StreamEvent
		for range row.maxEvents {
			event, err := stream.Receive(t.Context())
			if err != nil {
				t.Fatal(err)
			}
			events = append(events, event)
		}
		if _, err := stream.Receive(t.Context()); !errors.Is(err, io.EOF) {
			t.Fatalf("end error = %v", err)
		}
		// Each event is read after the scanner has taken every later line, so a value sharing its buffer would show here.
		for index, event := range events {
			if event.Sequence != uint64(index+1) || event.Invocation.Kind() != artifact.KindRun || string(event.Value) != lines[index] {
				t.Fatalf("event = %+v", event)
			}
			if _, err := event.ArtifactContent(); err != nil {
				t.Fatal(err)
			}
		}
		terminal, err := stream.Terminal()
		if err != nil {
			t.Fatal(err)
		}
		if terminal.State != row.state || terminal.Events != row.maxEvents || terminal.Bytes == 0 || !strings.Contains(terminal.Error, row.failure) {
			t.Fatalf("terminal = %+v", terminal)
		}
		if _, err := terminal.ArtifactContent(); err != nil {
			t.Fatal(err)
		}
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
