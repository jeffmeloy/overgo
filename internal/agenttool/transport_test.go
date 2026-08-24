package agenttool

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"overgo/internal/artifact"
)

func inspectionManual(t *testing.T, name string, transport Transport) Manual {
	t.Helper()
	manual, err := NewManual(Manual{
		Name:        name,
		Description: "Transport test tool.",
		Effect:      EffectInspection,
		Arguments: []Field{
			{Name: "pattern", Kind: FieldString, Required: true},
			{Name: "limit", Kind: FieldInteger},
			{Name: "exact", Kind: FieldBoolean},
			{Name: "scope", Kind: FieldObject},
		},
		Transport: transport,
	})
	if err != nil {
		t.Fatal(err)
	}
	return manual
}

func TestTransportBuiltinRoundTripAndRefusals(t *testing.T) {
	ctx := context.Background()
	executor := NewExecutor()
	manual := inspectionManual(t, "echo.args", Transport{Kind: TransportBuiltin})
	if err := executor.registerBuiltin("echo.args", func(_ context.Context, arguments json.RawMessage) (json.RawMessage, error) {
		return arguments, nil
	}); err != nil {
		t.Fatal(err)
	}
	if err := executor.registerBuiltin("echo.args", nil); err == nil {
		t.Fatal("nil duplicate registration accepted")
	}
	result, err := executor.Invoke(ctx, manual, json.RawMessage(`{"pattern":"x","limit":3,"exact":true,"scope":{"dir":"internal"}}`))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(result), `"pattern"`) {
		t.Fatalf("result = %s", result)
	}
	unregistered := inspectionManual(t, "echo.other", Transport{Kind: TransportBuiltin})
	if _, err := executor.Invoke(ctx, unregistered, json.RawMessage(`{"pattern":"x"}`)); err == nil {
		t.Fatal("unregistered builtin invoked")
	}
}

func TestTransportArgumentValidation(t *testing.T) {
	ctx := context.Background()
	executor := NewExecutor()
	manual := inspectionManual(t, "echo.args", Transport{Kind: TransportBuiltin})
	if err := executor.registerBuiltin("echo.args", func(_ context.Context, arguments json.RawMessage) (json.RawMessage, error) {
		return arguments, nil
	}); err != nil {
		t.Fatal(err)
	}
	refusals := map[string]string{
		"missing required":   `{"limit":1}`,
		"undeclared key":     `{"pattern":"x","depth":2}`,
		"wrong string kind":  `{"pattern":7}`,
		"wrong integer kind": `{"pattern":"x","limit":1.5}`,
		"wrong boolean kind": `{"pattern":"x","exact":"yes"}`,
		"wrong object kind":  `{"pattern":"x","scope":[1]}`,
		"non-object payload": `["pattern"]`,
	}
	for name, payload := range refusals {
		if _, err := executor.Invoke(ctx, manual, json.RawMessage(payload)); err == nil {
			t.Fatalf("%s: invocation accepted", name)
		}
	}
}

func TestTransportHTTPBoundedStrictJSON(t *testing.T) {
	ctx := context.Background()
	executor := NewOperatorExecutor()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/ok":
			w.Write([]byte(`{"answer":42}`))
		case "/error":
			http.Error(w, "endpoint unavailable", http.StatusBadGateway)
		case "/text":
			w.Write([]byte("not json"))
		case "/flood":
			w.Write(make([]byte, artifact.MaxContentBytes+1))
		}
	}))
	defer server.Close()
	invoke := func(path string) (json.RawMessage, error) {
		manual := inspectionManual(t, "web.probe", Transport{Kind: TransportHTTP, URL: server.URL + path})
		return executor.Invoke(ctx, manual, json.RawMessage(`{"pattern":"x"}`))
	}
	result, err := invoke("/ok")
	if err != nil || string(result) != `{"answer":42}` {
		t.Fatalf("ok result = %s, %v", result, err)
	}
	for _, path := range []string{"/error", "/text", "/flood"} {
		if _, err := invoke(path); err == nil {
			t.Fatalf("%s: invocation accepted", path)
		}
	}
}

func TestTransportArgvReturnsStdoutAsJSONString(t *testing.T) {
	ctx := context.Background()
	executor := NewExecutor()
	manual := inspectionManual(t, "go.env", Transport{
		Kind: TransportArgv, Program: "go", Args: []string{"env", "GOOS"},
	})
	result, err := executor.Invoke(ctx, manual, json.RawMessage(`{"pattern":"x"}`))
	if err != nil {
		t.Fatal(err)
	}
	var text string
	if err := json.Unmarshal(result, &text); err != nil || text == "" {
		t.Fatalf("result = %s, %v", result, err)
	}
	failing := inspectionManual(t, "go.fail", Transport{
		Kind: TransportArgv, Program: "go", Args: []string{"env", "-no-such-flag"},
	})
	if _, err := executor.Invoke(ctx, failing, json.RawMessage(`{"pattern":"x"}`)); err == nil {
		t.Fatal("failing program accepted")
	}
}
