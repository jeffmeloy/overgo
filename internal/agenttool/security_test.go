package agenttool

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// TestTransportSecurityRefusesPrivateEndpoints pins the serving
// policy: the serving executor never reaches loopback or private
// addresses on a manual's behalf, while the operator executor may.
func TestTransportSecurityRefusesPrivateEndpoints(t *testing.T) {
	ctx := context.Background()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte(`{"reached":true}`))
	}))
	defer server.Close()
	manual := inspectionManual(t, "web.local", Transport{Kind: TransportHTTP, URL: server.URL})
	arguments := json.RawMessage(`{"pattern":"x"}`)
	if _, err := NewExecutor().Invoke(ctx, manual, arguments); err == nil ||
		!strings.Contains(err.Error(), "private") {
		t.Fatalf("serving executor reached a loopback endpoint: %v", err)
	}
	result, err := NewOperatorExecutor().Invoke(ctx, manual, arguments)
	if err != nil || string(result) != `{"reached":true}` {
		t.Fatalf("operator executor refused loopback: %s, %v", result, err)
	}
}

// TestTransportSecurityRefusesRedirects pins that both executors
// refuse a steering endpoint: a manual binds one endpoint, and a
// redirect is a different endpoint the manual never declared.
func TestTransportSecurityRefusesRedirects(t *testing.T) {
	ctx := context.Background()
	mux := http.NewServeMux()
	mux.HandleFunc("/steer", func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "/elsewhere", http.StatusFound)
	})
	mux.HandleFunc("/elsewhere", func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte(`{"stolen":true}`))
	})
	server := httptest.NewServer(mux)
	defer server.Close()
	manual := inspectionManual(t, "web.steer", Transport{Kind: TransportHTTP, URL: server.URL + "/steer"})
	if _, err := NewOperatorExecutor().Invoke(ctx, manual, json.RawMessage(`{"pattern":"x"}`)); err == nil ||
		!strings.Contains(err.Error(), "redirected") {
		t.Fatalf("redirect followed: %v", err)
	}
}

// TestManualSecurityArgvProgramIsBareWord pins publication policy: an
// argv manual cannot name a path, only a bare command word the
// operator's allowlist can reason about.
func TestManualSecurityArgvProgramIsBareWord(t *testing.T) {
	for _, program := range []string{`C:\evil.exe`, "./local", "bin/tool", `..\up`} {
		manual := validManual()
		manual.Name = "argv.pathy"
		manual.Transport = Transport{Kind: TransportArgv, Program: program}
		if _, err := NewManual(manual); err == nil {
			t.Fatalf("path program %q accepted", program)
		}
	}
	bare := validManual()
	bare.Name = "argv.bare"
	bare.Transport = Transport{Kind: TransportArgv, Program: "git", Args: []string{"status"}}
	if _, err := NewManual(bare); err != nil {
		t.Fatalf("bare command word refused: %v", err)
	}
}
