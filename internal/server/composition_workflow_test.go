package server

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"overgo/internal/repodb"
)

func TestCompositionAPIWorkflow(t *testing.T) {
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	handler, err := New(Config{Repository: store}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = handler.Close()
		_ = store.Close()
	})

	response := serveTestRequest(handler, http.MethodGet, "/compositions", "")
	if response.Code != http.StatusOK {
		t.Fatalf("inventory status = %d body=%s", response.Code, response.Body.String())
	}
	var inventory compositionInventoryResponse
	if err := json.Unmarshal(response.Body.Bytes(), &inventory); err != nil {
		t.Fatal(err)
	}
	if len(inventory.Compositions) != 0 {
		t.Fatalf("inventory = %+v", inventory)
	}

	invalid := serveTestRequest(handler, http.MethodPost, "/compositions/activate", `{}`)
	if invalid.Code != http.StatusBadRequest {
		t.Fatalf("invalid activation status = %d body=%s", invalid.Code, invalid.Body.String())
	}
}

func TestCompositionGUIWorkflow(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	module := serveTestRequest(handler, http.MethodGet, "/mod/compositions.js", "")
	if module.Code != http.StatusOK {
		t.Fatalf("composition module status = %d", module.Code)
	}
	for _, token := range []string{
		`api.get("/compositions")`, `api.post("/compositions/activate"`,
		"compatible", "recipe graph", "Bridge training controls and metrics",
		"Evaluation and promotion history", "Runtime memory and latency evidence",
		"contract-tested", "cuda-verified", "production-active",
	} {
		if !strings.Contains(module.Body.String(), token) {
			t.Errorf("composition GUI missing %q", token)
		}
	}
	shell := serveTestRequest(handler, http.MethodGet, "/app.html", "")
	if !strings.Contains(shell.Body.String(), "/mod/compositions.js") {
		t.Fatal("workbench shell does not load composition module")
	}
}
