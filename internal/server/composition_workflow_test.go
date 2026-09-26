package server

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"overgo/internal/overgodb"
)

func TestCompositionAPIWorkflow(t *testing.T) {
	t.Parallel()
	store, err := overgodb.Open(t.TempDir())
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

// TestCompositionGUIWorkflow: the workbench shell declares the compositions
// tab; the compositions leg drives it over a published authority.
func TestCompositionGUIWorkflow(t *testing.T) {
	t.Parallel()
	handler := newTestHandler(t, &fakeGenerator{})
	shell := serveTestRequest(handler, http.MethodGet, "/workspace/manifest", "")
	if !strings.Contains(shell.Body.String(), `"id":"compositions"`) {
		t.Fatal("workbench shell does not load composition module")
	}
}

func TestCompositeGenerationUnpromotedRefusal(t *testing.T) {
	t.Parallel()
	store, err := overgodb.Open(t.TempDir())
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
	response := serveTestRequest(handler, http.MethodPost, "/compositions/generate", `{
		"source":"model:sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		"target":"model:sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
		"task":"generation"
	}`)
	if response.Code != http.StatusBadRequest || !strings.Contains(response.Body.String(), "not evidence-promoted") {
		t.Fatalf("unpromoted response status=%d body=%s", response.Code, response.Body.String())
	}
}
