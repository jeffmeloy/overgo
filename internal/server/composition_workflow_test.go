package server

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/testcheck"
)

func TestCompositionAPIWorkflow(t *testing.T) {
	t.Parallel()
	store := testcheck.Must(overgodb.Open(t.TempDir()))(t)
	handler := testcheck.Must(New(Config{Repository: store}, &fakeGenerator{}))(t)
	t.Cleanup(func() {
		_ = handler.Close()
		_ = store.Close()
	})

	response := serveTestRequest(handler, http.MethodGet, "/compositions", "")
	if response.Code != http.StatusOK {
		t.Fatalf("inventory status = %d body=%s", response.Code, response.Body.String())
	}
	var inventory compositionInventoryResponse
	testcheck.NoError(t, json.Unmarshal(response.Body.Bytes(), &inventory))
	if len(inventory.Compositions) != 0 {
		t.Fatalf("inventory = %+v", inventory)
	}

	invalid := serveTestRequest(handler, http.MethodPost, "/compositions/activate", `{}`)
	if invalid.Code != http.StatusBadRequest {
		t.Fatalf("invalid activation status = %d body=%s", invalid.Code, invalid.Body.String())
	}
}

func TestCompositeGenerationUnpromotedRefusal(t *testing.T) {
	t.Parallel()
	store := testcheck.Must(overgodb.Open(t.TempDir()))(t)
	handler := testcheck.Must(New(Config{Repository: store}, &fakeGenerator{}))(t)
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
