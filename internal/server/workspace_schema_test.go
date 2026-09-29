package server

import (
	"encoding/json"
	"net/http"
	"testing"
)

func TestWorkspaceSchema(t *testing.T) {
	t.Parallel()
	handler := newTestHandler(t, &fakeGenerator{})
	response := serveTestRequest(handler, http.MethodGet, "/workspace/schema?id=agent-definition", "")
	if response.Code != http.StatusOK {
		t.Fatalf("workspace schema status=%d body=%s", response.Code, response.Body.String())
	}
	var schema workspaceObjectSchema
	if err := json.Unmarshal(response.Body.Bytes(), &schema); err != nil {
		t.Fatal(err)
	}
	if schema.ID != "agent-definition" || len(schema.Fields) == 0 {
		t.Fatalf("workspace schema = %+v", schema)
	}
	byName := make(map[string]workspaceFieldSchema, len(schema.Fields))
	for _, field := range schema.Fields {
		byName[field.Name] = field
	}
	if byName["name"].Type != "text" || byName["name"].Pattern == "" ||
		byName["prompt"].Type != "identity" || byName["prompt"].IdentityKind == "" ||
		byName["policies"].Type != "string-list" || byName["policies"].MinimumItems == nil {
		t.Fatalf("typed workspace fields = %+v", byName)
	}
	missing := serveTestRequest(handler, http.MethodGet, "/workspace/schema?id=absent", "")
	if missing.Code != http.StatusNotFound {
		t.Fatalf("missing schema status = %d", missing.Code)
	}
}
