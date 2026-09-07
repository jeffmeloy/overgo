package server

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"
)

func TestWorkspaceSchema(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	response := serveTestRequest(handler, http.MethodGet, "/workspace/schema?id=automation-trigger", "")
	if response.Code != http.StatusOK {
		t.Fatalf("workspace schema status=%d body=%s", response.Code, response.Body.String())
	}
	var schema workspaceObjectSchema
	if err := json.Unmarshal(response.Body.Bytes(), &schema); err != nil {
		t.Fatal(err)
	}
	if schema.ID != "automation-trigger" || len(schema.Fields) == 0 {
		t.Fatalf("workspace schema = %+v", schema)
	}
	byName := make(map[string]workspaceFieldSchema, len(schema.Fields))
	for _, field := range schema.Fields {
		byName[field.Name] = field
	}
	if byName["kind"].Type != "enum" || len(byName["kind"].Options) != 2 ||
		byName["schedule"].Type != "duration" || byName["schedule"].Unit == "" ||
		byName["schedule"].When == nil || byName["anchor_unix_nano"].Minimum == nil {
		t.Fatalf("typed workspace fields = %+v", byName)
	}
	missing := serveTestRequest(handler, http.MethodGet, "/workspace/schema?id=absent", "")
	if missing.Code != http.StatusNotFound {
		t.Fatalf("missing schema status = %d", missing.Code)
	}
}

func TestWorkspaceFormRenderer(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	renderer := serveTestRequest(handler, http.MethodGet, "/schema_form.js", "").Body.String()
	for _, expected := range []string{
		"window.overgo.schemaForm", "field.identity_kind", "field.options", "field.unit",
		"field.minimum", "field.maximum", "applicable(field)", "root.checkValidity()", "minimum_items",
	} {
		if !strings.Contains(renderer, expected) {
			t.Errorf("schema form renderer lacks %q", expected)
		}
	}
	app := serveTestRequest(handler, http.MethodGet, "/boot.js", "").Body.String()
	if !strings.Contains(app, "/schema_form.js") {
		t.Fatal("workspace shell does not load schema form renderer")
	}
}

func TestWorkspaceUnsavedChanges(t *testing.T) {
	renderer := serveTestRequest(newTestHandler(t, &fakeGenerator{}), http.MethodGet, "/schema_form.js", "").Body.String()
	for _, expected := range []string{
		"function dirty()", "beforeunload", "event.preventDefault()", "event.returnValue", "markSaved", "dispose()",
	} {
		if !strings.Contains(renderer, expected) {
			t.Errorf("unsaved-change protection lacks %q", expected)
		}
	}
}
