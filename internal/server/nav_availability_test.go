package server

import (
	"encoding/json"
	"net/http"
	"regexp"
	"testing"
)

// developerWords are what a refusal must not show a user: a launch flag, a
// command, a package path or a capability identifier.
var developerWords = regexp.MustCompile(`(^|\s)-[a-z]|go run|cmd/|\./|workflow\.|capability|unavailable`)

// TestNavAvailability: on a plain launch every refused workspace and
// composer mode says in plain words what is missing and how to turn it on,
// naming no flag, command or capability identifier, and an open workspace
// carries no refusal.
func TestNavAvailability(t *testing.T) {
	t.Parallel()
	handler := newTestHandler(t, &fakeGenerator{})
	var manifest workspaceManifestResponse
	if err := json.Unmarshal(serveTestRequest(handler, http.MethodGet, "/workspace/manifest", "").Body.Bytes(), &manifest); err != nil {
		t.Fatal(err)
	}
	refused := 0
	for _, tab := range manifest.Tabs {
		switch {
		case tab.Enabled && (tab.Refusal != "" || tab.Action != ""):
			t.Errorf("open workspace %s carries a refusal %q %q", tab.ID, tab.Refusal, tab.Action)
		case !tab.Enabled && (tab.Refusal == "" || tab.Action == ""):
			t.Errorf("refused workspace %s says %q and %q", tab.ID, tab.Refusal, tab.Action)
		case !tab.Enabled && (developerWords.MatchString(tab.Refusal) || developerWords.MatchString(tab.Action)):
			t.Errorf("refused workspace %s speaks to developers: %q %q", tab.ID, tab.Refusal, tab.Action)
		}
		if !tab.Enabled {
			refused++
		}
	}
	if refused == 0 {
		t.Fatal("a plain launch refused no workspace; the test sees no refusal to judge")
	}
	if manifest.Model == nil {
		t.Fatal("the manifest carries no model document")
	}
	for _, mode := range manifest.Model.Modes {
		if !mode.Enabled && (mode.Refusal == "" || developerWords.MatchString(mode.Refusal)) {
			t.Errorf("refused mode %s says %q", mode.ID, mode.Refusal)
		}
	}
}
