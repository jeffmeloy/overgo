package server

import (
	"net/http"
	"strings"
	"testing"
)

// TestFrontPageMediaRoundtrip pins media in and out as one loop
// (professional GUI campaign, gui-media-roundtrip): every media output
// the thread renders offers itself back as input, its bytes fetched from
// the artifact the store holds through the API client and re-entering the
// composer as a file of its own kind, where the served capability accepts
// or refuses it like any attachment; the chat page wires the hook.
func TestFrontPageMediaRoundtrip(t *testing.T) {
	handler := newTestHandlerWithRepository(t, responseRecipeGenerator(t, &fakeGenerator{}))
	defer handler.Close()
	get := func(path string) string { return serveTestRequest(handler, http.MethodGet, path, "").Body.String() }

	boot := get("/boot.js")
	if !strings.Contains(boot, "async blob(path, opts)") || !strings.Contains(boot, `{ method: "GET" }`) {
		t.Error("the API client fetches no artifact bytes")
	}
	composer := get("/composer.js")
	for _, needle := range []string{`"use as input"`, "overgo.api.blob(event.url)", "options.reuse", "reuse(new File(", "function thread(host, options)"} {
		if !strings.Contains(composer, needle) {
			t.Errorf("composer missing %q", needle)
		}
	}
	chat := get("/mod/chat.js")
	// The hook takes the card's file and its stored id: a mode whose request
	// names an artifact takes the id, any other turn takes the file.
	for _, needle := range []string{"reuse: (file, artifact) =>", `field.control.type === "artifact"`, "field.input.value = artifact", "composer.addFile(file)"} {
		if !strings.Contains(chat, needle) {
			t.Errorf("the chat page does not take media outputs back as input: missing %q", needle)
		}
	}
}
