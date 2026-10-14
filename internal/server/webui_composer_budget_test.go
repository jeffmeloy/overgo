package server

import (
	"io/fs"
	"strings"
	"testing"
)

func webuiJavaScript(t *testing.T) map[string]string {
	t.Helper()
	sources := map[string]string{}
	if err := fs.WalkDir(webuiFS, ".", func(name string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !entry.IsDir() && strings.HasSuffix(name, ".js") {
			data, err := fs.ReadFile(webuiFS, name)
			if err != nil {
				return err
			}
			sources[name] = string(data)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	return sources
}

// TestWebUIComposerBudget holds the client's one stream renderer to the
// event vocabulary this package declares (stream_events.go): every event
// the server can send has a case in composer.js.
func TestWebUIComposerBudget(t *testing.T) {
	t.Parallel()
	composer, ok := webuiJavaScript(t)["composer.js"]
	if !ok {
		t.Fatal("composer.js is absent")
	}
	for _, event := range streamEventTypes {
		if !strings.Contains(composer, `case "`+event+`":`) {
			t.Errorf("composer.js does not render stream event %q", event)
		}
	}
}
