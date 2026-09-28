package main

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

// TestCompatibilityReadsNoCommandInventory holds this package's inputs away
// from what a command or route change moves: the API manifest the gate
// regenerates on every such change, and the web client's tree. go test
// reruns a package when a file it opens changes, and this suite takes
// minutes, so a document it reads must be one it checks. The names are
// split so this file does not name what it looks for.
func TestCompatibilityReadsNoCommandInventory(t *testing.T) {
	t.Parallel()
	sources, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range sources {
		data, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		for _, input := range []string{"docs/" + "api_manifest.json", "internal/server/" + "webui"} {
			if bytes.Contains(data, []byte(input)) {
				t.Errorf("%s reads %s, which a command or route change moves; read the narrower source its check needs", name, input)
			}
		}
	}
}
