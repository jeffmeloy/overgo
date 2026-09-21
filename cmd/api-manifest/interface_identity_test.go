package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// TestManifestChangesOnlyWithTheInterface holds the published manifest to the
// release interface and nothing else. It once carried a hash of the whole
// source tree, so every landing rewrote all of it and a reviewer could not
// tell a changed route from a touched file. The manifest names no identity of
// the tree; each document still names the identity of the one source file that
// declares it, which moves only when that file does; and the file in the tree
// is what the source compiles to now.
func TestManifestChangesOnlyWithTheInterface(t *testing.T) {
	t.Parallel()
	root := filepath.Join("..", "..")
	manifest, err := compile(root)
	if err != nil {
		t.Fatal(err)
	}
	content, err := manifest.Content()
	if err != nil {
		t.Fatal(err)
	}
	var header map[string]json.RawMessage
	if err := json.Unmarshal(content.Data, &header); err != nil {
		t.Fatal(err)
	}
	if _, carried := header["source_identity"]; carried {
		t.Fatal("the manifest carries an identity of the whole source tree again")
	}
	if len(manifest.Documents) == 0 {
		t.Fatal("the manifest declares no documents")
	}
	for _, document := range manifest.Documents {
		if document.SourceIdentity == "" {
			t.Errorf("document %s lost the identity of its declaring source %s", document.Name, document.Source)
		}
	}
	published, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(manifestJSONPath)))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(bytes.TrimSpace(published), bytes.TrimSpace(content.Data)) {
		t.Fatalf("%s is not what the source compiles to; regenerate with go run ./cmd/api-manifest -update", manifestJSONPath)
	}
}
