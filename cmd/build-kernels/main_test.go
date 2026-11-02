package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// TestBuildFlagsMatchTheManifest: the flags each module builds with are the ones the manifest records for it, and the table
// and the manifest name the same modules.
func TestBuildFlagsMatchTheManifest(t *testing.T) {
	t.Parallel()
	data, err := os.ReadFile(filepath.Join("..", "..", "kernels", "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	type recorded struct {
		Source        string   `json:"source"`
		CompilerFlags []string `json:"compilerFlags"`
	}
	var document struct {
		CompilerFlags []string   `json:"compilerFlags"`
		Modules       []recorded `json:"modules"`
	}
	if err := json.Unmarshal(data, &document); err != nil {
		t.Fatal(err)
	}
	if len(document.Modules) != len(kernels) {
		t.Fatalf("the manifest holds %d modules and the build table %d", len(document.Modules), len(kernels))
	}
	for _, item := range kernels {
		at := slices.IndexFunc(document.Modules, func(module recorded) bool { return module.Source == item.source })
		if at < 0 {
			t.Fatalf("the manifest holds no module for %s", item.source)
		}
		flags := document.Modules[at].CompilerFlags
		if len(flags) == 0 {
			flags = document.CompilerFlags
		}
		if !slices.Equal(item.flags(), flags) {
			t.Errorf("%s builds with %v and the manifest records %v", item.source, item.flags(), flags)
		}
	}
}

// TestNormalizePTXRewritesLineEnds holds a generated PTX file to Unix line ends, which the pins kernel-bindings
// writes from the manifest hash whatever nvcc wrote.
func TestNormalizePTXRewritesLineEnds(t *testing.T) {
	asset := filepath.Join(t.TempDir(), "kernel.ptx")
	if err := os.WriteFile(asset, []byte("first\r\nsecond\r\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := normalizePTX(asset); err != nil {
		t.Fatal(err)
	}
	payload, err := os.ReadFile(asset)
	if err != nil {
		t.Fatal(err)
	}
	if string(payload) != "first\nsecond\n" {
		t.Fatalf("payload = %q, want Unix line ends", payload)
	}
}
