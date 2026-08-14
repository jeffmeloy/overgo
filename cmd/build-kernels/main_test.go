package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const normalizedPTXFixtureSHA256 = "dbea9325179efe46ea2add94f7b6b745ca983fabb208dc6d34aa064623d7ee23"

func TestRefreshPinsNormalizesGeneratedPTX(t *testing.T) {
	directory := t.TempDir()
	asset := filepath.Join(directory, "kernel.ptx")
	validation := filepath.Join(directory, "manifest.go")
	if err := os.WriteFile(asset, []byte("first\r\nsecond\r\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(validation, []byte("const (\n\tTestSHA256 = \""+strings.Repeat("0", 64)+"\"\n)\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	prior := runtimePins
	runtimePins = map[string]string{"TestSHA256": asset}
	t.Cleanup(func() { runtimePins = prior })
	if err := refreshPins(validation); err != nil {
		t.Fatal(err)
	}
	payload, err := os.ReadFile(asset)
	if err != nil {
		t.Fatal(err)
	}
	wantPayload := []byte("first\nsecond\n")
	if string(payload) != string(wantPayload) {
		t.Fatalf("payload = %q", payload)
	}
	manifest, err := os.ReadFile(validation)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(manifest), normalizedPTXFixtureSHA256) {
		t.Fatalf("manifest = %q", manifest)
	}
}
