package main

import (
	"os"
	"path/filepath"
	"testing"
)

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
