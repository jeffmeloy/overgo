package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestASTStructuralProfileGate(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "internal", "p", "p.go")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("package p\nfunc F(v int) int { return v }\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	g := gateContext{repo: root, paths: []string{"internal/p/p.go"}}
	skipped, err := g.stepProfile()
	if err != nil || skipped || len(g.honesty) != 1 || !strings.Contains(g.honesty[0], "production=1 files") {
		t.Fatalf("profile step = skipped %v, err %v, honesty %v", skipped, err, g.honesty)
	}
}
