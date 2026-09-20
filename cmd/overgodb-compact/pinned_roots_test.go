package main

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
)

// TestPinnedRootsScanCommittedDocuments collects every identity the checkout's
// JSON documents name, once each and sorted, and ignores text that is not an
// identity and files that are not JSON.
func TestPinnedRootsScanCommittedDocuments(t *testing.T) {
	checkout := t.TempDir()
	if err := os.MkdirAll(filepath.Join(checkout, "docs", "verification"), 0o755); err != nil {
		t.Fatal(err)
	}
	first := "evidence:sha256:" + strings.Repeat("a", 64)
	second := "model:sha256:" + strings.Repeat("b", 64)
	ignored := "file:sha256:" + strings.Repeat("c", 64)
	write := func(relative, text string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(checkout, relative), []byte(text), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("docs/verification/receipt.json", `{"evidence":["`+first+`","`+first+`"],"note":"not:an:id sha256:short"}`)
	write("compatibility.json", `{"model":"`+second+`"}`)
	write("docs/notes.md", ignored)
	roots, err := pinnedRoots(checkout)
	if err != nil {
		t.Fatal(err)
	}
	want := []artifact.ID{}
	for _, text := range []string{first, second} {
		id, err := artifact.ParseID(text)
		if err != nil {
			t.Fatal(err)
		}
		want = append(want, id)
	}
	slices.SortFunc(want, artifact.CompareID)
	if !slices.Equal(roots, want) {
		t.Fatalf("pinned roots = %v, want %v", roots, want)
	}
}
