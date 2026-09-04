package longform

import (
	"os"
	"path/filepath"
	"testing"
)

// The file digest is a function of root-relative paths and contents:
// the same files under another root digest the same, one changed byte
// changes it, and the order the files are named in does not matter.
func TestDigestFilesIsPathAndContentBound(t *testing.T) {
	write := func(root string, contents map[string]string) []string {
		var files []string
		for name, content := range contents {
			path := filepath.Join(root, name)
			if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
				t.Fatal(err)
			}
			files = append(files, path)
		}
		return files
	}
	contents := map[string]string{"a/x.go": "package a\n", "kernels/k.cu": "__global__ void k() {}\n"}
	first, second := t.TempDir(), t.TempDir()
	firstFiles, secondFiles := write(first, contents), write(second, contents)
	firstDigest, err := digestFiles(first, firstFiles)
	if err != nil {
		t.Fatal(err)
	}
	secondDigest, err := digestFiles(second, []string{secondFiles[1], secondFiles[0]})
	if err != nil {
		t.Fatal(err)
	}
	if firstDigest != secondDigest || len(firstDigest) != 64 {
		t.Fatalf("digests differ across roots: %s vs %s", firstDigest, secondDigest)
	}
	if err := os.WriteFile(filepath.Join(first, "a/x.go"), []byte("package a // changed\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	changed, err := digestFiles(first, firstFiles)
	if err != nil {
		t.Fatal(err)
	}
	if changed == firstDigest {
		t.Fatal("a changed file left the digest unchanged")
	}
}

// The repository's own inference surface resolves through the Go tool
// and is stable within a process.
func TestSurfaceOfTheRepositoryIsStable(t *testing.T) {
	ctx := t.Context()
	first, err := Surface(ctx, "../..")
	if err != nil {
		t.Fatal(err)
	}
	second, err := Surface(ctx, "../..")
	if err != nil || first != second || len(first) != 64 {
		t.Fatalf("surface = %q, %q, %v", first, second, err)
	}
}
