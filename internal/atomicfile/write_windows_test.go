//go:build windows

package atomicfile

import (
	"os"
	"path/filepath"
	"testing"
)

func TestCompareAndSwapReadOnlyReplacementFailsBeforeDetach(t *testing.T) {
	root := t.TempDir()
	directory := filepath.Join(root, "documents")
	scratch := filepath.Join(root, "scratch")
	if err := os.Mkdir(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(scratch, 0o700); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(directory, "document")
	if err := os.WriteFile(path, []byte("before"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := CompareAndSwap(path, scratch, []byte("before"), []byte("after"), 0o400); err == nil {
		t.Fatal("read-only Windows replacement unexpectedly succeeded")
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "before" {
		t.Fatalf("refused replacement changed target to %q", got)
	}
	entries, err := os.ReadDir(scratch)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Fatalf("refused replacement retained %d scratch entries", len(entries))
	}
}
