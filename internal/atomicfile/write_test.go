package atomicfile

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestWriteReplacesExistingFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "document")
	original := []byte("before")
	if err := os.WriteFile(path, original, 0o600); err != nil {
		t.Fatal(err)
	}
	failed := errors.New("replace interrupted")
	err := write(path, []byte("after"), 0o600, func(string, string) error { return failed })
	if !errors.Is(err, failed) {
		t.Fatalf("write error = %v, want %v", err, failed)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(original) {
		t.Fatalf("failed replacement changed target: got %q want %q", got, original)
	}
	if err := Write(path, []byte("after"), 0o600); err != nil {
		t.Fatal(err)
	}
	got, err = os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "after" {
		t.Fatalf("replacement = %q, want after", got)
	}
}

func TestWriteRequiresExistingParent(t *testing.T) {
	directory := filepath.Join(t.TempDir(), "caller-owned")
	path := filepath.Join(directory, "document")
	if err := Write(path, []byte("after"), 0o600); err == nil {
		t.Fatal("write created a caller-owned parent directory")
	}
	if _, err := os.Stat(directory); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("parent directory state = %v, want absent", err)
	}
}
