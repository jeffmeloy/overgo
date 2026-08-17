package export

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestCopyFilesPublishesExactBytes(t *testing.T) {
	source, target := t.TempDir(), t.TempDir()
	name := "config.json"
	data := []byte(`{"format":"fixture"}`)
	if err := os.WriteFile(filepath.Join(source, name), data, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := CopyFiles(source, target, []string{name}); err != nil {
		t.Fatal(err)
	}
	written, err := os.ReadFile(filepath.Join(target, name))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(written, data) {
		t.Fatalf("written = %q", written)
	}
	if err := CopyFiles(source, target, []string{name}); err == nil {
		t.Fatal("existing export overwritten")
	}
}
