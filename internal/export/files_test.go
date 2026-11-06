package export

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/testcheck"
)

func TestCopyFilesPublishesExactBytes(t *testing.T) {
	source, target := t.TempDir(), t.TempDir()
	name := "config.json"
	data := []byte(`{"format":"fixture"}`)
	testcheck.NoError(t, os.WriteFile(filepath.Join(source, name), data, 0o644))
	testcheck.NoError(t, CopyFiles(source, target, []string{name}))
	written := testcheck.Must(os.ReadFile(filepath.Join(target, name)))(t)
	if !bytes.Equal(written, data) {
		t.Fatalf("written = %q", written)
	}
	if err := CopyFiles(source, target, []string{name}); err == nil {
		t.Fatal("existing export overwritten")
	}
}
