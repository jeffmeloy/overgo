package modelartifact

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
)

func TestRegistrationRejectsUnboundedMetadata(t *testing.T) {
	for _, size := range []int64{0, artifact.MaxContentBytes + 1} {
		path := filepath.Join(t.TempDir(), "metadata.json")
		file, err := os.Create(path)
		if err != nil {
			t.Fatal(err)
		}
		if err := file.Truncate(size); err != nil {
			t.Fatal(err)
		}
		if err := file.Close(); err != nil {
			t.Fatal(err)
		}
		if _, err := PrepareRegistration(t.Context(), t.TempDir(), path); err == nil {
			t.Fatalf("accepted metadata size %d", size)
		}
	}
	if _, err := artifact.ReadContentFile(t.TempDir()); err == nil {
		t.Fatal("accepted a directory as metadata")
	}
}
