package modelartifact

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testcheck"
)

func TestRegistrationRejectsUnboundedMetadata(t *testing.T) {
	for _, size := range []int64{0, artifact.MaxContentBytes + 1} {
		path := filepath.Join(t.TempDir(), "metadata.json")
		file := testcheck.Must(os.Create(path))(t)
		testcheck.NoError(t, file.Truncate(size))
		testcheck.NoError(t, file.Close())
		if _, err := PrepareRegistration(t.Context(), t.TempDir(), path); err == nil {
			t.Fatalf("accepted metadata size %d", size)
		}
	}
	if _, err := artifact.ReadContentFile(t.TempDir()); err == nil {
		t.Fatal("accepted a directory as metadata")
	}
}
