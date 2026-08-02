package projector

import (
	"os"
	"path/filepath"
	"testing"

	"llamacpp2go/internal/gguf"
)

func writeProjectorFixture(
	t *testing.T,
	name string,
	metadata []gguf.Metadata,
	tensors []gguf.TensorData,
) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	rewriteProjectorFixture(t, path, metadata, tensors)
	return path
}

func rewriteProjectorFixture(
	t *testing.T,
	path string,
	metadata []gguf.Metadata,
	tensors []gguf.TensorData,
) {
	t.Helper()
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := gguf.Write(file, metadata, tensors, gguf.WriteOptions{}); err != nil {
		_ = file.Close()
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
}
