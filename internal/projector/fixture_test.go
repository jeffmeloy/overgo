package projector

import (
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/testevidence"
	"overgo/internal/testutil"
)

func classifyProjectorIntegration(t *testing.T) {
	t.Helper()
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip)
	}
}

func writeProjectorFixture(
	t *testing.T,
	name string,
	metadata []gguf.Metadata,
	tensors []gguf.TensorData,
) string {
	t.Helper()
	return testutil.TempGGUF(t, name, metadata, tensors)
}

func rewriteProjectorFixture(
	t *testing.T,
	path string,
	metadata []gguf.Metadata,
	tensors []gguf.TensorData,
) {
	t.Helper()
	testutil.WriteGGUF(t, path, metadata, tensors)
}
