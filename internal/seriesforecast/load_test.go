package seriesforecast

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/modeltest"
)

// artifactDir resolves the real timesfm artifact through the data-root
// contract; absent artifact skips LOUDLY (UNAVAILABLE is never green — the
// skip names what was not tested).
func artifactDir(t *testing.T) string {
	t.Helper()
	dir := modeltest.Directory(t, "timesfm-2.5-200m-transformers")
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
		t.Skipf("UNAVAILABLE: timesfm artifact absent at %s; dims-vs-artifact NOT verified", dir)
	}
	return dir
}
