package server

import (
	"path/filepath"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/testcheck"
)

// TestServingEnvironmentCommitIsIdempotent pins restart idempotency: two
// serving sessions on the same machine derive the identical environment
// record, and the second session must open against the stored fact
// instead of refusing on a no-op batch.
func TestServingEnvironmentCommitIsIdempotent(t *testing.T) {
	t.Parallel()
	store, err := overgodb.Create(filepath.Join(t.TempDir(), "repodb"))
	testcheck.NoError(t, err)
	defer store.Close()
	for session := range 2 {
		if _, _, err := openServingRepository(Config{Repository: store}); err != nil {
			t.Fatalf("serving session %d environment: %v", session, err)
		}
	}
}
