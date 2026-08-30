package server

import (
	"path/filepath"
	"testing"

	"overgo/internal/overgodb"
)

// TestServingEnvironmentCommitIsIdempotent pins restart idempotency: two
// serving sessions on the same machine derive the identical environment
// record, and the second session must open against the stored fact
// instead of refusing on a no-op batch.
func TestServingEnvironmentCommitIsIdempotent(t *testing.T) {
	store, err := overgodb.Open(filepath.Join(t.TempDir(), "repodb"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	for session := 0; session < 2; session++ {
		if _, _, err := openServingRepository(Config{Repository: store}); err != nil {
			t.Fatalf("serving session %d environment: %v", session, err)
		}
	}
}
