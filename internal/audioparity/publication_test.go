package audioparity

import (
	"errors"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

// audioPublication is the shared idempotent publication boundary for real-model
// integration fixtures. It owns no model or training-specific policy.
type audioPublication struct{ store *overgodb.Store }

func (p *audioPublication) commit(t *testing.T, batch artifact.Batch, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(t.Context(), p.store, batch); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		t.Fatal(err)
	}
}
