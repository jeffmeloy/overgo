package overgodb

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/artifact/repositorytest"
	"overgo/internal/testcheck"
)

func TestRepositoryContract(t *testing.T) {
	t.Parallel()
	repositorytest.Run(t, func(t *testing.T) artifact.Repository {
		t.Helper()
		store, err := Open(t.TempDir())
		testcheck.NoError(t, err)
		return store
	})
}
