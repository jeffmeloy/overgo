package overgodb

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/artifact/repositorytest"
)

func TestRepositoryContract(t *testing.T) {
	t.Parallel()
	repositorytest.Run(t, func(t *testing.T) artifact.Repository {
		t.Helper()
		store, err := Open(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		return store
	})
}
