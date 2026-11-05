package main

import (
	"io"
	"path/filepath"
	"testing"

	"overgo/internal/testcheck"
)

// TestPublishedMatricesAreCurrent holds the committed compatibility and
// training matrices to the claims they are generated from, and every claim to
// the sources it binds, as `validate -claims` does, so go test answers for
// them. A claim whose source moved fails here until `validate -claims -run`
// re-runs its verify.
func TestPublishedMatricesAreCurrent(t *testing.T) {
	t.Parallel()
	testcheck.NoError(t, publishedClaims(t.Context(), io.Discard, filepath.Join("..", ".."), false))
}
