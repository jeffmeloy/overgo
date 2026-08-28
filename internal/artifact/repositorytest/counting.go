package repositorytest

import (
	"context"

	"overgo/internal/artifact"
)

// CountingRepository wraps a Repository and counts Commit calls, so a
// test can assert one semantic transition publishes through exactly
// one atomic batch instead of a sequence of per-document commits.
type CountingRepository struct {
	artifact.Repository
	Commits int
}

// Commit delegates to the wrapped repository and counts the call.
func (c *CountingRepository) Commit(ctx context.Context, batch artifact.Batch) (artifact.CommitID, error) {
	c.Commits++
	return c.Repository.Commit(ctx, batch)
}
