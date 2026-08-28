package repositorytest

import (
	"context"
	"io"

	"overgo/internal/artifact"
)

// CountingRepository wraps a Repository and counts Commit calls, so a
// test can assert one semantic transition publishes through exactly
// one atomic batch instead of a sequence of per-document commits.
type CountingRepository struct {
	artifact.Repository
	Commits int
	// Opens counts per-item content opens; Batches counts batched
	// content visits, so a test can assert reads do not scale per item.
	Opens   int
	Batches int
}

// Commit delegates to the wrapped repository and counts the call.
func (c *CountingRepository) Commit(ctx context.Context, batch artifact.Batch) (artifact.CommitID, error) {
	c.Commits++
	return c.Repository.Commit(ctx, batch)
}

// OpenContent delegates and counts one per-item open.
func (c *CountingRepository) OpenContent(ctx context.Context, id artifact.ID) (artifact.Descriptor, io.Reader, bool, error) {
	c.Opens++
	return c.Repository.OpenContent(ctx, id)
}

// VisitContents delegates to the wrapped repository's batch visitor
// and counts one batch; a repository without one fails the test that
// expected batching.
func (c *CountingRepository) VisitContents(ctx context.Context, ids []artifact.ID, visit func(artifact.Descriptor, io.Reader) error) error {
	c.Batches++
	return c.Repository.(artifact.ContentVisitor).VisitContents(ctx, ids, visit)
}

// PresentContents delegates to the wrapped repository's presence
// batch and counts one batch acquisition.
func (c *CountingRepository) PresentContents(ctx context.Context, ids []artifact.ID) ([]artifact.ID, error) {
	c.Batches++
	return c.Repository.(artifact.ContentPresence).PresentContents(ctx, ids)
}
