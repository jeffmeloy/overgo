// Package repositorytest provides the storage-neutral Repository contract.
package repositorytest

import (
	"context"
	"errors"
	"testing"

	"overgo/internal/artifact"
)

// Factory opens an isolated writable repository for one contract subtest.
type Factory func(*testing.T) artifact.Repository

// Run exercises behavior every artifact repository backend must preserve.
func Run(t *testing.T, open Factory) {
	t.Helper()
	t.Run("conditional commit", func(t *testing.T) {
		repository := open(t)
		defer repository.Close()
		ctx := context.Background()
		empty := artifact.CommitID{}
		first := contractBatch(t, "repository-contract/first", "first")
		first.ExpectedHead = &empty
		firstCommit, err := repository.Commit(ctx, first)
		if err != nil {
			t.Fatal(err)
		}

		stale := contractBatch(t, "repository-contract/stale", "stale")
		stale.ExpectedHead = &empty
		if _, err := repository.Commit(ctx, stale); !errors.Is(err, artifact.ErrCommitPrecondition) {
			t.Fatalf("stale commit error = %v", err)
		}
		if found, err := anyArtifactFound(ctx, repository, stale); err != nil || found {
			t.Fatalf("stale artifact published = (%v, %v)", found, err)
		}

		next := contractBatch(t, "repository-contract/next", "next")
		next.ExpectedHead = &firstCommit
		if _, err := repository.Commit(ctx, next); err != nil {
			t.Fatal(err)
		}
		// Retry is checked by key and canonical payload before the now-stale
		// head precondition, so uncertain callers can safely repeat it.
		replayed, err := repository.Commit(ctx, first)
		if err != nil || replayed != firstCommit {
			t.Fatalf("idempotent conditional retry = (%s, %v), want (%s, nil)", replayed, err, firstCommit)
		}
	})

	t.Run("cancelled commit", func(t *testing.T) {
		repository := open(t)
		defer repository.Close()
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		batch := contractBatch(t, "repository-contract/cancelled", "cancelled")
		if _, err := repository.Commit(ctx, batch); !errors.Is(err, context.Canceled) {
			t.Fatalf("cancelled commit error = %v", err)
		}
		if found, err := anyArtifactFound(context.Background(), repository, batch); err != nil || found {
			t.Fatalf("cancelled artifact published = (%v, %v)", found, err)
		}
	})
}

func anyArtifactFound(ctx context.Context, repository artifact.Repository, batch artifact.Batch) (bool, error) {
	for _, descriptor := range batch.Artifacts {
		_, found, err := repository.Artifact(ctx, descriptor.ID)
		if err != nil || found {
			return found, err
		}
	}
	return false, nil
}

func contractBatch(t *testing.T, key, payload string) artifact.Batch {
	t.Helper()
	id, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte(payload))
	if err != nil {
		t.Fatal(err)
	}
	return artifact.Batch{Key: key, Artifacts: []artifact.Descriptor{{
		ID: id, Size: uint64(len(payload)), MediaType: "application/octet-stream",
	}}}
}
