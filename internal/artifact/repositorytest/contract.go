// Package repositorytest provides the storage-neutral Repository contract.
package repositorytest

import (
	"context"
	"errors"
	"io"
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

	t.Run("read back", func(t *testing.T) {
		repository := open(t)
		defer repository.Close()
		ctx := context.Background()
		payload := "read-back"
		batch := contractBatch(t, "repository-contract/read", payload)
		id := batch.Artifacts[0].ID
		batch.Contents = []artifact.Content{{Descriptor: batch.Artifacts[0], Data: []byte(payload)}}
		batch.Aliases = []artifact.AliasBinding{{Name: "repository-contract/alias", Target: id}}
		if _, err := repository.Commit(ctx, batch); err != nil {
			t.Fatal(err)
		}
		descriptor, found, err := repository.Artifact(ctx, id)
		if err != nil || !found || descriptor != batch.Artifacts[0] {
			t.Fatalf("artifact = (%+v, %v, %v)", descriptor, found, err)
		}
		_, reader, found, err := repository.OpenContent(ctx, id)
		if err != nil || !found {
			t.Fatalf("content = (%v, %v)", found, err)
		}
		data, err := io.ReadAll(reader)
		if err != nil || string(data) != payload {
			t.Fatalf("content bytes = (%q, %v)", data, err)
		}
		target, found, err := repository.ResolveAlias(ctx, "repository-contract/alias")
		if err != nil || !found || target != id {
			t.Fatalf("alias = (%s, %v, %v)", target, found, err)
		}
	})

	t.Run("lineage", func(t *testing.T) {
		repository := open(t)
		defer repository.Close()
		ctx := context.Background()
		parent := contractBatch(t, "repository-contract/parent", "lineage-parent")
		child := contractBatch(t, "repository-contract/child", "lineage-child")
		child.Lineage = []artifact.Lineage{{
			Child: child.Artifacts[0].ID, Parent: parent.Artifacts[0].ID, Relation: artifact.RelationDerivedFrom,
		}}
		if _, err := repository.Commit(ctx, parent); err != nil {
			t.Fatal(err)
		}
		if _, err := repository.Commit(ctx, child); err != nil {
			t.Fatal(err)
		}
		parents, err := repository.Parents(ctx, child.Artifacts[0].ID)
		if err != nil || len(parents) != 1 || parents[0].Parent != parent.Artifacts[0].ID {
			t.Fatalf("parents = (%+v, %v)", parents, err)
		}
		children, err := repository.Children(ctx, parent.Artifacts[0].ID)
		if err != nil || len(children) != 1 || children[0].Child != child.Artifacts[0].ID {
			t.Fatalf("children = (%+v, %v)", children, err)
		}
		cycle := artifact.Batch{Key: "repository-contract/cycle", Lineage: []artifact.Lineage{{
			Child: parent.Artifacts[0].ID, Parent: child.Artifacts[0].ID, Relation: artifact.RelationDerivedFrom,
		}}}
		if _, err := repository.Commit(ctx, cycle); err == nil {
			t.Fatal("lineage cycle admitted")
		}
	})

	t.Run("closed repository refuses work", func(t *testing.T) {
		repository := open(t)
		if err := repository.Close(); err != nil {
			t.Fatal(err)
		}
		ctx := context.Background()
		if _, err := repository.Commit(ctx, contractBatch(t, "repository-contract/closed", "closed")); err == nil {
			t.Fatal("closed repository accepted a commit")
		}
		if _, _, err := repository.Artifact(ctx, contractBatch(t, "repository-contract/closed-read", "closed-read").Artifacts[0].ID); err == nil {
			t.Fatal("closed repository answered a read")
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
