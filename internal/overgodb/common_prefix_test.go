package overgodb

import (
	"context"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
)

func TestCommonCommitPrefixFindsGreatestSharedCoordinate(t *testing.T) {
	leftRoot := filepath.Join(t.TempDir(), "left")
	left, err := Open(leftRoot)
	if err != nil {
		t.Fatal(err)
	}
	first := commonPrefixCommit(t, left, "first")
	second := commonPrefixCommit(t, left, "second")
	rightRoot := filepath.Join(t.TempDir(), "right")
	if _, err := left.Backup(t.Context(), rightRoot); err != nil {
		t.Fatal(err)
	}
	commonPrefixCommit(t, left, "left-tail")
	right, err := Open(rightRoot)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = left.Close(); _ = right.Close() })
	commonPrefixCommit(t, right, "right-tail")

	prefix, err := CommonCommitPrefix(t.Context(), left, right)
	if err != nil {
		t.Fatal(err)
	}
	if prefix != (CommitPrefix{Commit: second, Sequence: 2}) || prefix.Commit == first {
		t.Fatalf("common prefix = %+v, want second commit", prefix)
	}
}

func TestCommonCommitPrefixDistinguishesIdenticalAndForeignStores(t *testing.T) {
	leftRoot := filepath.Join(t.TempDir(), "left")
	left, err := Open(leftRoot)
	if err != nil {
		t.Fatal(err)
	}
	head := commonPrefixCommit(t, left, "shared")
	backupRoot := filepath.Join(t.TempDir(), "backup")
	if _, err := left.Backup(t.Context(), backupRoot); err != nil {
		t.Fatal(err)
	}
	backup, err := OpenReadOnly(backupRoot)
	if err != nil {
		t.Fatal(err)
	}
	foreign, err := Open(filepath.Join(t.TempDir(), "foreign"))
	if err != nil {
		t.Fatal(err)
	}
	commonPrefixCommit(t, foreign, "foreign")
	t.Cleanup(func() { _ = left.Close(); _ = backup.Close(); _ = foreign.Close() })

	identical, err := CommonCommitPrefix(t.Context(), left, backup)
	if err != nil || identical != (CommitPrefix{Commit: head, Sequence: 1}) {
		t.Fatalf("identical prefix = (%+v, %v)", identical, err)
	}
	separate, err := CommonCommitPrefix(t.Context(), left, foreign)
	if err != nil || separate != (CommitPrefix{}) {
		t.Fatalf("foreign prefix = (%+v, %v)", separate, err)
	}
	if _, err := CommonCommitPrefix(nil, left, backup); err == nil {
		t.Fatal("nil context gained prefix authority")
	}
	cancelled, cancel := context.WithCancelCause(t.Context())
	cancel(context.Canceled)
	if _, err := CommonCommitPrefix(cancelled, left, backup); err == nil {
		t.Fatal("cancelled prefix comparison succeeded")
	}
}

func commonPrefixCommit(t *testing.T, store *Store, label string) artifact.CommitID {
	t.Helper()
	payload := []byte(label)
	id, err := artifact.IdentifyBytes(artifact.KindEvidence, payload)
	if err != nil {
		t.Fatal(err)
	}
	commit, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{
		Key: "fixture/common-prefix/" + label,
		Contents: []artifact.Content{{
			Descriptor: artifact.Descriptor{ID: id, Size: uint64(len(payload)), MediaType: "text/plain"},
			Data:       payload,
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	return commit
}
