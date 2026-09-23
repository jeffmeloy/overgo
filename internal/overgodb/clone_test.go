package overgodb

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// TestCloneStoreSharesBlobsAndKeepsTheChain clones a snapshotted store: the
// clone opens at the same head, its blobs are the source's files, releasing
// in the clone leaves the source's bytes in place, and an existing
// destination is refused.
func TestCloneStoreSharesBlobsAndKeepsTheChain(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	fixture := newReleaseFixture(t)
	if _, err := fixture.store.Snapshot(ctx); err != nil {
		t.Fatal(err)
	}
	head, sequence := fixture.store.Head()
	destination := filepath.Join(t.TempDir(), "candidate")
	report, err := CloneStore(fixture.root, destination)
	if err != nil {
		t.Fatal(err)
	}
	if report.LinkedBlobs != 3 || report.CopiedFiles == 0 || report.CopiedBytes == 0 {
		t.Fatalf("clone report = %+v", report)
	}
	if _, err := CloneStore(fixture.root, destination); err == nil {
		t.Fatal("clone wrote into an existing destination")
	}
	if _, err := os.Stat(filepath.Join(destination, lockFilename)); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("clone carried the lock: %v", err)
	}

	candidate, err := Open(destination)
	if err != nil {
		t.Fatal(err)
	}
	defer candidate.Close()
	if cloneHead, cloneSequence := candidate.Head(); cloneHead != head || cloneSequence != sequence {
		t.Fatalf("clone head (%s, %d), want (%s, %d)", cloneHead, cloneSequence, head, sequence)
	}
	orphan := fixture.orphan.Descriptor.ID
	sourceBlob, err := fixture.store.blobs.path(orphan)
	if err != nil {
		t.Fatal(err)
	}
	cloneBlob, err := candidate.blobs.path(orphan)
	if err != nil {
		t.Fatal(err)
	}
	sourceInfo, sourceErr := os.Stat(sourceBlob)
	cloneInfo, cloneErr := os.Stat(cloneBlob)
	if sourceErr != nil || cloneErr != nil || !os.SameFile(sourceInfo, cloneInfo) {
		t.Fatalf("clone blob is not the source file: %v %v", sourceErr, cloneErr)
	}
	if _, err := Release(ctx, candidate, nil, RetentionPolicy{}, releasable); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(cloneBlob); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("released clone blob still present: %v", err)
	}
	if has, err := fixture.store.HasContent(ctx, orphan); err != nil || !has {
		t.Fatalf("release in the clone touched the source: (%v, %v)", has, err)
	}
	_, reader, found, err := fixture.store.OpenContent(ctx, orphan)
	if err != nil || !found {
		t.Fatalf("source bytes after the clone's release = (%v, %v)", found, err)
	}
	if closer, ok := reader.(interface{ Close() error }); ok {
		_ = closer.Close()
	}
}
