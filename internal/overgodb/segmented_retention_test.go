package overgodb

import (
	"bytes"
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
)

// TestSegmentedRetentionPreservesAuthorityClosure holds the in-place release
// on the segmented layout to its contract: the alias-rooted authority closure
// keeps every referenced blob readable, what nothing reaches loses its blob,
// and the closing snapshot replaces a checkpoint set that no longer forms a
// valid anchored set.
func TestSegmentedRetentionPreservesAuthorityClosure(t *testing.T) {
	ctx := t.Context()
	root := t.TempDir()
	store, err := Open(root)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	rootedPayload := []byte("segmented-retention rooted content")
	rootedID, err := artifact.IdentifyBytes(artifact.KindModel, rootedPayload)
	if err != nil {
		t.Fatal(err)
	}
	rooted := artifact.Descriptor{ID: rootedID, Size: uint64(len(rootedPayload)), MediaType: "application/octet-stream"}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:      "retention/rooted",
		Contents: []artifact.Content{{Descriptor: rooted, Data: rootedPayload}},
		Aliases:  []artifact.AliasBinding{{Name: "retention/root", Target: rootedID}},
	}); err != nil {
		t.Fatal(err)
	}
	// Blob-sized: the release below must remove a blob file, and smaller
	// content rides in its frame and has none.
	orphanPayload := bytes.Repeat([]byte("segmented-retention unreachable content "), inlineContentLimit)
	orphanID, err := artifact.IdentifyBytes(artifact.KindOutput, orphanPayload)
	if err != nil {
		t.Fatal(err)
	}
	orphan := artifact.Descriptor{ID: orphanID, Size: uint64(len(orphanPayload)), MediaType: "application/octet-stream"}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:      "retention/orphan",
		Contents: []artifact.Content{{Descriptor: orphan, Data: orphanPayload}},
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Snapshot(ctx); err != nil {
		t.Fatal(err)
	}

	// Invalidate the checkpoint set; the release's snapshot must replace it.
	victim := filepath.Join(newestCheckpoints(t, root), "aliases"+checkpointExtension)
	data, err := os.ReadFile(victim)
	if err != nil {
		t.Fatal(err)
	}
	data[len(data)-2] ^= 0xFF
	if err := os.WriteFile(victim, data, storeFileMode); err != nil {
		t.Fatal(err)
	}
	if _, _, valid, _ := loadProjectionCheckpoints(root); valid {
		t.Fatal("corrupted checkpoint set still loads")
	}

	report, err := Release(ctx, store, nil, RetentionPolicy{}, releaseEverything)
	if err != nil {
		t.Fatal(err)
	}
	if report.Released != 1 || report.BlobsRemoved != 1 || report.ReleasedBytes != int64(len(orphanPayload)) {
		t.Fatalf("unreachable blob not released: %+v", report)
	}
	blob, err := store.blobs.path(orphanID)
	if err != nil {
		t.Fatal(err)
	}
	if _, statErr := os.Stat(blob); !errors.Is(statErr, os.ErrNotExist) {
		t.Fatalf("unreachable blob stayed on disk: %v", statErr)
	}
	if _, _, valid, _ := loadProjectionCheckpoints(root); !valid {
		t.Fatal("release left an invalid checkpoint set")
	}
	target, found, err := store.ResolveAlias(ctx, "retention/root")
	if err != nil || !found || target != rootedID {
		t.Fatalf("alias closure lost: (%s, %v, %v)", target, found, err)
	}
	_, reader, found, err := store.OpenContent(ctx, rootedID)
	if err != nil || !found {
		t.Fatalf("retained content = (%v, %v)", found, err)
	}
	if data, err := io.ReadAll(reader); err != nil || string(data) != string(rootedPayload) {
		t.Fatalf("retained content bytes = (%q, %v)", data, err)
	}
	if has, err := store.HasContent(ctx, orphanID); err != nil || has {
		t.Fatalf("unreachable content survived the release: has=%v err=%v", has, err)
	}
}
