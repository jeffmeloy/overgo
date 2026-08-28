package overgodb

import (
	"context"
	"io"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
)

// TestSegmentedRetentionPreservesAuthorityClosure holds retention on
// the segmented layout to its contract: the destination rebuild
// retains the alias-rooted authority closure with every referenced
// blob readable, drops what nothing reaches, and the report names the
// reclaimable source material -- unreachable blobs by count and bytes,
// and checkpoint files that no longer form a valid set -- without
// deleting anything in place.
func TestSegmentedRetentionPreservesAuthorityClosure(t *testing.T) {
	ctx := context.Background()
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
	orphanPayload := []byte("segmented-retention unreachable content")
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

	// Invalidate the checkpoint set so its files count as obsolete.
	victim := filepath.Join(root, checkpointDirectory, "aliases"+checkpointExtension)
	data, err := os.ReadFile(victim)
	if err != nil {
		t.Fatal(err)
	}
	data[len(data)-2] ^= 0xFF
	if err := os.WriteFile(victim, data, storeFileMode); err != nil {
		t.Fatal(err)
	}

	destination := filepath.Join(t.TempDir(), "retained")
	report, err := Compact(ctx, store, destination)
	if err != nil {
		t.Fatal(err)
	}
	if report.ReclaimableBlobs < 1 || report.ReclaimableBlobBytes < int64(len(orphanPayload)) {
		t.Fatalf("unreachable blob not reported: %+v", report)
	}
	if report.ObsoleteCheckpoints == 0 {
		t.Fatalf("invalid checkpoint set not reported: %+v", report)
	}
	if blob, err := store.blobs.path(orphanID); err != nil {
		t.Fatal(err)
	} else if _, statErr := os.Stat(blob); statErr != nil {
		t.Fatal("retention deleted source material in place")
	}

	retained, err := OpenReadOnly(destination)
	if err != nil {
		t.Fatal(err)
	}
	defer retained.Close()
	target, found, err := retained.ResolveAlias(ctx, "retention/root")
	if err != nil || !found || target != rootedID {
		t.Fatalf("alias closure lost: (%s, %v, %v)", target, found, err)
	}
	_, reader, found, err := retained.OpenContent(ctx, rootedID)
	if err != nil || !found {
		t.Fatalf("retained content = (%v, %v)", found, err)
	}
	if data, err := io.ReadAll(reader); err != nil || string(data) != string(rootedPayload) {
		t.Fatalf("retained content bytes = (%q, %v)", data, err)
	}
	if _, found, err := retained.Artifact(ctx, orphanID); err != nil || found {
		t.Fatalf("unreachable artifact survived retention: found=%v err=%v", found, err)
	}
	if retained.blobs.has(orphanID) {
		t.Fatal("unreachable blob copied into the destination")
	}
}
