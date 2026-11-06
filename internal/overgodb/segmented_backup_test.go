package overgodb

import (
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testcheck"
)

// TestSegmentedBackupAndReadOnlyRefresh holds maintenance to the
// segmented layout's contract: the snapshot boundary seals the active
// journal, backup captures the stable extent -- sealed segments, the
// active tail to the captured offset, every required blob, and the
// checkpoint anchors -- so the restored store answers the bounded
// query surface identically at the same head; and a read-only handle
// tailing across a snapshot-driven rotation refreshes onto the
// writer's head.
func TestSegmentedBackupAndReadOnlyRefresh(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	root := t.TempDir()
	copyScaleCorpus(t, root)

	writer := testcheck.Must(Create(root))(t)
	defer writer.Close()
	if sealed, err := filepath.Glob(filepath.Join(root, segmentDirectory, "*"+segmentExtension)); err != nil || len(sealed) == 0 {
		t.Fatalf("snapshot boundary did not seal: (%v, %v)", sealed, err)
	}
	reader := testcheck.Must(OpenReadOnly(root))(t)
	defer reader.Close()

	tailPayload := []byte("segmented-backup tail")
	tailID := testcheck.Must(artifact.IdentifyBytes(artifact.KindRun, tailPayload))(t)
	tailDescriptor := artifact.Descriptor{ID: tailID, Size: uint64(len(tailPayload)), MediaType: "text/plain"}
	if _, err := writer.Commit(ctx, artifact.Batch{
		Key:      "segmented/tail",
		Contents: []artifact.Content{{Descriptor: tailDescriptor, Data: tailPayload}},
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := writer.Snapshot(ctx); err != nil {
		t.Fatal(err)
	}
	postPayload := []byte("segmented-backup post-rotation")
	postID := testcheck.Must(artifact.IdentifyBytes(artifact.KindRun, postPayload))(t)
	postDescriptor := artifact.Descriptor{ID: postID, Size: uint64(len(postPayload)), MediaType: "text/plain"}
	if _, err := writer.Commit(ctx, artifact.Batch{
		Key:      "segmented/post-rotation",
		Contents: []artifact.Content{{Descriptor: postDescriptor, Data: postPayload}},
	}); err != nil {
		t.Fatal(err)
	}
	testcheck.NoError(t, reader.Refresh(ctx))
	writerHead, writerSequence := writer.Head()
	if head, sequence := reader.Head(); head != writerHead || sequence != writerSequence {
		t.Fatalf("reader (%s, %d) behind writer (%s, %d) across snapshot rotation", head, sequence, writerHead, writerSequence)
	}

	destination := filepath.Join(t.TempDir(), "segmented-backup")
	backupReport, err := writer.Backup(t.Context(), destination)
	backupHead, backupSequence := backupReport.Head, backupReport.Sequence
	testcheck.NoError(t, err)
	if backupHead != writerHead || backupSequence != writerSequence {
		t.Fatalf("backup (%s, %d) differs from writer (%s, %d)", backupHead, backupSequence, writerHead, writerSequence)
	}
	restored := testcheck.Must(OpenReadOnly(destination))(t)
	defer restored.Close()
	if !restored.SnapshotReplay().Loaded {
		t.Fatal("captured checkpoint anchors did not accelerate the restored open")
	}
	sourceDigest := querySurfaceDigestBounded(t, writer, scaleCorpusCommits)
	if restoredDigest := querySurfaceDigestBounded(t, restored, scaleCorpusCommits); restoredDigest != sourceDigest {
		t.Fatal("restored store answers differently from its source")
	}
	for _, id := range []artifact.ID{tailID, postID} {
		content, found, err := artifact.ReadContent(ctx, restored, id)
		if err != nil || !found || len(content.Data) == 0 {
			t.Fatalf("restored blob content %s = (%v, %v)", id, found, err)
		}
	}
	if _, err := writer.Commit(ctx, artifact.Batch{
		Key: "segmented/after-backup",
		Artifacts: []artifact.Descriptor{{
			ID:        mustScaleID(t, "after-backup"),
			Size:      uint64(len("after-backup")),
			MediaType: "text/plain",
		}},
	}); err != nil {
		t.Fatalf("source writer blocked after backup: %v", err)
	}
}

func mustScaleID(t *testing.T, payload string) artifact.ID {
	t.Helper()
	id := testcheck.Must(artifact.IdentifyBytes(artifact.KindRun, []byte(payload)))(t)
	return id
}
