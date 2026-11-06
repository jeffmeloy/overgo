package overgodb

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestAnOpenNeverStopsShortOfASealedSegment holds an open to the whole journal
// when the checkpoint that should follow a seal never arrived. A seal empties
// the active log, and every generation written at a seal anchors that empty
// log, so an older generation verifies against it and an open would stop at
// that generation's sequence, with the next commit taking a number already
// used. The checkpoint is blocked after the seal, as a failed write or a kill
// between the two leaves it: the open replays the segments, says why, loses
// nothing and goes on from the true head, and Close reports what it could not
// write.
func TestAnOpenNeverStopsShortOfASealedSegment(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	root := filepath.Join(t.TempDir(), "store")
	written := 0
	commit := func(store *Store, count int) {
		t.Helper()
		for range count {
			written++
			id := testutil.ArtifactID(t, artifact.KindEvidence, fmt.Sprintf("record-%d", written))
			if _, err := store.Commit(ctx, artifact.Batch{Key: fmt.Sprintf("seal/%d", written), Artifacts: []artifact.Descriptor{{ID: id}}}); err != nil {
				t.Fatal(err)
			}
		}
	}
	store := testcheck.Must(Create(root))(t)
	commit(store, 20)
	if _, err := store.Snapshot(ctx); err != nil {
		t.Fatal(err)
	}
	commit(store, 20)
	before, _ := store.Head()
	blocked := filepath.Join(root, checkpointDirectory, checkpointGeneration(40))
	testcheck.NoError(t, os.WriteFile(blocked, nil, storeFileMode))
	store.tail = 1
	if err := store.Close(); err == nil {
		t.Fatal("Close did not report the checkpoint it could not write")
	}
	testcheck.NoError(t, os.Remove(blocked))
	reopened := testcheck.Must(Create(root))(t)
	if head, sequence := reopened.Head(); head != before || sequence != 40 {
		t.Fatalf("an open stopped at sequence %d, head %s, after commits through 40, head %s", sequence, head, before)
	}
	if replay := reopened.SnapshotReplay(); replay.Loaded || replay.Fallback == "" {
		t.Fatalf("an open anchored behind a sealed segment: %+v", replay)
	}
	commit(reopened, 1)
	after, sequence := reopened.Head()
	if sequence != 41 {
		t.Fatalf("the next commit took sequence %d, want 41", sequence)
	}
	reopened.tail = ^uint64(0)
	testcheck.NoError(t, reopened.Close())
	testcheck.NoError(t, os.RemoveAll(filepath.Join(root, checkpointDirectory)))
	replayed := testcheck.Must(OpenReadOnly(root))(t)
	defer replayed.Close()
	if head, sequence := replayed.Head(); head != after || sequence != 41 {
		t.Fatalf("a full replay reached sequence %d, head %s, want 41, head %s", sequence, head, after)
	}
}
