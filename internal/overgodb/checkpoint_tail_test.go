package overgodb

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

// TestAnOpenReplaysNoMoreThanTheBound checks checkpoint maintenance on uncontended close.
//   - Read-only handles never write checkpoints.
//   - A reader held across rotation must match a fresh reader.
func TestAnOpenReplaysNoMoreThanTheBound(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	root := filepath.Join(t.TempDir(), "store")
	const tail = 3
	// write commits so many records through one writer whose tail is the test's, and closes it.
	written := 0
	write := func(records int) {
		t.Helper()
		store, err := Open(root)
		if err != nil {
			t.Fatal(err)
		}
		if store.tail != checkpointTail {
			t.Fatalf("an open gave its handle the tail %d, want the store's own %d", store.tail, checkpointTail)
		}
		store.tail = tail
		for range records {
			written++
			id := testutil.ArtifactID(t, artifact.KindEvidence, fmt.Sprintf("record-%d", written))
			if _, err := store.Commit(ctx, artifact.Batch{Key: fmt.Sprintf("tail/%d", written), Artifacts: []artifact.Descriptor{{ID: id}}}); err != nil {
				t.Fatal(err)
			}
		}
		if err := store.Close(); err != nil {
			t.Fatal(err)
		}
	}
	// read opens the store to read it and returns the handle with how many records it replayed
	// past its checkpoint, and whether it was loaded from one.
	read := func() (*Store, uint64, bool) {
		t.Helper()
		store, err := OpenReadOnly(root)
		if err != nil {
			t.Fatal(err)
		}
		store.tail = 0
		return store, store.sequence - store.anchored, store.SnapshotReplay().Loaded
	}
	generations := func() []string {
		t.Helper()
		entries, err := os.ReadDir(filepath.Join(root, checkpointDirectory))
		if err != nil && !os.IsNotExist(err) {
			t.Fatal(err)
		}
		var names []string
		for _, entry := range entries {
			names = append(names, entry.Name())
		}
		return names
	}

	write(tail)
	held, replayed, anchored := read()
	if anchored || replayed != tail || len(generations()) != 0 {
		t.Fatalf("after a writer closed with a tail of %d the open was anchored: %t, replayed %d records, and the store holds the generations %q; want no checkpoint and the whole journal replayed", tail, anchored, replayed, generations())
	}
	// A handle that only reads writes no checkpoint, with a tail past its own bound of none.
	if err := held.Close(); err != nil || len(generations()) != 0 {
		t.Fatalf("closing a handle that only reads left the generations %q (%v), want none", generations(), err)
	}

	// The reader opens before the writer that crosses the tail, and is held across its checkpoint.
	reader, _, _ := read()
	write(1)
	if got, want := generations(), []string{checkpointGeneration(tail + 1)}; len(got) != 1 || got[0] != want[0] {
		t.Fatalf("after a writer closed with %d records past no checkpoint the store holds the generations %q, want %q", tail+1, got, want)
	}
	fresh, replayed, anchored := read()
	if !anchored || replayed != 0 || fresh.sequence != tail+1 {
		t.Fatalf("the open after the checkpoint was anchored: %t, replayed %d records and stands at %d; want it anchored at %d with nothing to replay", anchored, replayed, fresh.sequence, tail+1)
	}
	if err := reader.Refresh(ctx); err != nil {
		t.Fatal(err)
	}
	if replay := reader.SnapshotReplay(); !replay.Loaded || reader.anchored != tail+1 || reader.sequence != tail+1 {
		t.Fatalf("the reader held across the checkpoint rebuilt as %+v, anchored at %d and standing at %d; want it loaded from the checkpoint at %d", replay, reader.anchored, reader.sequence, tail+1)
	}
	if catalogDigest(t, reader.state) != catalogDigest(t, fresh.state) {
		t.Fatal("the reader held across the checkpoint holds another catalog than a fresh open")
	}
	if err := reader.Close(); err != nil {
		t.Fatal(err)
	}
	if err := fresh.Close(); err != nil {
		t.Fatal(err)
	}

	write(tail - 1)
	last, replayed, anchored := read()
	if !anchored || replayed != tail-1 || len(generations()) != 1 || generations()[0] != checkpointGeneration(tail+1) {
		t.Fatalf("after a writer left %d records past the checkpoint the open was anchored: %t and replayed %d, with the generations %q; want the one checkpoint and %d records replayed", tail-1, anchored, replayed, generations(), tail-1)
	}
	if err := last.Close(); err != nil {
		t.Fatal(err)
	}

	// A writer that wrote a checkpoint itself counts its tail from that one: with one record
	// after it, it closes without writing another, however many it wrote before it.
	store, err := Open(root)
	if err != nil {
		t.Fatal(err)
	}
	store.tail = tail
	for record := range tail {
		id := testutil.ArtifactID(t, artifact.KindEvidence, fmt.Sprintf("before-snapshot-%d", record))
		if _, err := store.Commit(ctx, artifact.Batch{Key: fmt.Sprintf("tail/before-snapshot/%d", record), Artifacts: []artifact.Descriptor{{ID: id}}}); err != nil {
			t.Fatal(err)
		}
	}
	info, err := store.Snapshot(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{Key: "tail/after-snapshot", Artifacts: []artifact.Descriptor{{ID: testutil.ArtifactID(t, artifact.KindEvidence, "after-snapshot")}}}); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	if got := generations(); len(got) != 1 || got[0] != checkpointGeneration(info.Sequence) {
		t.Fatalf("a writer that wrote the checkpoint at %d and one record after it left the generations %q, want that checkpoint alone", info.Sequence, got)
	}
}
