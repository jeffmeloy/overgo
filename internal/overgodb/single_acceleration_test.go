package overgodb

import (
	"os"
	"path/filepath"
	"testing"
)

// TestCheckpointsAreTheOnlyAcceleration holds the store to one acceleration
// format. A maintenance point writes a checkpoint generation, names it as its
// path, and removes the monolithic snapshot an older store left behind; an
// open with no generation ignores whatever sits in the old snapshot
// directory, replays the journal, says why, and answers as the anchored open
// did.
func TestCheckpointsAreTheOnlyAcceleration(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	head, _, _ := copyScaleCorpus(t, root)
	digestOf := func(wantLoaded bool) string {
		t.Helper()
		store, err := OpenReadOnly(root)
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		replay := store.SnapshotReplay()
		if replay.Loaded != wantLoaded || wantLoaded == (replay.Fallback != "") {
			t.Fatalf("replay = %+v, want loaded=%v with a reason only when it is not", replay, wantLoaded)
		}
		if current, _ := store.Head(); current != head {
			t.Fatalf("head %s, corpus %s", current, head)
		}
		return catalogDigest(t, store.state)
	}
	anchored := digestOf(true)

	legacy := filepath.Join(root, snapshotDirectory)
	if err := os.MkdirAll(legacy, storeDirectoryMode); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(legacy, "00000000000000000001.snapshot"), []byte("not a catalog"), storeFileMode); err != nil {
		t.Fatal(err)
	}
	if err := os.RemoveAll(filepath.Join(root, checkpointDirectory)); err != nil {
		t.Fatal(err)
	}
	if replayed := digestOf(false); replayed != anchored {
		t.Fatal("a full replay answered differently from the anchored open")
	}

	store, err := Open(root)
	if err != nil {
		t.Fatal(err)
	}
	info, err := store.Snapshot(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	if info.Path != newestCheckpoints(t, root) {
		t.Fatalf("the maintenance point names %s, the newest generation is %s", info.Path, newestCheckpoints(t, root))
	}
	if _, err := os.Stat(legacy); !os.IsNotExist(err) {
		t.Fatalf("the monolithic snapshot directory survived a maintenance point: %v", err)
	}
	if again := digestOf(true); again != anchored {
		t.Fatal("the generation written after a full replay answered differently")
	}
}
