package overgodb

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/testcheck"
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
		store := testcheck.Must(OpenReadOnly(root))(t)
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
	testcheck.NoError(t, os.MkdirAll(legacy, storeDirectoryMode))
	testcheck.NoError(t, os.WriteFile(filepath.Join(legacy, "00000000000000000001.snapshot"), []byte("not a catalog"), storeFileMode))
	testcheck.NoError(t, os.RemoveAll(filepath.Join(root, checkpointDirectory)))
	if replayed := digestOf(false); replayed != anchored {
		t.Fatal("a full replay answered differently from the anchored open")
	}

	store := testcheck.Must(Create(root))(t)
	info := testcheck.Must(store.Snapshot(t.Context()))(t)
	testcheck.NoError(t, store.Close())
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
