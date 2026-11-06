package overgodb

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/testcheck"
)

// TestCheckpointGenerationsSurviveATornWrite holds the generations to their
// reason for existing. A newer generation left half written -- one member of
// seven -- is refused and the complete one before it loads, so the open is
// anchored and answers as before; the flat set of a store written before
// generations loads as the oldest generation; and the next complete
// generation retires the torn one and the flat set with it.
func TestCheckpointGenerationsSurviveATornWrite(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	head, _, _ := copyScaleCorpus(t, root)
	anchored := func() string {
		t.Helper()
		store := testcheck.Must(OpenReadOnly(root))(t)
		defer store.Close()
		if replay := store.SnapshotReplay(); !replay.Loaded || replay.Fallback != "" {
			t.Fatalf("open was not anchored at a checkpoint generation: %+v", replay)
		}
		if current, _ := store.Head(); current != head {
			t.Fatalf("head %s, corpus %s", current, head)
		}
		return catalogDigest(t, store.state)
	}
	baseline := anchored()

	complete := newestCheckpoints(t, root)
	parent := filepath.Dir(complete)
	member := testcheck.Must(os.ReadFile(filepath.Join(complete, "aliases"+checkpointExtension)))(t)
	torn := filepath.Join(parent, checkpointGeneration(^uint64(0)))
	testcheck.NoError(t, os.MkdirAll(torn, storeDirectoryMode))
	testcheck.NoError(t, os.WriteFile(filepath.Join(torn, "aliases"+checkpointExtension), member, storeFileMode))
	if digest := anchored(); digest != baseline {
		t.Fatal("the generation before a torn one answered differently")
	}

	members := testcheck.Must(os.ReadDir(complete))(t)
	for _, entry := range members {
		testcheck.NoError(t, os.Rename(filepath.Join(complete, entry.Name()), filepath.Join(parent, entry.Name())))
	}
	testcheck.NoError(t, os.Remove(complete))
	if digest := anchored(); digest != baseline {
		t.Fatal("the flat set of a store written before generations answered differently")
	}

	store := testcheck.Must(Create(root))(t)
	if _, err := store.Snapshot(t.Context()); err != nil {
		t.Fatal(err)
	}
	testcheck.NoError(t, store.Close())
	left, err := os.ReadDir(parent)
	if err != nil || len(left) != 1 || !left[0].IsDir() || filepath.Join(parent, left[0].Name()) == torn {
		t.Fatalf("after a complete generation the checkpoint directory holds %v, %v; want that generation alone", left, err)
	}
	if digest := anchored(); digest != baseline {
		t.Fatal("the new generation answered differently")
	}
}
