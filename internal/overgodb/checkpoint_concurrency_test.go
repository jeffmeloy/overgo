package overgodb

import (
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"overgo/internal/testcheck"
)

// TestCheckpointSetLoadsConcurrently holds the checkpoint loader to loading
// its members at once and judging them in registration order. No read
// returns until every member has asked for its own, which a loader that
// takes them one after another never reaches; the state it assembles answers
// as an open does; and with two members corrupt the reason names the one
// registered first, whichever finished first.
func TestCheckpointSetLoadsConcurrently(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	copyScaleCorpus(t, root)
	directory := newestCheckpoints(t, root)
	together := func() func(string) ([]byte, error) {
		var asked sync.WaitGroup
		asked.Add(len(projections(new(catalogState))))
		return func(name string) ([]byte, error) {
			asked.Done()
			asked.Wait()
			return os.ReadFile(filepath.Join(directory, name+checkpointExtension))
		}
	}

	state, anchor, loaded, reason := loadCheckpointSet(together())
	if !loaded {
		t.Fatalf("complete set refused: %s", reason)
	}
	store, err := OpenReadOnly(root)
	testcheck.NoError(t, err)
	defer store.Close()
	if head, _ := store.Head(); anchor.head != head || catalogDigest(t, state) != catalogDigest(t, store.state) {
		t.Fatalf("set anchored at %s answers differently from the store opened at %s", anchor.head, head)
	}

	for _, name := range []string{"commits", "contents"} {
		path := filepath.Join(directory, name+checkpointExtension)
		document, err := os.ReadFile(path)
		testcheck.NoError(t, err)
		document[len(document)-2] ^= 0xFF
		testcheck.NoError(t, os.WriteFile(path, document, storeFileMode))
	}
	if _, _, loaded, reason := loadCheckpointSet(together()); loaded || !strings.Contains(reason, "contents") {
		t.Fatalf("loaded=%v reason %q, want the set refused for contents", loaded, reason)
	}
}
