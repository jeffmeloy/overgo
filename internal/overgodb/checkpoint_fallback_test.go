package overgodb

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestProjectionCheckpointAnchorAndFallback holds the checkpoint set
// to its contract: a complete set restores state anchored to the
// journal head; a corrupt payload, an unknown projection version or a
// member anchored to a different head each discard the set with a named
// reason and open falls back to the next verified anchor with identical
// answers. Checkpoints accelerate; they are never authority.
func TestProjectionCheckpointAnchorAndFallback(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	head, _, _ := copyScaleCorpus(t, root)
	directory := filepath.Join(root, checkpointDirectory)

	openAndDigest := func(t *testing.T, expectLoaded bool, expectReason string) string {
		t.Helper()
		store, err := OpenReadOnly(root)
		testcheck.NoError(t, err)
		defer store.Close()
		replay := store.SnapshotReplay()
		if replay.Loaded != expectLoaded {
			t.Fatalf("replay = %+v, want loaded=%v", replay, expectLoaded)
		}
		if expectReason != "" && !strings.Contains(replay.Fallback, expectReason) {
			t.Fatalf("fallback %q does not name %q", replay.Fallback, expectReason)
		}
		if currentHead, _ := store.Head(); currentHead != head {
			t.Fatalf("head %s, corpus %s", currentHead, head)
		}
		return catalogDigest(t, store.state)
	}

	baseline := openAndDigest(t, true, "")

	corrupt := func(t *testing.T, name string, mutate func([]byte) []byte) {
		t.Helper()
		path := filepath.Join(newestCheckpoints(t, root), name+checkpointExtension)
		data, err := os.ReadFile(path)
		testcheck.NoError(t, err)
		testcheck.NoError(t, os.WriteFile(path, mutate(data), storeFileMode))
	}
	restore := func(t *testing.T) {
		t.Helper()
		store, err := Create(root)
		testcheck.NoError(t, err)
		if _, err := store.Snapshot(t.Context()); err != nil {
			t.Fatal(err)
		}
		testcheck.NoError(t, store.Close())
	}

	corrupt(t, "aliases", func(data []byte) []byte {
		data[len(data)-2] ^= 0xFF
		return data
	})
	if digest := openAndDigest(t, false, "checkpoint aliases payload digest differs"); digest != baseline {
		t.Fatal("payload-corrupt fallback answered differently")
	}
	restore(t)

	// An unknown version and an anchor at another head each discard the set, one header field edited.
	for _, edit := range []struct {
		member, reason string
		header         func(*checkpointHeader)
	}{
		{"lineage", "version", func(header *checkpointHeader) { header.Version++ }},
		{"contents", "anchors a different head", func(header *checkpointHeader) { header.Sequence-- }},
	} {
		corrupt(t, edit.member, func(data []byte) []byte {
			line, payload, _ := bytes.Cut(data, []byte{'\n'})
			var header checkpointHeader
			testcheck.NoError(t, json.Unmarshal(line, &header))
			edit.header(&header)
			mutated, err := json.Marshal(header)
			testcheck.NoError(t, err)
			return append(append(mutated, '\n'), payload...)
		})
		if digest := openAndDigest(t, false, edit.reason); digest != baseline {
			t.Fatalf("the fallback for a member that %s answered differently", edit.reason)
		}
		restore(t)
	}

	testcheck.NoError(t, os.RemoveAll(directory))
	if digest := openAndDigest(t, false, "unreadable"); digest != baseline {
		t.Fatal("full-replay fallback answered differently")
	}
}

// newestCheckpoints returns the directory of the store's newest checkpoint
// generation, the one an open loads first.
func newestCheckpoints(t *testing.T, root string) string {
	t.Helper()
	parent := filepath.Join(root, checkpointDirectory)
	entries, err := os.ReadDir(parent)
	testcheck.NoError(t, err)
	for index := len(entries) - 1; index >= 0; index-- {
		if entries[index].IsDir() {
			return filepath.Join(parent, entries[index].Name())
		}
	}
	t.Fatalf("store %s holds no checkpoint generation", root)
	return ""
}
