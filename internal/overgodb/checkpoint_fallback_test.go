package overgodb

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestProjectionCheckpointAnchorAndFallback holds the checkpoint set
// to its contract: a complete set restores state anchored to the
// journal head; a corrupt payload, an unknown projection version, a
// member anchored to a different head, or a non-canonical body each
// discard the set with a named reason and open falls back to the next
// verified anchor with identical answers. Checkpoints accelerate;
// they are never authority.
func TestProjectionCheckpointAnchorAndFallback(t *testing.T) {
	root := t.TempDir()
	head, _, _, _ := buildScaleCorpus(t, root)
	directory := filepath.Join(root, checkpointDirectory)

	openAndDigest := func(t *testing.T, expectLoaded bool, expectReason string) string {
		t.Helper()
		store, err := OpenReadOnly(root)
		if err != nil {
			t.Fatal(err)
		}
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
		path := filepath.Join(directory, name+checkpointExtension)
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, mutate(data), storeFileMode); err != nil {
			t.Fatal(err)
		}
	}
	restore := func(t *testing.T) {
		t.Helper()
		store, err := Open(root)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.Snapshot(t.Context()); err != nil {
			t.Fatal(err)
		}
		if err := store.Close(); err != nil {
			t.Fatal(err)
		}
	}

	corrupt(t, "aliases", func(data []byte) []byte {
		data[len(data)-2] ^= 0xFF
		return data
	})
	if digest := openAndDigest(t, true, "aliases"); digest != baseline {
		t.Fatal("payload-corrupt fallback answered differently")
	}
	restore(t)

	corrupt(t, "lineage", func(data []byte) []byte {
		newline := 0
		for index, b := range data {
			if b == '\n' {
				newline = index
				break
			}
		}
		var header checkpointHeader
		if err := json.Unmarshal(data[:newline], &header); err != nil {
			t.Fatal(err)
		}
		header.Version++
		mutated, err := json.Marshal(header)
		if err != nil {
			t.Fatal(err)
		}
		return append(append(mutated, '\n'), data[newline+1:]...)
	})
	if digest := openAndDigest(t, true, "version"); digest != baseline {
		t.Fatal("unknown-version fallback answered differently")
	}
	restore(t)

	corrupt(t, "contents", func(data []byte) []byte {
		newline := 0
		for index, b := range data {
			if b == '\n' {
				newline = index
				break
			}
		}
		var header checkpointHeader
		if err := json.Unmarshal(data[:newline], &header); err != nil {
			t.Fatal(err)
		}
		header.Sequence--
		mutated, err := json.Marshal(header)
		if err != nil {
			t.Fatal(err)
		}
		return append(append(mutated, '\n'), data[newline+1:]...)
	})
	if digest := openAndDigest(t, true, "anchors a different head"); digest != baseline {
		t.Fatal("stale-anchor fallback answered differently")
	}
	restore(t)

	if err := os.RemoveAll(directory); err != nil {
		t.Fatal(err)
	}
	if err := os.RemoveAll(filepath.Join(root, snapshotDirectory)); err != nil {
		t.Fatal(err)
	}
	if digest := openAndDigest(t, false, "unreadable"); digest != baseline {
		t.Fatal("full-replay fallback answered differently")
	}
}
