package overgodb

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
)

// TestLegacyStoreCompatibilityCorpus pins every behavior of the current
// disk format that the storage campaign's new layout must preserve, on
// one deterministic corpus: inline-log replay without a snapshot,
// snapshot-anchored replay with fallback on a damaged snapshot,
// torn-tail recovery, refusal of mid-log corruption, read-only refresh
// against a live writer, backup round-trip, and retention compaction.
// Existing stores are an immutable external format boundary; this
// corpus is the contract a migrated reader is judged against.
func TestLegacyStoreCompatibilityCorpus(t *testing.T) {
	source := t.TempDir()
	_, snapshotPath, _ := copyScaleCorpus(t, source)
	// Rotation empties the active segment at the snapshot boundary;
	// append a post-seal tail so torn-tail and corruption behavior stay
	// exercised against real active frames.
	const corpusTail = 3
	tailWriter, err := Open(source)
	if err != nil {
		t.Fatal(err)
	}
	for ordinal := range corpusTail {
		payload := []byte(fmt.Sprintf("compatibility-tail/%d", ordinal))
		id, idErr := artifact.IdentifyBytes(artifact.KindRun, payload)
		if idErr != nil {
			t.Fatal(idErr)
		}
		descriptor := artifact.Descriptor{ID: id, Size: uint64(len(payload)), MediaType: "text/plain"}
		if _, commitErr := tailWriter.Commit(t.Context(), artifact.Batch{
			Key:      fmt.Sprintf("compatibility/tail/%d", ordinal),
			Contents: []artifact.Content{{Descriptor: descriptor, Data: payload}},
		}); commitErr != nil {
			t.Fatal(commitErr)
		}
	}
	head, corpusSequence := tailWriter.Head()
	if err := tailWriter.Close(); err != nil {
		t.Fatal(err)
	}
	journal := filepath.Join(source, storeFilename)

	copyCorpus := func(t *testing.T, withSnapshot bool) string {
		t.Helper()
		root := t.TempDir()
		copyCorpusFile(t, journal, filepath.Join(root, storeFilename))
		copyCorpusTree(t, filepath.Join(source, blobDirectory), filepath.Join(root, blobDirectory))
		copyCorpusTree(t, filepath.Join(source, segmentDirectory), filepath.Join(root, segmentDirectory))
		if withSnapshot {
			if err := os.MkdirAll(filepath.Join(root, snapshotDirectory), storeDirectoryMode); err != nil {
				t.Fatal(err)
			}
			copyCorpusFile(t, snapshotPath, filepath.Join(root, snapshotDirectory, filepath.Base(snapshotPath)))
		}
		return root
	}
	requireHead := func(t *testing.T, store *Store) {
		t.Helper()
		replayed, sequence := store.Head()
		if replayed != head || sequence != corpusSequence {
			t.Fatalf("head %s seq %d, corpus %s seq %d", replayed, sequence, head, corpusSequence)
		}
	}

	t.Run("inline log replays without a snapshot", func(t *testing.T) {
		store, err := OpenReadOnly(copyCorpus(t, false))
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		if replay := store.SnapshotReplay(); replay.Loaded {
			t.Fatalf("replay used a snapshot that does not exist: %+v", replay)
		}
		requireHead(t, store)
	})

	t.Run("damaged snapshot falls back to the log", func(t *testing.T) {
		root := copyCorpus(t, true)
		damaged := filepath.Join(root, snapshotDirectory, filepath.Base(snapshotPath))
		data, err := os.ReadFile(damaged)
		if err != nil {
			t.Fatal(err)
		}
		data[len(data)/2] ^= 0xFF
		if err := os.WriteFile(damaged, data, storeFileMode); err != nil {
			t.Fatal(err)
		}
		store, err := OpenReadOnly(root)
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		replay := store.SnapshotReplay()
		if replay.Loaded || replay.Fallback == "" {
			t.Fatalf("damaged snapshot must fall back with a reason: %+v", replay)
		}
		requireHead(t, store)
	})

	t.Run("torn tail recovers to the last complete commit", func(t *testing.T) {
		root := copyCorpus(t, false)
		path := filepath.Join(root, storeFilename)
		info, err := os.Stat(path)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.Truncate(path, info.Size()-tornTailBytes); err != nil {
			t.Fatal(err)
		}
		store, err := Open(root)
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		_, sequence := store.Head()
		if sequence != corpusSequence-1 {
			t.Fatalf("torn tail recovered to sequence %d, want %d", sequence, corpusSequence-1)
		}
		content := []byte("post-recovery commit")
		id, err := artifact.IdentifyBytes(artifact.KindRun, content)
		if err != nil {
			t.Fatal(err)
		}
		descriptor := artifact.Descriptor{ID: id, Size: uint64(len(content)), MediaType: "text/plain"}
		if _, err := store.Commit(t.Context(), artifact.Batch{
			Key:       "compatibility/post-recovery",
			Artifacts: []artifact.Descriptor{descriptor},
			Contents:  []artifact.Content{{Descriptor: descriptor, Data: content}},
		}); err != nil {
			t.Fatalf("recovered store refuses new commits: %v", err)
		}
	})

	t.Run("mid log corruption is refused, not truncated", func(t *testing.T) {
		root := copyCorpus(t, false)
		path := filepath.Join(root, storeFilename)
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		data[len(data)/2] ^= 0xFF
		if err := os.WriteFile(path, data, storeFileMode); err != nil {
			t.Fatal(err)
		}
		if _, err := OpenReadOnly(root); err == nil ||
			!strings.Contains(err.Error(), "invalid checksum") && !strings.Contains(err.Error(), "breaks commit chain") {
			t.Fatalf("corrupted log opened: err=%v", err)
		}
	})

	t.Run("read-only refresh observes a live writer", func(t *testing.T) {
		root := copyCorpus(t, true)
		reader, err := OpenReadOnly(root)
		if err != nil {
			t.Fatal(err)
		}
		defer reader.Close()
		requireHead(t, reader)
		writer, err := Open(root)
		if err != nil {
			t.Fatal(err)
		}
		defer writer.Close()
		content := []byte("refresh-visible commit")
		id, err := artifact.IdentifyBytes(artifact.KindRun, content)
		if err != nil {
			t.Fatal(err)
		}
		descriptor := artifact.Descriptor{ID: id, Size: uint64(len(content)), MediaType: "text/plain"}
		written, err := writer.Commit(t.Context(), artifact.Batch{
			Key:       "compatibility/refresh",
			Artifacts: []artifact.Descriptor{descriptor},
			Contents:  []artifact.Content{{Descriptor: descriptor, Data: content}},
		})
		if err != nil {
			t.Fatal(err)
		}
		if err := reader.Refresh(t.Context()); err != nil {
			t.Fatal(err)
		}
		refreshed, sequence := reader.Head()
		if refreshed != written || sequence != corpusSequence+1 {
			t.Fatalf("refresh saw %s seq %d, writer wrote %s", refreshed, sequence, written)
		}
	})

	t.Run("backup round-trips the head", func(t *testing.T) {
		store, err := Open(copyCorpus(t, true))
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		destination := filepath.Join(t.TempDir(), "backup")
		backupReport, err := store.Backup(t.Context(), destination)
		backupHead, sequence := backupReport.Head, backupReport.Sequence
		if err != nil {
			t.Fatal(err)
		}
		if backupHead != head || sequence != corpusSequence {
			t.Fatalf("backup head %s seq %d, corpus %s seq %d", backupHead, sequence, head, corpusSequence)
		}
		restored, err := OpenReadOnly(destination)
		if err != nil {
			t.Fatal(err)
		}
		defer restored.Close()
		requireHead(t, restored)
	})

	t.Run("retention compaction preserves the reachable catalog", func(t *testing.T) {
		store, err := Open(copyCorpus(t, false))
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		destination := filepath.Join(t.TempDir(), "compacted")
		if _, err := Compact(t.Context(), store, destination, nil); err != nil {
			t.Fatal(err)
		}
		compacted, err := OpenReadOnly(destination)
		if err != nil {
			t.Fatal(err)
		}
		defer compacted.Close()
		ctx := t.Context()
		for ordinal := 0; ordinal < scaleCorpusCommits; ordinal += scaleAliasStride {
			id, err := artifact.IdentifyBytes(artifact.KindRun, scaleContent(ordinal))
			if err != nil {
				t.Fatal(err)
			}
			target, found, err := compacted.ResolveAlias(ctx, aliasName(ordinal))
			if err != nil || !found || target != id {
				t.Fatalf("compacted alias %d: found=%v err=%v", ordinal, found, err)
			}
		}
	})
}

// copyCorpusTree mirrors the blob tree beside a copied journal; the
// external-blob format makes the pair one corpus.
func copyCorpusTree(t *testing.T, source, destination string) {
	t.Helper()
	entries, err := os.ReadDir(source)
	if os.IsNotExist(err) {
		return
	}
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(destination, storeDirectoryMode); err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		if entry.IsDir() {
			copyCorpusTree(t, filepath.Join(source, entry.Name()), filepath.Join(destination, entry.Name()))
			continue
		}
		copyCorpusFile(t, filepath.Join(source, entry.Name()), filepath.Join(destination, entry.Name()))
	}
}

func copyCorpusFile(t *testing.T, source, destination string) {
	t.Helper()
	data, err := os.ReadFile(source)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(destination, data, storeFileMode); err != nil {
		t.Fatal(err)
	}
}
