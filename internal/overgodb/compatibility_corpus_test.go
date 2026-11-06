package overgodb

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testcheck"
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
	t.Parallel()
	source := t.TempDir()
	_, snapshotPath, _ := copyScaleCorpus(t, source)
	// Rotation empties the active segment at the snapshot boundary;
	// append a post-seal tail so torn-tail and corruption behavior stay
	// exercised against real active frames.
	const corpusTail = 3
	tailWriter := testcheck.Must(Create(source))(t)
	for ordinal := range corpusTail {
		payload := fmt.Appendf(nil, "compatibility-tail/%d", ordinal)
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
	testcheck.NoError(t, tailWriter.Close())
	journal := filepath.Join(source, storeFilename)

	copyCorpus := func(t *testing.T, withSnapshot bool) string {
		t.Helper()
		root := t.TempDir()
		copyCorpusFile(t, journal, filepath.Join(root, storeFilename))
		copyCorpusTree(t, filepath.Join(source, blobDirectory), filepath.Join(root, blobDirectory))
		copyCorpusTree(t, filepath.Join(source, segmentDirectory), filepath.Join(root, segmentDirectory))
		if withSnapshot {
			copyCorpusTree(t, snapshotPath, filepath.Join(root, checkpointDirectory, filepath.Base(snapshotPath)))
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
		store := testcheck.Must(OpenReadOnly(copyCorpus(t, false)))(t)
		defer store.Close()
		if replay := store.SnapshotReplay(); replay.Loaded {
			t.Fatalf("replay used a snapshot that does not exist: %+v", replay)
		}
		requireHead(t, store)
	})

	t.Run("damaged snapshot falls back to the log", func(t *testing.T) {
		root := copyCorpus(t, true)
		damaged := filepath.Join(root, checkpointDirectory, filepath.Base(snapshotPath), "lineage"+checkpointExtension)
		data := testcheck.Must(os.ReadFile(damaged))(t)
		data[len(data)/2] ^= 0xFF
		testcheck.NoError(t, os.WriteFile(damaged, data, storeFileMode))
		store := testcheck.Must(OpenReadOnly(root))(t)
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
		info := testcheck.Must(os.Stat(path))(t)
		testcheck.NoError(t, os.Truncate(path, info.Size()-tornTailBytes))
		store := testcheck.Must(Create(root))(t)
		defer store.Close()
		_, sequence := store.Head()
		if sequence != corpusSequence-1 {
			t.Fatalf("torn tail recovered to sequence %d, want %d", sequence, corpusSequence-1)
		}
		content := []byte("post-recovery commit")
		id := testcheck.Must(artifact.IdentifyBytes(artifact.KindRun, content))(t)
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
		data := testcheck.Must(os.ReadFile(path))(t)
		data[len(data)/2] ^= 0xFF
		testcheck.NoError(t, os.WriteFile(path, data, storeFileMode))
		if _, err := OpenReadOnly(root); err == nil ||
			!strings.Contains(err.Error(), "invalid checksum") && !strings.Contains(err.Error(), "breaks commit chain") {
			t.Fatalf("corrupted log opened: err=%v", err)
		}
	})

	t.Run("read-only refresh observes a live writer", func(t *testing.T) {
		root := copyCorpus(t, true)
		reader := testcheck.Must(OpenReadOnly(root))(t)
		defer reader.Close()
		requireHead(t, reader)
		writer := testcheck.Must(Create(root))(t)
		defer writer.Close()
		content := []byte("refresh-visible commit")
		id := testcheck.Must(artifact.IdentifyBytes(artifact.KindRun, content))(t)
		descriptor := artifact.Descriptor{ID: id, Size: uint64(len(content)), MediaType: "text/plain"}
		written := testcheck.Must(writer.Commit(t.Context(), artifact.Batch{
			Key:       "compatibility/refresh",
			Artifacts: []artifact.Descriptor{descriptor},
			Contents:  []artifact.Content{{Descriptor: descriptor, Data: content}},
		}))(t)
		testcheck.NoError(t, reader.Refresh(t.Context()))
		refreshed, sequence := reader.Head()
		if refreshed != written || sequence != corpusSequence+1 {
			t.Fatalf("refresh saw %s seq %d, writer wrote %s", refreshed, sequence, written)
		}
	})

	t.Run("backup round-trips the head", func(t *testing.T) {
		store := testcheck.Must(Create(copyCorpus(t, true)))(t)
		defer store.Close()
		destination := filepath.Join(t.TempDir(), "backup")
		backupReport, err := store.Backup(t.Context(), destination)
		backupHead, sequence := backupReport.Head, backupReport.Sequence
		testcheck.NoError(t, err)
		if backupHead != head || sequence != corpusSequence {
			t.Fatalf("backup head %s seq %d, corpus %s seq %d", backupHead, sequence, head, corpusSequence)
		}
		restored := testcheck.Must(OpenReadOnly(destination))(t)
		defer restored.Close()
		requireHead(t, restored)
	})

	t.Run("retention release preserves the reachable catalog", func(t *testing.T) {
		store := testcheck.Must(Create(copyCorpus(t, false)))(t)
		defer store.Close()
		ctx := t.Context()
		if _, err := Release(ctx, store, nil, RetentionPolicy{}, releaseEverything); err != nil {
			t.Fatal(err)
		}
		for ordinal := 0; ordinal < scaleCorpusCommits; ordinal += scaleAliasStride {
			id := testcheck.Must(artifact.IdentifyBytes(artifact.KindRun, scaleContent(ordinal)))(t)
			target, found, err := store.ResolveAlias(ctx, aliasName(ordinal))
			if has, hasErr := store.HasContent(ctx, id); err != nil || hasErr != nil || !found || target != id || !has {
				t.Fatalf("released alias %d: found=%v has=%v err=%v/%v", ordinal, found, has, err, hasErr)
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
	testcheck.NoError(t, err)
	testcheck.NoError(t, os.MkdirAll(destination, storeDirectoryMode))
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
	data := testcheck.Must(os.ReadFile(source))(t)
	testcheck.NoError(t, os.WriteFile(destination, data, storeFileMode))
}
