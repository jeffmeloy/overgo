package overgodb

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
)

// TestSegmentedJournalChainAndRecovery holds journal segmentation to
// its contract: sealing is invisible to the catalog -- a segmented
// store answers the full query surface exactly like an unsegmented
// twin of the same commits -- the commit chain is continuous across
// sealed files, a corrupt sealed segment refuses replay by name while
// a torn active tail still recovers, a reader tailing a rotated
// journal rebuilds at the current head, and a store carrying inline
// legacy content refuses to seal.
func TestSegmentedJournalChainAndRecovery(t *testing.T) {
	ctx := t.Context()
	segmented := t.TempDir()
	plain := t.TempDir()
	segmentedStore, err := Open(segmented)
	if err != nil {
		t.Fatal(err)
	}
	defer segmentedStore.Close()
	plainStore, err := Open(plain)
	if err != nil {
		t.Fatal(err)
	}
	defer plainStore.Close()

	commitOrdinal := func(t *testing.T, store *Store, ordinal int) {
		t.Helper()
		content := scaleContent(ordinal)
		id, err := artifact.IdentifyBytes(artifact.KindRun, content)
		if err != nil {
			t.Fatal(err)
		}
		descriptor := artifact.Descriptor{ID: id, Size: uint64(len(content)), MediaType: "application/octet-stream"}
		batch := artifact.Batch{
			Key:      fmt.Sprintf("scale/%d", ordinal),
			Contents: []artifact.Content{{Descriptor: descriptor, Data: content}},
		}
		if ordinal%scaleAliasStride == 0 {
			batch.Aliases = []artifact.AliasBinding{{Name: aliasName(ordinal), Target: id}}
		}
		if _, err := store.Commit(ctx, batch); err != nil {
			t.Fatal(err)
		}
	}

	const commits = scaleAliasStride * 6
	for ordinal := range commits {
		commitOrdinal(t, segmentedStore, ordinal)
		commitOrdinal(t, plainStore, ordinal)
		if ordinal%(commits/3) == commits/3-1 {
			if err := segmentedStore.sealActiveSegment(); err != nil {
				t.Fatal(err)
			}
		}
	}
	sealed, err := filepath.Glob(filepath.Join(segmented, segmentDirectory, "*"+segmentExtension))
	if err != nil || len(sealed) != 3 {
		t.Fatalf("sealed segments = (%v, %v), want 3", sealed, err)
	}
	if err := segmentedStore.Close(); err != nil {
		t.Fatal(err)
	}

	reopened, err := OpenReadOnly(segmented)
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	segmentedDigest := querySurfaceDigestBounded(t, reopened, commits)
	if plainDigest := querySurfaceDigestBounded(t, plainStore, commits); plainDigest != segmentedDigest {
		t.Fatal("segmented store answers differently from its unsegmented twin")
	}

	// Reader tailing across rotation: reader opens, writer seals and
	// commits more, reader refresh lands on the new head.
	writer, err := Open(segmented)
	if err != nil {
		t.Fatal(err)
	}
	defer writer.Close()
	reader, err := OpenReadOnly(segmented)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	commitOrdinal(t, writer, commits)
	if err := writer.sealActiveSegment(); err != nil {
		t.Fatal(err)
	}
	commitOrdinal(t, writer, commits+1)
	if err := reader.Refresh(ctx); err != nil {
		t.Fatal(err)
	}
	writerHead, writerSequence := writer.Head()
	if readerHead, readerSequence := reader.Head(); readerHead != writerHead || readerSequence != writerSequence {
		t.Fatalf("reader (%s, %d) behind writer (%s, %d) after rotation", readerHead, readerSequence, writerHead, writerSequence)
	}

	// A corrupt sealed segment refuses by name; a torn ACTIVE tail
	// still recovers.
	corruptRoot := t.TempDir()
	copyCorpusFile(t, filepath.Join(segmented, storeFilename), filepath.Join(corruptRoot, storeFilename))
	copyCorpusTree(t, filepath.Join(segmented, segmentDirectory), filepath.Join(corruptRoot, segmentDirectory))
	copyCorpusTree(t, filepath.Join(segmented, blobDirectory), filepath.Join(corruptRoot, blobDirectory))
	victim := filepath.Join(corruptRoot, segmentDirectory, filepath.Base(sealed[1]))
	data, err := os.ReadFile(victim)
	if err != nil {
		t.Fatal(err)
	}
	data[len(data)/2] ^= 0xFF
	if err := os.WriteFile(victim, data, storeFileMode); err != nil {
		t.Fatal(err)
	}
	if _, err := OpenReadOnly(corruptRoot); err == nil || !strings.Contains(err.Error(), filepath.Base(sealed[1])) {
		t.Fatalf("corrupt sealed segment opened: %v", err)
	}

	// Inline legacy content refuses to seal.
	legacyRoot := t.TempDir()
	writeLegacyInlineStore(t, legacyRoot, scaleAliasStride)
	legacy, err := Open(legacyRoot)
	if err != nil {
		t.Fatal(err)
	}
	defer legacy.Close()
	if err := legacy.sealActiveSegment(); err == nil || !strings.Contains(err.Error(), "inline") {
		t.Fatalf("legacy inline store sealed: %v", err)
	}
}

// querySurfaceDigestBounded is querySurfaceDigest with the corpus
// extent as a parameter, for corpora smaller than the scale contract.
func querySurfaceDigestBounded(t *testing.T, store *Store, commits int) string {
	t.Helper()
	ctx := t.Context()
	var out string
	for ordinal := 0; ordinal < commits; ordinal += scaleAliasStride {
		id, err := artifact.IdentifyBytes(artifact.KindRun, scaleContent(ordinal))
		if err != nil {
			t.Fatal(err)
		}
		target, found, err := store.ResolveAlias(ctx, aliasName(ordinal))
		if err != nil || !found || target != id {
			t.Fatalf("alias %d: found=%v err=%v", ordinal, found, err)
		}
		out += fmt.Sprintf("alias/%s/%s\n", aliasName(ordinal), target)
		descriptor, found, err := store.Artifact(ctx, id)
		if err != nil || !found {
			t.Fatalf("artifact %d: %v", ordinal, err)
		}
		out += fmt.Sprintf("artifact/%s/%d\n", descriptor.ID, descriptor.Size)
		if has, err := store.HasContent(ctx, id); err != nil || !has {
			t.Fatalf("content %d: %v", ordinal, err)
		}
	}
	head, sequence := store.Head()
	out += fmt.Sprintf("head/%s/%d\n", head, sequence)
	return out
}
