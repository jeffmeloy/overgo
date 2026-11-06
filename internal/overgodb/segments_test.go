package overgodb

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testcheck"
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
	t.Parallel()
	ctx := t.Context()
	segmented := t.TempDir()
	plain := t.TempDir()
	segmentedStore := testcheck.Must(Create(segmented))(t)
	defer segmentedStore.Close()
	plainStore := testcheck.Must(Create(plain))(t)
	defer plainStore.Close()

	commitOrdinal := func(t *testing.T, store *Store, ordinal int) {
		t.Helper()
		content := scaleContent(ordinal)
		id := testcheck.Must(artifact.IdentifyBytes(artifact.KindRun, content))(t)
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
			testcheck.NoError(t, segmentedStore.sealActiveSegment(t.Context()))
		}
	}
	sealed, err := filepath.Glob(filepath.Join(segmented, segmentDirectory, "*"+segmentExtension))
	if err != nil || len(sealed) != 3 {
		t.Fatalf("sealed segments = (%v, %v), want 3", sealed, err)
	}
	testcheck.NoError(t, segmentedStore.Close())

	reopened := testcheck.Must(OpenReadOnly(segmented))(t)
	defer reopened.Close()
	segmentedDigest := querySurfaceDigestBounded(t, reopened, commits)
	if plainDigest := querySurfaceDigestBounded(t, plainStore, commits); plainDigest != segmentedDigest {
		t.Fatal("segmented store answers differently from its unsegmented twin")
	}

	// Reader tailing across rotation: reader opens, writer seals and
	// commits more, reader refresh lands on the new head.
	writer := testcheck.Must(Create(segmented))(t)
	defer writer.Close()
	reader := testcheck.Must(OpenReadOnly(segmented))(t)
	defer reader.Close()
	commitOrdinal(t, writer, commits)
	testcheck.NoError(t, writer.sealActiveSegment(t.Context()))
	commitOrdinal(t, writer, commits+1)
	testcheck.NoError(t, reader.Refresh(ctx))
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
	data := testcheck.Must(os.ReadFile(victim))(t)
	data[len(data)/2] ^= 0xFF
	testcheck.NoError(t, os.WriteFile(victim, data, storeFileMode))
	if _, err := OpenReadOnly(corruptRoot); err == nil || !strings.Contains(err.Error(), filepath.Base(sealed[1])) {
		t.Fatalf("corrupt sealed segment opened: %v", err)
	}
}

// TestSealWithInlineContent seals a journal that holds inline content, which
// used to refuse: the bytes leave the journal with their frames, so every
// content must read back identically from the segment -- straight after the
// seal, from a store reopened at the snapshot the seal wrote, from one
// reopened with no snapshot or checkpoint at all, where replay binds each
// locator to the segment its frame lies in, and after a second seal, which
// must leave the first segment's locators where they are.
func TestSealWithInlineContent(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	root := t.TempDir()
	ids := writeLegacyInlineStore(t, root, scaleAliasStride)
	read := func(store *Store, when string) []string {
		t.Helper()
		bodies := make([]string, 0, len(ids))
		for _, id := range ids {
			content, found, err := artifact.ReadContent(ctx, store, id)
			if err != nil || !found {
				t.Fatalf("%s: inline content %s found=%v: %v", when, id, found, err)
			}
			bodies = append(bodies, string(content.Data))
		}
		return bodies
	}
	store := testcheck.Must(Create(root))(t)
	want := read(store, "before the seal")
	if !strings.HasPrefix(want[0], "legacy-inline/0/") {
		t.Fatalf("fixture content = %q", want[0])
	}
	seal := func(when string) {
		t.Helper()
		if _, err := store.Snapshot(ctx); err != nil {
			t.Fatalf("%s: %v", when, err)
		}
		if info, err := os.Stat(filepath.Join(root, storeFilename)); err != nil || info.Size() != storeHeaderBytes {
			t.Fatalf("%s left the journal at %v bytes: %v", when, info, err)
		}
		if got := read(store, when); !slices.Equal(got, want) {
			t.Fatalf("%s changed inline content", when)
		}
	}
	seal("the first seal")
	later := retentionContent(t, artifact.KindOutput, map[string]any{"after": "the seal"})
	if _, err := store.Commit(ctx, artifact.Batch{Key: "inline/after-seal", Contents: []artifact.Content{later}}); err != nil {
		t.Fatal(err)
	}
	seal("the second seal")
	testcheck.NoError(t, store.Close())

	reopen := func(when string) {
		t.Helper()
		reopened, err := OpenReadOnly(root)
		if err != nil {
			t.Fatalf("%s: %v", when, err)
		}
		defer reopened.Close()
		if got := read(reopened, when); !slices.Equal(got, want) {
			t.Fatalf("%s changed inline content", when)
		}
	}
	reopen("a reopen at the snapshot")
	for _, derived := range []string{snapshotDirectory, checkpointDirectory} {
		testcheck.NoError(t, os.RemoveAll(filepath.Join(root, derived)))
	}
	reopen("a reopen by full replay")
}

// querySurfaceDigestBounded is querySurfaceDigest with the corpus
// extent as a parameter, for corpora smaller than the scale contract.
func querySurfaceDigestBounded(t *testing.T, store *Store, commits int) string {
	t.Helper()
	ctx := t.Context()
	var out strings.Builder
	for ordinal := 0; ordinal < commits; ordinal += scaleAliasStride {
		id := testcheck.Must(artifact.IdentifyBytes(artifact.KindRun, scaleContent(ordinal)))(t)
		target, found, err := store.ResolveAlias(ctx, aliasName(ordinal))
		if err != nil || !found || target != id {
			t.Fatalf("alias %d: found=%v err=%v", ordinal, found, err)
		}
		out.WriteString(fmt.Sprintf("alias/%s/%s\n", aliasName(ordinal), target))
		descriptor, found, err := store.Artifact(ctx, id)
		if err != nil || !found {
			t.Fatalf("artifact %d: %v", ordinal, err)
		}
		out.WriteString(fmt.Sprintf("artifact/%s/%d\n", descriptor.ID, descriptor.Size))
		if has, err := store.HasContent(ctx, id); err != nil || !has {
			t.Fatalf("content %d: %v", ordinal, err)
		}
	}
	head, sequence := store.Head()
	out.WriteString(fmt.Sprintf("head/%s/%d\n", head, sequence))
	return out.String()
}

func (s *Store) sealActiveSegment(ctx context.Context) error {
	return s.writeTransaction(ctx, s.sealLocked)
}
