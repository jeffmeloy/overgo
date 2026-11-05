package overgodb

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testcheck"
)

// The test's journals: the commits each holds, and the payload a frame whose header alone landed declares.
const (
	tailCommits      = 4
	tailPayloadBytes = 100
	flippedByte      = 0xff
)

// TestAnUnwrittenTailEndsTheJournal: a journal whose end never reached the disk opens at its last whole commit. Its
// frame there may be zeros to the end of the file, a header whose trailer and everything after it are zeros, or, after
// a seal, a first frame of zeros. Open, OpenReadOnly and Create each open such a journal with its head intact, and a
// writer cuts the tail and says how much it cut, where each had refused and repair advised a restore. A tail that is not
// zeros still refuses, and so does a sealed segment that ends in zeros. A journal that never began, a header that did
// not reach the disk and no sealed segment before it, is started by a writer.
func TestAnUnwrittenTailEndsTheJournal(t *testing.T) {
	t.Parallel()
	zeros := make([]byte, frameHeaderBytes+tailPayloadBytes)
	headerAlone := append(encodeFrameHeader(frameVersion, tailCommits+1, artifact.CommitID{}, artifact.CommitID{}, tailPayloadBytes), make([]byte, tailPayloadBytes+4)...)
	for name, tail := range map[string][]byte{"zeros past the last frame": zeros, "a header whose body did not land": headerAlone} {
		// Each open meets the tail on a journal of its own: a writer's cuts it, and a reader leaves it.
		for _, open := range []func(string) (*Store, error){OpenReadOnly, Open, Create} {
			root, head := committedJournal(t, false)
			appendTo(t, filepath.Join(root, storeFilename), tail)
			store, err := open(root)
			if err != nil {
				t.Fatalf("%s: the journal does not open: %v", name, err)
			}
			if got, sequence := store.Head(); got != head || sequence != tailCommits {
				t.Fatalf("%s: the journal opens at %s@%d, want %s@%d", name, got, sequence, head, tailCommits)
			}
			if cut := int64(len(tail)); store.readOnly {
				cut = 0
			} else if torn := store.TornTail(); torn != cut {
				t.Fatalf("%s: the open says it cut %d bytes, want %d", name, torn, cut)
			}
			testcheck.NoError(t, store.Close())
		}
	}

	// After a seal the journal holds its header alone, and a first frame of zeros is no pending rotation.
	root, head := committedJournal(t, true)
	appendTo(t, filepath.Join(root, storeFilename), zeros)
	store, err := Open(root)
	if err != nil {
		t.Fatalf("a sealed store whose journal ends in zeros does not open: %v", err)
	}
	if got, _ := store.Head(); got != head {
		t.Fatalf("the sealed store opens at %s, want %s", got, head)
	}
	testcheck.NoError(t, store.Close())

	// Bytes that are not zeros are damage, and a sealed segment is immutable: both still refuse.
	root, _ = committedJournal(t, false)
	appendTo(t, filepath.Join(root, storeFilename), bytes.Repeat([]byte{flippedByte}, len(zeros)))
	if _, err := Open(root); err == nil {
		t.Fatal("a journal ending in bytes that are not zeros opened")
	}
	root, _ = committedJournal(t, true)
	segments, err := sealedSegmentPaths(root)
	if err != nil || len(segments) == 0 {
		t.Fatalf("no sealed segment: %v", err)
	}
	appendTo(t, segments[len(segments)-1], zeros)
	// Without the checkpoint the seal wrote, which anchors an open past the segments, the open replays them.
	testcheck.NoError(t, os.RemoveAll(filepath.Join(root, checkpointDirectory)))
	if _, err := Open(root); err == nil {
		t.Fatal("a sealed segment ending in zeros opened")
	}

	// A journal whose header never reached the disk, with no sealed segment, was never begun.
	unbegun := t.TempDir()
	testcheck.NoError(t, os.WriteFile(filepath.Join(unbegun, storeFilename), make([]byte, storeHeaderBytes/2), storeFileMode))
	store, err = Create(unbegun)
	if err != nil {
		t.Fatalf("a writer does not start a journal that never began: %v", err)
	}
	if _, sequence := store.Head(); sequence != 0 {
		t.Fatalf("the started journal is at sequence %d", sequence)
	}
	testcheck.NoError(t, store.Close())
}

// committedJournal is a store of tailCommits commits, sealed where asked, and its head.
func committedJournal(t *testing.T, sealed bool) (string, artifact.CommitID) {
	t.Helper()
	root := t.TempDir()
	store, err := Create(root)
	testcheck.NoError(t, err)
	var head artifact.CommitID
	for index := range tailCommits {
		name := fmt.Sprintf("tail-%d", index)
		if head, err = store.Commit(t.Context(), artifact.Batch{Key: "tail/" + name, Artifacts: []artifact.Descriptor{fixtureDescriptor(t, artifact.KindModel, name)}}); err != nil {
			t.Fatal(err)
		}
	}
	if sealed {
		if _, err := store.Snapshot(t.Context()); err != nil {
			t.Fatal(err)
		}
	}
	testcheck.NoError(t, store.Close())
	return root, head
}

// appendTo writes tail at the end of the file at path.
func appendTo(t *testing.T, path string, tail []byte) {
	t.Helper()
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, storeFileMode)
	testcheck.NoError(t, err)
	if _, err := file.Write(tail); err != nil {
		t.Fatal(err)
	}
	testcheck.NoError(t, file.Close())
}
