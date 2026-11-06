package overgodb

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testcheck"
)

// writeLegacyInlineStore builds a journal in the retired inline layout:
// each frame's envelope lists Content descriptors and the raw bytes
// ride inside the frame payload. This is the immutable external format
// existing stores carry; only tests may ever write it again.
func writeLegacyInlineStore(t *testing.T, root string, commits int) []artifact.ID {
	t.Helper()
	file := testcheck.Must(os.OpenFile(filepath.Join(root, storeFilename), os.O_CREATE|os.O_RDWR, storeFileMode))(t)
	defer file.Close()
	testcheck.NoError(t, writeAll(file, encodeStoreHeader()))
	var previous artifact.CommitID
	var previousArtifact artifact.ID
	ids := make([]artifact.ID, 0, commits)
	for ordinal := range commits {
		payload := fmt.Appendf(nil, "legacy-inline/%d/%s", ordinal, scaleContent(ordinal)[:64])
		id := testcheck.Must(artifact.IdentifyBytes(artifact.KindRun, payload))(t)
		descriptor := artifact.Descriptor{ID: id, Size: uint64(len(payload)), MediaType: "text/plain"}
		batch := artifact.Batch{Key: fmt.Sprintf("legacy/%d", ordinal), Artifacts: []artifact.Descriptor{descriptor}}
		if ordinal%scaleAliasStride == 0 {
			batch.Aliases = []artifact.AliasBinding{{Name: aliasName(ordinal), Target: id}}
		}
		if previousArtifact.Valid() {
			batch.Lineage = []artifact.Lineage{{Child: id, Parent: previousArtifact, Relation: artifact.RelationDerivedFrom}}
		}
		normalized, requestHash, err := encodeBatch(batch)
		testcheck.NoError(t, err)
		normalized.Contents = nil
		metadata := testcheck.Must(json.Marshal(persistedTransaction{
			Request: requestHash, Delta: normalized, Content: []artifact.Descriptor{descriptor},
		}))(t)
		frame := binary.LittleEndian.AppendUint32(nil, uint32(len(metadata)))
		frame = append(frame, metadata...)
		frame = append(frame, payload...)
		id2, header, trailer := encodeFrame(uint64(ordinal)+1, previous, frame)
		testcheck.NoError(t, writeAll(file, header))
		testcheck.NoError(t, writeAll(file, frame))
		testcheck.NoError(t, writeAll(file, trailer[:]))
		previous = id2
		previousArtifact = id
		ids = append(ids, id)
	}
	testcheck.NoError(t, file.Sync())
	return ids
}

// TestLegacyInlineToBlobRebuildRoundTrip pins the two content forms across
// a store's history: a store of inline frames opens and answers reads; a
// new commit of large content onto it is a blob and leaves its bytes out of
// the journal; and rebuild migrates the corpus into a destination that
// places each content by the size rule while artifact, content, alias, and
// lineage semantics match the source exactly.
func TestLegacyInlineToBlobRebuildRoundTrip(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	root := t.TempDir()
	const commits = 24
	ids := writeLegacyInlineStore(t, root, commits)

	store := testcheck.Must(Create(root))(t)
	defer store.Close()
	if _, sequence := store.Head(); sequence != commits {
		t.Fatalf("legacy replay sequence %d, want %d", sequence, commits)
	}
	_, reader, found, err := store.OpenContent(ctx, ids[0])
	if err != nil || !found {
		t.Fatalf("legacy inline content = (%v, %v)", found, err)
	}
	if _, err := io.ReadAll(reader); err != nil {
		t.Fatal(err)
	}

	newPayload := bytes.Repeat([]byte("format-flip"), 1024)
	newID := testcheck.Must(artifact.IdentifyBytes(artifact.KindRun, newPayload))(t)
	newDescriptor := artifact.Descriptor{ID: newID, Size: uint64(len(newPayload)), MediaType: "text/plain"}
	journalBefore := testcheck.Must(os.Stat(filepath.Join(root, storeFilename)))(t)
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "legacy/new-write", Contents: []artifact.Content{{Descriptor: newDescriptor, Data: newPayload}},
	}); err != nil {
		t.Fatal(err)
	}
	if !store.blobs.has(newID) {
		t.Fatal("new write onto a legacy store did not use the blob format")
	}
	journalAfter := testcheck.Must(os.Stat(filepath.Join(root, storeFilename)))(t)
	if growth := journalAfter.Size() - journalBefore.Size(); growth >= int64(len(newPayload)) {
		t.Fatalf("new frame grew the journal %d bytes; inline bytes were written", growth)
	}

	destination := filepath.Join(t.TempDir(), "migrated")
	if _, err := Rebuild(ctx, store, destination, nil, nil, false); err != nil {
		t.Fatal(err)
	}
	migrated := testcheck.Must(OpenReadOnly(destination))(t)
	defer migrated.Close()
	for ordinal, id := range ids {
		descriptor, found, err := migrated.Artifact(ctx, id)
		if err != nil || !found {
			t.Fatalf("migrated artifact %d absent: %v", ordinal, err)
		}
		content, found, err := artifact.ReadContent(ctx, migrated, id)
		if err != nil || !found || content.Descriptor != descriptor {
			t.Fatalf("migrated content %d = (%v, %v)", ordinal, found, err)
		}
		if migrated.blobs.has(id) != (descriptor.Size >= inlineContentLimit) {
			t.Fatalf("migrated content %d of %d bytes is not placed by the size rule", ordinal, descriptor.Size)
		}
		if ordinal%scaleAliasStride == 0 {
			target, found, err := migrated.ResolveAlias(ctx, aliasName(ordinal))
			if err != nil || !found || target != id {
				t.Fatalf("migrated alias %d = (%v, %v)", ordinal, found, err)
			}
		}
		if ordinal > 0 {
			parents, err := migrated.Parents(ctx, id)
			if err != nil || len(parents) != 1 || parents[0].Parent != ids[ordinal-1] {
				t.Fatalf("migrated lineage %d = (%+v, %v)", ordinal, parents, err)
			}
		}
	}
}
