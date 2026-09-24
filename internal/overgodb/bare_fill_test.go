package overgodb

import (
	"bytes"
	"errors"
	"io"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
)

// TestBatchFillsBareRecord holds the store to the one way a declared identity
// gains its facts. A batch that names an identity it does not carry leaves a
// bare record; a later batch carrying the content fills it: the full
// descriptor is served, indexed and read back, after a reopen as well. A bare
// declaration of a known identity changes nothing. A full descriptor that
// arrives without its content, or one that disagrees with a full record, is
// still a conflict.
func TestBatchFillsBareRecord(t *testing.T) {
	t.Parallel()
	root := filepath.Join(t.TempDir(), "store")
	store, err := Open(root)
	if err != nil {
		t.Fatal(err)
	}
	content := retentionContent(t, artifact.KindEvidence, "an environment first declared bare")
	id := content.Descriptor.ID
	bareBatch := func(key string) artifact.Batch {
		return artifact.Batch{Key: key, Artifacts: []artifact.Descriptor{{ID: id}}}
	}
	if _, err := store.Commit(t.Context(), bareBatch("bare/first")); err != nil {
		t.Fatal(err)
	}
	withoutContent := content.Descriptor
	if _, err := store.Commit(t.Context(), artifact.Batch{Key: "full/no-content", Artifacts: []artifact.Descriptor{withoutContent}}); !errors.Is(err, ErrArtifactConflict) {
		t.Fatalf("a full descriptor without its content filled a bare record: %v", err)
	}
	if _, err := store.Commit(t.Context(), artifact.Batch{Key: "full/fill", Artifacts: []artifact.Descriptor{content.Descriptor}, Contents: []artifact.Content{content}}); err != nil {
		t.Fatalf("the content-bearing batch did not fill the bare record: %v", err)
	}
	// A later batch that references the identity bare, as a gate declares
	// the parent it depends on, commits beside its own new content.
	sibling := retentionContent(t, artifact.KindEvidence, "a document naming the environment")
	again := bareBatch("bare/again")
	again.Artifacts = append(again.Artifacts, sibling.Descriptor)
	again.Contents = []artifact.Content{sibling}
	if _, err := store.Commit(t.Context(), again); err != nil {
		t.Fatalf("a bare declaration of a filled identity conflicted: %v", err)
	}
	disagreeing := content.Descriptor
	disagreeing.Schema = "test/other/v1"
	if _, err := store.Commit(t.Context(), artifact.Batch{Key: "full/disagree", Artifacts: []artifact.Descriptor{disagreeing}}); !errors.Is(err, ErrArtifactConflict) {
		t.Fatalf("a disagreeing descriptor replaced a full record: %v", err)
	}
	served := func(store *Store) {
		t.Helper()
		descriptor, found, err := store.Artifact(t.Context(), id)
		if err != nil || !found || descriptor != content.Descriptor {
			t.Fatalf("descriptor = %+v found=%v err=%v", descriptor, found, err)
		}
		if !slices.Contains(store.state.artifacts.bySchema[content.Descriptor.Schema], id) {
			t.Fatal("the filled record is missing from its schema index")
		}
		_, reader, found, err := store.OpenContent(t.Context(), id)
		if err != nil || !found {
			t.Fatalf("content found=%v err=%v", found, err)
		}
		data, err := io.ReadAll(reader)
		if err != nil || !bytes.Equal(data, content.Data) {
			t.Fatalf("content read back %d bytes, %v", len(data), err)
		}
	}
	served(store)
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	reopened, err := OpenReadOnly(root)
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	served(reopened)
}
