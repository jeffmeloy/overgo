package overgodb

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
)

func TestCommitDeltaAtReturnsOneExactCanonicalTransaction(t *testing.T) {
	root := t.TempDir()
	store, err := Open(root)
	if err != nil {
		t.Fatal(err)
	}
	first := commitDeltaContent(t, "first")
	firstCommit, err := store.Commit(t.Context(), artifact.Batch{
		Key: "fixture/commit-delta/first", Contents: []artifact.Content{first},
	})
	if err != nil {
		t.Fatal(err)
	}
	record := commitDeltaContent(t, "review record")
	expected := firstCommit
	request := artifact.Batch{
		Key: "fixture/commit-delta/review", ExpectedHead: &expected,
		Contents: []artifact.Content{record},
		Aliases:  []artifact.AliasBinding{{Name: "fixture/review", Target: record.Descriptor.ID}},
	}
	_, requestDigest, err := encodeBatch(request)
	if err != nil {
		t.Fatal(err)
	}
	commit, err := store.Commit(t.Context(), request)
	if err != nil {
		t.Fatal(err)
	}
	for ordinal := range 32 {
		payload := []byte{byte(ordinal)}
		id, err := artifact.IdentifyBytes(artifact.KindFile, payload)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.Commit(t.Context(), artifact.Batch{
			Key:       fmt.Sprintf("fixture/commit-delta/unrelated/%d", ordinal),
			Artifacts: []artifact.Descriptor{{ID: id, Size: uint64(len(payload))}},
		}); err != nil {
			t.Fatal(err)
		}
	}
	assert := func(label string) {
		t.Helper()
		view, found, err := store.CommitDeltaAt(t.Context(), 2)
		if err != nil || !found {
			t.Fatalf("%s exact delta = (found=%t, %v)", label, found, err)
		}
		if view.Commit != (CommitView{Key: request.Key, ID: commit, Sequence: 2}) ||
			view.Previous != firstCommit || view.Request != requestDigest {
			t.Fatalf("%s commit authority = %+v", label, view)
		}
		delta := view.Delta
		if delta.Key != request.Key || delta.ExpectedHead == nil || *delta.ExpectedHead != firstCommit ||
			len(delta.Artifacts) != 1 || delta.Artifacts[0] != record.Descriptor ||
			len(delta.Contents) != 1 || delta.Contents[0].Descriptor != record.Descriptor ||
			string(delta.Contents[0].Data) != string(record.Data) ||
			len(delta.Aliases) != 1 || delta.Aliases[0].Name != "fixture/review" ||
			delta.Aliases[0].Target != record.Descriptor.ID ||
			len(delta.Manifests)+len(delta.Lineage)+len(delta.Causality)+len(delta.Locations) != 0 {
			t.Fatalf("%s canonical delta = %+v", label, delta)
		}
	}
	assert("active")
	if err := store.sealActiveSegment(t.Context()); err != nil {
		t.Fatal(err)
	}
	assert("sealed")
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	store, err = OpenReadOnly(root)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	assert("reopened sealed")
}

func TestCommitDeltaAtDoesNotReadUnrelatedCoordinates(t *testing.T) {
	const unrelatedCommitPopulation = 128

	root := t.TempDir()
	store, err := Open(root)
	if err != nil {
		t.Fatal(err)
	}
	var coordinate commitCoordinate
	for ordinal := range unrelatedCommitPopulation {
		unrelated := commitDeltaContent(t, fmt.Sprintf("unrelated before reviewed commit %d", ordinal))
		if _, err := store.Commit(t.Context(), artifact.Batch{
			Key:      fmt.Sprintf("fixture/commit-delta/unrelated-before/%d", ordinal),
			Contents: []artifact.Content{unrelated},
		}); err != nil {
			t.Fatal(err)
		}
		if ordinal == unrelatedCommitPopulation/2 {
			coordinate = store.state.commits.at(ordinal).coordinate
		}
	}
	reviewed := commitDeltaContent(t, "bounded reviewed commit")
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key: "fixture/commit-delta/bounded", Contents: []artifact.Content{reviewed},
	}); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	store, err = OpenReadOnly(root)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	journal, err := os.OpenFile(filepath.Join(root, storeFilename), os.O_RDWR, storeFileMode)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := journal.WriteAt([]byte{0xff}, coordinate.offset+frameHeaderBytes); err != nil {
		t.Fatal(err)
	}
	if err := journal.Close(); err != nil {
		t.Fatal(err)
	}

	targetSequence := uint64(unrelatedCommitPopulation + 1)
	view, found, err := store.CommitDeltaAt(t.Context(), targetSequence)
	if err != nil || !found || len(view.Delta.Contents) != 1 ||
		view.Delta.Contents[0].Descriptor.ID != reviewed.Descriptor.ID {
		t.Fatalf("bounded exact read = (%+v, found=%t, %v)", view, found, err)
	}
	corruptedSequence := uint64(unrelatedCommitPopulation/2 + 1)
	if _, _, err := store.CommitDeltaAt(t.Context(), corruptedSequence); err == nil {
		t.Fatal("corrupted unrelated coordinate still validated")
	}
}

func commitDeltaContent(t *testing.T, value string) artifact.Content {
	t.Helper()
	payload := []byte(value)
	id, err := artifact.IdentifyBytes(artifact.KindEvidence, payload)
	if err != nil {
		t.Fatal(err)
	}
	return artifact.Content{
		Descriptor: artifact.Descriptor{
			ID: id, Size: uint64(len(payload)), MediaType: "text/plain", Schema: "fixture/commit-delta/v1",
		},
		Data: payload,
	}
}
