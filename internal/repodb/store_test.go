package repodb

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

const (
	fixtureAlias       = "models/current"
	fixtureBatchKey    = "fixture/import/v1"
	fixtureMediaType   = "application/octet-stream"
	fixtureWorkerCount = 12
	tornTailBytes      = 7
)

func fixtureDescriptor(t *testing.T, kind artifact.Kind, payload string) artifact.Descriptor {
	t.Helper()
	id := testutil.ArtifactID(t, kind, payload)
	return artifact.Descriptor{ID: id, Size: uint64(len(payload)), MediaType: fixtureMediaType}
}

func fixtureBatch(t *testing.T) artifact.Batch {
	t.Helper()
	model := fixtureDescriptor(t, artifact.KindModel, "model")
	weights := fixtureDescriptor(t, artifact.KindTensorSet, "weights")
	return artifact.Batch{
		Key:       fixtureBatchKey,
		Artifacts: []artifact.Descriptor{weights, model},
		Lineage: []artifact.Lineage{{
			Child: model.ID, Parent: weights.ID, Relation: artifact.RelationContains,
		}},
		Aliases: []artifact.AliasBinding{{Name: fixtureAlias, Target: model.ID}},
	}
}

func TestCommitReplayAndReadOnlyQueries(t *testing.T) {
	root := t.TempDir()
	store, err := Open(root)
	if err != nil {
		t.Fatal(err)
	}
	batch := fixtureBatch(t)
	commit, err := store.Commit(context.Background(), batch)
	if err != nil {
		t.Fatal(err)
	}
	if !commit.Valid() {
		t.Fatal("commit has zero identity")
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	store, err = OpenReadOnly(root)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	resolved, ok, err := store.ResolveAlias(context.Background(), fixtureAlias)
	if err != nil || !ok || resolved != batch.Aliases[0].Target {
		t.Fatalf("resolve = (%s, %v, %v)", resolved, ok, err)
	}
	parents, err := store.Parents(context.Background(), resolved)
	if err != nil || len(parents) != 1 || parents[0] != batch.Lineage[0] {
		t.Fatalf("parents = (%v, %v)", parents, err)
	}
	children, err := store.Children(context.Background(), batch.Lineage[0].Parent)
	if err != nil || len(children) != 1 || children[0] != batch.Lineage[0] {
		t.Fatalf("children = (%v, %v)", children, err)
	}
	head, sequence := store.Head()
	if head != commit || sequence != 1 {
		t.Fatalf("head = (%s, %d), want (%s, 1)", head, sequence, commit)
	}
	if _, err := store.Commit(context.Background(), batch); !errors.Is(err, ErrReadOnly) {
		t.Fatalf("read-only commit error = %v", err)
	}
}

func TestCommitIsIdempotentByKeyAndContent(t *testing.T) {
	store, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	batch := fixtureBatch(t)
	first, err := store.Commit(context.Background(), batch)
	if err != nil {
		t.Fatal(err)
	}
	second, err := store.Commit(context.Background(), batch)
	if err != nil {
		t.Fatal(err)
	}
	if first != second {
		t.Fatalf("idempotent commits differ: %s != %s", first, second)
	}
	slices.Reverse(batch.Artifacts)
	third, err := store.Commit(context.Background(), batch)
	if err != nil || third != first {
		t.Fatalf("reordered commit = (%s, %v), want (%s, nil)", third, err, first)
	}
	batch.Artifacts[0].Schema = "changed"
	if _, err := store.Commit(context.Background(), batch); !errors.Is(err, ErrBatchKeyConflict) {
		t.Fatalf("changed batch error = %v", err)
	}
	_, sequence := store.Head()
	if sequence != 1 {
		t.Fatalf("sequence = %d, want 1", sequence)
	}
}

func TestFailedBatchPublishesNothing(t *testing.T) {
	store, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	base := fixtureBatch(t)
	if _, err := store.Commit(context.Background(), base); err != nil {
		t.Fatal(err)
	}
	added := fixtureDescriptor(t, artifact.KindAdapter, "adapter")
	wrong := fixtureDescriptor(t, artifact.KindModel, "wrong").ID
	failed := artifact.Batch{
		Key:       "fixture/failed/v1",
		Artifacts: []artifact.Descriptor{added},
		Aliases: []artifact.AliasBinding{{
			Name: fixtureAlias, Target: added.ID, Previous: &wrong,
		}},
	}
	if _, err := store.Commit(context.Background(), failed); !errors.Is(err, ErrAliasConflict) {
		t.Fatalf("failed batch error = %v", err)
	}
	if _, ok, err := store.Artifact(context.Background(), added.ID); err != nil || ok {
		t.Fatalf("failed artifact published: ok=%v err=%v", ok, err)
	}
	_, sequence := store.Head()
	if sequence != 1 {
		t.Fatalf("sequence = %d, want 1", sequence)
	}
}

func TestAliasCompareAndSet(t *testing.T) {
	store, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	base := fixtureBatch(t)
	if _, err := store.Commit(context.Background(), base); err != nil {
		t.Fatal(err)
	}
	next := fixtureDescriptor(t, artifact.KindModel, "next-model")
	previous := base.Aliases[0].Target
	update := artifact.Batch{
		Key:       "fixture/alias/v2",
		Artifacts: []artifact.Descriptor{next},
		Aliases: []artifact.AliasBinding{{
			Name: fixtureAlias, Target: next.ID, Previous: &previous,
		}},
	}
	if _, err := store.Commit(context.Background(), update); err != nil {
		t.Fatal(err)
	}
	resolved, ok, err := store.ResolveAlias(context.Background(), fixtureAlias)
	if err != nil || !ok || resolved != next.ID {
		t.Fatalf("updated alias = (%s, %v, %v)", resolved, ok, err)
	}
}

func TestAliasTransitionRequiresExpectedBinding(t *testing.T) {
	store, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	base := fixtureBatch(t)
	if _, err := store.Commit(context.Background(), base); err != nil {
		t.Fatal(err)
	}

	next := fixtureDescriptor(t, artifact.KindModel, "replacement")
	withoutExpected := artifact.Batch{
		Key:       "fixture/alias/without-expected",
		Artifacts: []artifact.Descriptor{next},
		Aliases:   []artifact.AliasBinding{{Name: fixtureAlias, Target: next.ID}},
	}
	if _, err := store.Commit(context.Background(), withoutExpected); !errors.Is(err, ErrAliasConflict) {
		t.Fatalf("unguarded alias transition error = %v, want ErrAliasConflict", err)
	}
	if _, ok, err := store.Artifact(context.Background(), next.ID); err != nil || ok {
		t.Fatalf("failed transition published immutable content: ok=%v err=%v", ok, err)
	}

	previous := base.Aliases[0].Target
	withExpected := withoutExpected
	withExpected.Key = "fixture/alias/with-expected"
	withExpected.Aliases[0].Previous = &previous
	if _, err := store.Commit(context.Background(), withExpected); err != nil {
		t.Fatal(err)
	}
	resolved, ok, err := store.ResolveAlias(context.Background(), fixtureAlias)
	if err != nil || !ok || resolved != next.ID {
		t.Fatalf("guarded alias transition = (%s, %v, %v)", resolved, ok, err)
	}
}

func TestAliasCompareAndSetRetirement(t *testing.T) {
	store, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	base := fixtureBatch(t)
	if _, err := store.Commit(context.Background(), base); err != nil {
		t.Fatal(err)
	}
	previous := base.Aliases[0].Target
	retire := artifact.Batch{Key: "fixture/alias/retire", Aliases: []artifact.AliasBinding{{
		Name: fixtureAlias, Target: previous, Previous: &previous, Remove: true,
	}}}
	if _, err := store.Commit(context.Background(), retire); err != nil {
		t.Fatal(err)
	}
	if resolved, ok, err := store.ResolveAlias(context.Background(), fixtureAlias); err != nil || ok {
		t.Fatalf("retired alias = (%s, %v, %v)", resolved, ok, err)
	}
}

func TestLineageCycleRejected(t *testing.T) {
	store, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	base := fixtureBatch(t)
	if _, err := store.Commit(context.Background(), base); err != nil {
		t.Fatal(err)
	}
	reverse := artifact.Batch{
		Key: "fixture/cycle/v1",
		Lineage: []artifact.Lineage{{
			Child: base.Lineage[0].Parent, Parent: base.Lineage[0].Child,
			Relation: artifact.RelationDerivedFrom,
		}},
	}
	if _, err := store.Commit(context.Background(), reverse); !errors.Is(err, ErrLineageCycle) {
		t.Fatalf("cycle error = %v", err)
	}
}

func TestLineageCycleWithinBatchRejected(t *testing.T) {
	store, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	first := fixtureDescriptor(t, artifact.KindModel, "cycle-first")
	second := fixtureDescriptor(t, artifact.KindModel, "cycle-second")
	batch := artifact.Batch{
		Key:       "fixture/cycle-within-batch/v1",
		Artifacts: []artifact.Descriptor{first, second},
		Lineage: []artifact.Lineage{
			{Child: first.ID, Parent: second.ID, Relation: artifact.RelationDerivedFrom},
			{Child: second.ID, Parent: first.ID, Relation: artifact.RelationDerivedFrom},
		},
	}
	if _, err := store.Commit(context.Background(), batch); !errors.Is(err, ErrLineageCycle) {
		t.Fatalf("cycle error = %v", err)
	}
}

func TestSecondWriterFailsClosed(t *testing.T) {
	root := t.TempDir()
	first, err := Open(root)
	if err != nil {
		t.Fatal(err)
	}
	if second, err := Open(root); err == nil {
		_ = second.Close()
		t.Fatal("second writer acquired lock")
	}
	if err := first.Close(); err != nil {
		t.Fatal(err)
	}
	second, err := Open(root)
	if err != nil {
		t.Fatalf("writer after release: %v", err)
	}
	_ = second.Close()
}

func TestReadOnlyStoreCanOpenBesideWriter(t *testing.T) {
	root := t.TempDir()
	writer, err := Open(root)
	if err != nil {
		t.Fatal(err)
	}
	defer writer.Close()
	batch := fixtureBatch(t)
	if _, err := writer.Commit(context.Background(), batch); err != nil {
		t.Fatal(err)
	}
	reader, err := OpenReadOnly(root)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	if _, ok, err := reader.Artifact(context.Background(), batch.Artifacts[0].ID); err != nil || !ok {
		t.Fatalf("read beside writer = (%v, %v)", ok, err)
	}
}

func TestTornTailRecovered(t *testing.T) {
	root := t.TempDir()
	store, err := Open(root)
	if err != nil {
		t.Fatal(err)
	}
	batch := fixtureBatch(t)
	if _, err := store.Commit(context.Background(), batch); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(root, storeFilename)
	before, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, storeFileMode)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := file.Write(make([]byte, tornTailBytes)); err != nil {
		t.Fatal(err)
	}
	_ = file.Close()
	store, err = Open(root)
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	after, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if after.Size() != before.Size() {
		t.Fatalf("recovered size = %d, want %d", after.Size(), before.Size())
	}
}

func TestCompleteCorruptFrameRejected(t *testing.T) {
	root := t.TempDir()
	store, err := Open(root)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(context.Background(), fixtureBatch(t)); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(root, storeFilename)
	file, err := os.OpenFile(path, os.O_RDWR, storeFileMode)
	if err != nil {
		t.Fatal(err)
	}
	info, err := file.Stat()
	if err != nil {
		t.Fatal(err)
	}
	offset := info.Size() - frameChecksumSize - 1
	value := []byte{0}
	if _, err := file.ReadAt(value, offset); err != nil {
		t.Fatal(err)
	}
	value[0] ^= 0xff
	if _, err := file.WriteAt(value, offset); err != nil {
		t.Fatal(err)
	}
	_ = file.Close()
	if reopened, err := Open(root); err == nil {
		_ = reopened.Close()
		t.Fatal("corrupt complete frame accepted")
	}
}

func TestLegacyArtifactFrameReplays(t *testing.T) {
	root := t.TempDir()
	batch := fixtureBatch(t)
	payload, _, _, err := encodeBatch(batch)
	if err != nil {
		t.Fatal(err)
	}
	_, frame := encodeRecordVersion(minimumFrameVersion, 1, artifact.CommitID{}, payload)
	data := append(encodeStoreHeader(), frame...)
	if err := os.WriteFile(filepath.Join(root, storeFilename), data, storeFileMode); err != nil {
		t.Fatal(err)
	}
	store, err := OpenReadOnly(root)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, ok, err := store.Artifact(context.Background(), batch.Artifacts[0].ID); err != nil || !ok {
		t.Fatalf("legacy artifact = (%v, %v)", ok, err)
	}
}

func TestConcurrentCommitsSerialize(t *testing.T) {
	store, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	var wait sync.WaitGroup
	errorsByWorker := make(chan error, fixtureWorkerCount)
	for worker := 0; worker < fixtureWorkerCount; worker++ {
		wait.Add(1)
		go func(index int) {
			defer wait.Done()
			payload := fmt.Sprintf("output-%d", index)
			id, err := artifact.IdentifyBytes(artifact.KindOutput, []byte(payload))
			if err != nil {
				errorsByWorker <- err
				return
			}
			descriptor := artifact.Descriptor{ID: id, Size: uint64(len(payload)), MediaType: fixtureMediaType}
			_, err = store.Commit(context.Background(), artifact.Batch{
				Key:       fmt.Sprintf("fixture/worker/%d", index),
				Artifacts: []artifact.Descriptor{descriptor},
			})
			errorsByWorker <- err
		}(worker)
	}
	wait.Wait()
	close(errorsByWorker)
	for err := range errorsByWorker {
		if err != nil {
			t.Fatal(err)
		}
	}
	_, sequence := store.Head()
	if sequence != fixtureWorkerCount {
		t.Fatalf("sequence = %d, want %d", sequence, fixtureWorkerCount)
	}
}
