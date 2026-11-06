package overgodb

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/processlock"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

const (
	fixtureAlias       = "models/current"
	fixtureBatchKey    = "fixture/import/v1"
	fixtureMediaType   = "application/octet-stream"
	fixturePayload     = "canonical-content"
	fixtureWorkerCount = 12
	tornTailBytes      = 7
)

func TestCanonicalCatalogOwnership(t *testing.T) {
	t.Parallel()
	state := newCatalogState()
	if state.artifacts.records == nil || state.artifacts.byMedia == nil || state.artifacts.bySchema == nil ||
		state.contents.locators == nil || state.lineage.parents == nil || state.locations.byArtifact == nil ||
		state.aliases.bindings == nil || state.commits.byKey == nil {
		t.Fatal("catalog facet indexes are not initialized")
	}
}

func TestOvergoDBRuntimeAuthority(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	legacy := filepath.Join(root, "repodb.log")
	testcheck.NoError(t, os.WriteFile(legacy, []byte("external legacy input"), storeFileMode))
	store := testcheck.Must(Create(root))(t)
	testcheck.NoError(t, store.Close())
	if _, err := os.Stat(filepath.Join(root, storeFilename)); err != nil {
		t.Fatalf("current store log is absent: %v", err)
	}
	data := testcheck.Must(os.ReadFile(legacy))(t)
	if string(data) != "external legacy input" {
		t.Fatalf("legacy import source was modified: %q", data)
	}
}

func TestLineageIndexesReferenceCanonicalEdges(t *testing.T) {
	t.Parallel()
	state := newCatalogState()
	batch := fixtureBatch(t)
	state.apply(batch, nil, 1)
	edge := batch.Lineage[0]
	key := relationKey{child: edge.Child, parent: edge.Parent, relation: edge.Relation}
	if _, ok := slices.BinarySearchFunc(state.lineage.parentsOf(edge.Child), key, compareRelation); !ok {
		t.Fatal("parent index lacks canonical edge key")
	}
	if _, ok := slices.BinarySearchFunc(state.lineage.childrenOf(edge.Parent), key, compareRelation); !ok {
		t.Fatal("child index lacks canonical edge key")
	}
}

func TestContentUsesDescriptorAuthority(t *testing.T) {
	t.Parallel()
	store := testcheck.Must(Open(t.TempDir()))(t)
	defer store.Close()
	descriptor := fixtureDescriptor(t, artifact.KindOutput, fixturePayload)
	content := artifact.Content{Descriptor: descriptor, Data: []byte(fixturePayload)}
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key: "fixture/content-authority/v1", Contents: []artifact.Content{content},
	}); err != nil {
		t.Fatal(err)
	}
	descriptor.Schema = "fixture/canonical/v1"
	record, _ := store.state.artifacts.record(descriptor.ID)
	record.descriptor = descriptor
	got, ok, err := artifact.ReadContent(t.Context(), store, descriptor.ID)
	if err != nil || !ok || got.Descriptor != descriptor || !slices.Equal(got.Data, content.Data) {
		t.Fatalf("content = (%+v, %v, %v)", got, ok, err)
	}
}

func TestMetadataSnapshot(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	store := testcheck.Must(Create(root))(t)
	descriptor := fixtureDescriptor(t, artifact.KindEvidence, fixturePayload)
	content := artifact.Content{Descriptor: descriptor, Data: []byte(fixturePayload)}
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key: "fixture/lazy-content/v1", Contents: []artifact.Content{content},
	}); err != nil {
		t.Fatal(err)
	}
	locator, _ := store.state.contents.locator(descriptor.ID)
	if locator.size != int64(len(content.Data)) {
		t.Fatalf("content locator = %+v", locator)
	}
	snapshot := testcheck.Must(store.Snapshot(t.Context()))(t)
	members, err := os.ReadDir(snapshot.Path)
	if err != nil || len(members) == 0 {
		t.Fatalf("checkpoint generation %s: %d members, %v", snapshot.Path, len(members), err)
	}
	for _, member := range members {
		memberData := testcheck.Must(os.ReadFile(filepath.Join(snapshot.Path, member.Name())))(t)
		if bytes.Contains(memberData, content.Data) {
			t.Fatalf("checkpoint %s retained content payload", member.Name())
		}
	}
	testcheck.NoError(t, store.Close())
	store = testcheck.Must(OpenReadOnly(root))(t)
	defer store.Close()
	if status := store.SnapshotReplay(); !status.Loaded || status.Path != filepath.Join(root, checkpointDirectory) || status.Fallback != "" {
		t.Fatalf("snapshot replay = %+v", status)
	}
	gotDescriptor, reader, found, err := store.OpenContent(t.Context(), descriptor.ID)
	if err != nil || !found || gotDescriptor != descriptor {
		t.Fatalf("open content = (%+v, %v, %v)", gotDescriptor, found, err)
	}
	data, err := io.ReadAll(reader)
	if err != nil || !slices.Equal(data, content.Data) {
		t.Fatalf("streamed content = (%q, %v)", data, err)
	}
}

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
	t.Parallel()
	root := t.TempDir()
	store := testcheck.Must(Create(root))(t)
	batch := fixtureBatch(t)
	commit := testcheck.Must(store.Commit(t.Context(), batch))(t)
	if !commit.Valid() {
		t.Fatal("commit has zero identity")
	}
	testcheck.NoError(t, store.Close())

	store = testcheck.Must(OpenReadOnly(root))(t)
	defer store.Close()
	resolved, ok, err := store.ResolveAlias(t.Context(), fixtureAlias)
	if err != nil || !ok || resolved != batch.Aliases[0].Target {
		t.Fatalf("resolve = (%s, %v, %v)", resolved, ok, err)
	}
	parents, err := store.Parents(t.Context(), resolved)
	if err != nil || len(parents) != 1 || parents[0] != batch.Lineage[0] {
		t.Fatalf("parents = (%v, %v)", parents, err)
	}
	children, err := store.Children(t.Context(), batch.Lineage[0].Parent)
	if err != nil || len(children) != 1 || children[0] != batch.Lineage[0] {
		t.Fatalf("children = (%v, %v)", children, err)
	}
	head, sequence := store.Head()
	if head != commit || sequence != 1 {
		t.Fatalf("head = (%s, %d), want (%s, 1)", head, sequence, commit)
	}
	if _, err := store.Commit(t.Context(), batch); !errors.Is(err, ErrReadOnly) {
		t.Fatalf("read-only commit error = %v", err)
	}
}

func TestReadOnlyRefreshAppliesCommittedTail(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	writer := testcheck.Must(Create(root))(t)
	defer writer.Close()
	if _, err := writer.Commit(t.Context(), fixtureBatch(t)); err != nil {
		t.Fatal(err)
	}
	reader := testcheck.Must(OpenReadOnly(root))(t)
	defer reader.Close()

	descriptor := fixtureDescriptor(t, artifact.KindOutput, fixturePayload)
	if _, err := writer.Commit(t.Context(), artifact.Batch{
		Key: "fixture/refresh/v1", Artifacts: []artifact.Descriptor{descriptor},
	}); err != nil {
		t.Fatal(err)
	}
	if _, found, err := reader.Artifact(t.Context(), descriptor.ID); err != nil || found {
		t.Fatalf("artifact before refresh = (%v, %v)", found, err)
	}
	testcheck.NoError(t, reader.Refresh(t.Context()))
	if got, found, err := reader.Artifact(t.Context(), descriptor.ID); err != nil || !found || got != descriptor {
		t.Fatalf("artifact after refresh = (%+v, %v, %v)", got, found, err)
	}
}

func TestReadOnlyRefreshRejectsFork(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	writer := testcheck.Must(Create(root))(t)
	if _, err := writer.Commit(t.Context(), fixtureBatch(t)); err != nil {
		t.Fatal(err)
	}
	testcheck.NoError(t, writer.Close())
	reader := testcheck.Must(OpenReadOnly(root))(t)
	defer reader.Close()
	head, sequence := reader.Head()
	descriptor := fixtureDescriptor(t, artifact.KindOutput, fixturePayload)
	forkBatch, _, err := encodeBatch(artifact.Batch{
		Key: "fixture/fork/v1", Artifacts: []artifact.Descriptor{descriptor},
	})
	testcheck.NoError(t, err)
	payload := testcheck.Must(json.Marshal(forkBatch))(t)
	_, header, trailer := encodeFrame(sequence+1, artifact.CommitID{}, payload)
	frame := append(header, payload...)
	frame = append(frame, trailer[:]...)
	file := testcheck.Must(os.OpenFile(filepath.Join(root, storeFilename), os.O_APPEND|os.O_WRONLY, storeFileMode))(t)
	if _, err := file.Write(frame); err != nil {
		_ = file.Close()
		t.Fatal(err)
	}
	testcheck.NoError(t, file.Close())
	if err := reader.Refresh(t.Context()); err == nil {
		t.Fatal("forked tail accepted")
	}
	if got, gotSequence := reader.Head(); got != head || gotSequence != sequence {
		t.Fatalf("head after fork = (%s, %d), want (%s, %d)", got, gotSequence, head, sequence)
	}
}

func TestCommitIsIdempotentByKeyAndContent(t *testing.T) {
	t.Parallel()
	store := testcheck.Must(Open(t.TempDir()))(t)
	defer store.Close()
	batch := fixtureBatch(t)
	first := testcheck.Must(store.Commit(t.Context(), batch))(t)
	second := testcheck.Must(store.Commit(t.Context(), batch))(t)
	if first != second {
		t.Fatalf("idempotent commits differ: %s != %s", first, second)
	}
	slices.Reverse(batch.Artifacts)
	third, err := store.Commit(t.Context(), batch)
	if err != nil || third != first {
		t.Fatalf("reordered commit = (%s, %v), want (%s, nil)", third, err, first)
	}
	batch.Artifacts[0].Schema = "changed"
	if _, err := store.Commit(t.Context(), batch); !errors.Is(err, ErrBatchKeyConflict) {
		t.Fatalf("changed batch error = %v", err)
	}
	_, sequence := store.Head()
	if sequence != 1 {
		t.Fatalf("sequence = %d, want 1", sequence)
	}
}

func TestCommitDeltaRetainsRequestIdentity(t *testing.T) {
	t.Parallel()
	store := testcheck.Must(Open(t.TempDir()))(t)
	defer store.Close()
	base := fixtureBatch(t)
	if _, err := store.Commit(t.Context(), base); err != nil {
		t.Fatal(err)
	}
	added := fixtureDescriptor(t, artifact.KindOutput, "delta-output")
	request := artifact.Batch{
		Key:       "fixture/delta/v1",
		Artifacts: append(slices.Clone(base.Artifacts), added),
		Lineage:   slices.Clone(base.Lineage),
	}
	normalized, requestDigest, err := encodeBatch(request)
	testcheck.NoError(t, err)
	delta := store.state.delta(normalized)
	if !slices.Equal(delta.Artifacts, []artifact.Descriptor{added}) || len(delta.Lineage) != 0 {
		t.Fatalf("delta = %+v", delta)
	}
	if _, err := store.Commit(t.Context(), request); err != nil {
		t.Fatal(err)
	}
	if committed, found := store.state.commit(request.Key); !found || committed.payload != requestDigest {
		t.Fatalf("request digest = %x, want %x", committed.payload, requestDigest)
	}
}

func TestRepeatedFactsDoNotAdvance(t *testing.T) {
	t.Parallel()
	store := testcheck.Must(Open(t.TempDir()))(t)
	defer store.Close()
	base := fixtureBatch(t)
	if _, err := store.Commit(t.Context(), base); err != nil {
		t.Fatal(err)
	}
	head, sequence := store.Head()
	previous := base.Aliases[0].Target
	repeated := base
	repeated.Key = "fixture/repeated/v1"
	repeated.Aliases = []artifact.AliasBinding{{
		Name: fixtureAlias, Target: previous, Previous: &previous,
	}}
	if _, err := store.Commit(t.Context(), repeated); !errors.Is(err, ErrNoChange) {
		t.Fatalf("repeated commit error = %v, want ErrNoChange", err)
	}
	if current, currentSequence := store.Head(); current != head || currentSequence != sequence {
		t.Fatalf("head after no-op = (%s, %d), want (%s, %d)", current, currentSequence, head, sequence)
	}
}

func TestFailedBatchPublishesNothing(t *testing.T) {
	t.Parallel()
	store := testcheck.Must(Open(t.TempDir()))(t)
	defer store.Close()
	base := fixtureBatch(t)
	if _, err := store.Commit(t.Context(), base); err != nil {
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
	if _, err := store.Commit(t.Context(), failed); !errors.Is(err, ErrAliasConflict) {
		t.Fatalf("failed batch error = %v", err)
	}
	if _, ok, err := store.Artifact(t.Context(), added.ID); err != nil || ok {
		t.Fatalf("failed artifact published: ok=%v err=%v", ok, err)
	}
	_, sequence := store.Head()
	if sequence != 1 {
		t.Fatalf("sequence = %d, want 1", sequence)
	}
}

func TestAliasCompareAndSet(t *testing.T) {
	t.Parallel()
	store := testcheck.Must(Open(t.TempDir()))(t)
	defer store.Close()
	base := fixtureBatch(t)
	if _, err := store.Commit(t.Context(), base); err != nil {
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
	if _, err := store.Commit(t.Context(), update); err != nil {
		t.Fatal(err)
	}
	resolved, ok, err := store.ResolveAlias(t.Context(), fixtureAlias)
	if err != nil || !ok || resolved != next.ID {
		t.Fatalf("updated alias = (%s, %v, %v)", resolved, ok, err)
	}
}

func TestAliasTransitionRequiresExpectedBinding(t *testing.T) {
	t.Parallel()
	store := testcheck.Must(Open(t.TempDir()))(t)
	defer store.Close()
	base := fixtureBatch(t)
	if _, err := store.Commit(t.Context(), base); err != nil {
		t.Fatal(err)
	}

	next := fixtureDescriptor(t, artifact.KindModel, "replacement")
	withoutExpected := artifact.Batch{
		Key:       "fixture/alias/without-expected",
		Artifacts: []artifact.Descriptor{next},
		Aliases:   []artifact.AliasBinding{{Name: fixtureAlias, Target: next.ID}},
	}
	if _, err := store.Commit(t.Context(), withoutExpected); !errors.Is(err, ErrAliasConflict) {
		t.Fatalf("unguarded alias transition error = %v, want ErrAliasConflict", err)
	}
	if _, ok, err := store.Artifact(t.Context(), next.ID); err != nil || ok {
		t.Fatalf("failed transition published immutable content: ok=%v err=%v", ok, err)
	}

	previous := base.Aliases[0].Target
	withExpected := withoutExpected
	withExpected.Key = "fixture/alias/with-expected"
	withExpected.Aliases[0].Previous = &previous
	if _, err := store.Commit(t.Context(), withExpected); err != nil {
		t.Fatal(err)
	}
	resolved, ok, err := store.ResolveAlias(t.Context(), fixtureAlias)
	if err != nil || !ok || resolved != next.ID {
		t.Fatalf("guarded alias transition = (%s, %v, %v)", resolved, ok, err)
	}
}

func TestAliasCompareAndSetRetirement(t *testing.T) {
	t.Parallel()
	store := testcheck.Must(Open(t.TempDir()))(t)
	defer store.Close()
	base := fixtureBatch(t)
	if _, err := store.Commit(t.Context(), base); err != nil {
		t.Fatal(err)
	}
	previous := base.Aliases[0].Target
	retire := artifact.Batch{Key: "fixture/alias/retire", Aliases: []artifact.AliasBinding{{
		Name: fixtureAlias, Target: previous, Previous: &previous, Remove: true,
	}}}
	if _, err := store.Commit(t.Context(), retire); err != nil {
		t.Fatal(err)
	}
	if resolved, ok, err := store.ResolveAlias(t.Context(), fixtureAlias); err != nil || ok {
		t.Fatalf("retired alias = (%s, %v, %v)", resolved, ok, err)
	}
}

func TestLineageCycleRejected(t *testing.T) {
	t.Parallel()
	store := testcheck.Must(Open(t.TempDir()))(t)
	defer store.Close()
	base := fixtureBatch(t)
	if _, err := store.Commit(t.Context(), base); err != nil {
		t.Fatal(err)
	}
	reverse := artifact.Batch{
		Key: "fixture/cycle/v1",
		Lineage: []artifact.Lineage{{
			Child: base.Lineage[0].Parent, Parent: base.Lineage[0].Child,
			Relation: artifact.RelationDerivedFrom,
		}},
	}
	if _, err := store.Commit(t.Context(), reverse); !errors.Is(err, ErrLineageCycle) {
		t.Fatalf("cycle error = %v", err)
	}
}

func TestLineageCycleWithinBatchRejected(t *testing.T) {
	t.Parallel()
	store := testcheck.Must(Open(t.TempDir()))(t)
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
	if _, err := store.Commit(t.Context(), batch); !errors.Is(err, ErrLineageCycle) {
		t.Fatalf("cycle error = %v", err)
	}
}

func TestSecondWriterFailsClosed(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	first := testcheck.Must(Create(root))(t)
	defer first.Close()
	lock := testcheck.Must(processlock.Acquire(writerLockPath(root), storeFileMode))(t)
	defer lock.Close()
	if second, err := Create(root); err == nil {
		_ = second.Close()
		t.Fatal("second writer acquired lock")
	}
	testcheck.NoError(t, lock.Close())
	second, err := Create(root)
	if err != nil {
		t.Fatalf("writer after release: %v", err)
	}
	_ = second.Close()
}

func TestReadOnlyStoreCanOpenBesideWriter(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	writer := testcheck.Must(Create(root))(t)
	defer writer.Close()
	batch := fixtureBatch(t)
	if _, err := writer.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	reader := testcheck.Must(OpenReadOnly(root))(t)
	defer reader.Close()
	if _, ok, err := reader.Artifact(t.Context(), batch.Artifacts[0].ID); err != nil || !ok {
		t.Fatalf("read beside writer = (%v, %v)", ok, err)
	}
}

func TestTornTailRecovered(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	store := testcheck.Must(Create(root))(t)
	batch := fixtureBatch(t)
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	testcheck.NoError(t, store.Close())
	path := filepath.Join(root, storeFilename)
	before := testcheck.Must(os.Stat(path))(t)
	file := testcheck.Must(os.OpenFile(path, os.O_WRONLY|os.O_APPEND, storeFileMode))(t)
	if _, err := file.Write(make([]byte, tornTailBytes)); err != nil {
		t.Fatal(err)
	}
	_ = file.Close()
	store = testcheck.Must(Create(root))(t)
	testcheck.NoError(t, store.Close())
	after := testcheck.Must(os.Stat(path))(t)
	if after.Size() != before.Size() {
		t.Fatalf("recovered size = %d, want %d", after.Size(), before.Size())
	}
}

func TestCompleteCorruptFrameRejected(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	store := testcheck.Must(Create(root))(t)
	if _, err := store.Commit(t.Context(), fixtureBatch(t)); err != nil {
		t.Fatal(err)
	}
	testcheck.NoError(t, store.Close())
	path := filepath.Join(root, storeFilename)
	file := testcheck.Must(os.OpenFile(path, os.O_RDWR, storeFileMode))(t)
	info := testcheck.Must(file.Stat())(t)
	offset := info.Size() - crc32.Size - 1
	value := []byte{0}
	if _, err := file.ReadAt(value, offset); err != nil {
		t.Fatal(err)
	}
	value[0] ^= 0xff
	if _, err := file.WriteAt(value, offset); err != nil {
		t.Fatal(err)
	}
	_ = file.Close()
	if reopened, err := Create(root); err == nil {
		_ = reopened.Close()
		t.Fatal("corrupt complete frame accepted")
	}
}

func TestConcurrentCommitsSerialize(t *testing.T) {
	t.Parallel()
	store := testcheck.Must(Open(t.TempDir()))(t)
	defer store.Close()
	var wait sync.WaitGroup
	errorsByWorker := make(chan error, fixtureWorkerCount)
	for worker := range fixtureWorkerCount {
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
			_, err = store.Commit(t.Context(), artifact.Batch{
				Key:       fmt.Sprintf("fixture/worker/%d", index),
				Artifacts: []artifact.Descriptor{descriptor},
			})
			errorsByWorker <- err
		}(worker)
	}
	wait.Wait()
	close(errorsByWorker)
	for err := range errorsByWorker {
		testcheck.NoError(t, err)
	}
	_, sequence := store.Head()
	if sequence != fixtureWorkerCount {
		t.Fatalf("sequence = %d, want %d", sequence, fixtureWorkerCount)
	}
}
