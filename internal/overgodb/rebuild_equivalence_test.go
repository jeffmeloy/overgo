package overgodb

import (
	"fmt"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
)

// querySurfaceDigest reduces the complete public query surface to one
// string: bounded queries with cursor pagination, aliases, lineage
// traversal in both directions, document listing, locations, and the
// commit idempotency record.
func querySurfaceDigest(t *testing.T, store *Store) string {
	t.Helper()
	ctx := t.Context()
	var out string
	query := Query{
		Kind: artifact.KindRun, MaxResults: scaleAliasStride,
		Projection:   ProjectArtifacts | ProjectContentPresence | ProjectAliases | ProjectCommits,
		FromSequence: 1, ToSequence: scaleCorpusCommits,
	}
	for page := 0; ; page++ {
		result, err := store.Query(ctx, query)
		if err != nil {
			t.Fatal(err)
		}
		out += fmt.Sprintf("page/%d/matched=%d/truncated=%v\n", page, result.Matched, result.Truncated)
		for _, descriptor := range result.Artifacts {
			out += fmt.Sprintf("artifact/%s/%d/%s\n", descriptor.ID, descriptor.Size, descriptor.MediaType)
		}
		for _, content := range result.Contents {
			out += fmt.Sprintf("content/%s\n", content.Artifact)
		}
		for _, alias := range result.Aliases {
			out += fmt.Sprintf("alias/%s/%s\n", alias.Name, alias.Target)
		}
		for _, commit := range result.Commits {
			out += fmt.Sprintf("commit/%s/%s/%d\n", commit.Key, commit.ID, commit.Sequence)
		}
		if result.Next == nil {
			break
		}
		query.Cursor = result.Next
	}
	for ordinal := 0; ordinal < scaleCorpusCommits; ordinal += scaleAliasStride {
		id, err := artifact.IdentifyBytes(artifact.KindRun, scaleContent(ordinal))
		if err != nil {
			t.Fatal(err)
		}
		target, found, err := store.ResolveAlias(ctx, aliasName(ordinal))
		if err != nil || !found {
			t.Fatalf("alias %d: %v", ordinal, err)
		}
		out += fmt.Sprintf("resolve/%s/%s\n", aliasName(ordinal), target)
		parents, err := store.Parents(ctx, id)
		if err != nil {
			t.Fatal(err)
		}
		for _, edge := range parents {
			out += fmt.Sprintf("parent/%s/%s/%s\n", edge.Child, edge.Parent, edge.Relation)
		}
		children, err := store.Children(ctx, id)
		if err != nil {
			t.Fatal(err)
		}
		for _, edge := range children {
			out += fmt.Sprintf("child/%s/%s/%s\n", edge.Child, edge.Parent, edge.Relation)
		}
		locations, err := store.Locations(ctx, id)
		if err != nil {
			t.Fatal(err)
		}
		for _, location := range locations {
			out += fmt.Sprintf("location/%s/%s\n", location.Kind, location.Value)
		}
	}
	if err := store.VisitAliases(ctx, "scale/", func(view AliasView) error {
		out += fmt.Sprintf("visit-alias/%s/%s\n", view.Name, view.Target)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	head, sequence := store.Head()
	out += fmt.Sprintf("head/%s/%d\n", head, sequence)
	return out
}

// TestProjectionRebuildEquivalence proves every derived view is
// reproducible from canonical commits alone: the checkpoint-anchored
// open, the cold journal replay, and an independent single-projection
// rebuild each answer the full query surface identically at the same
// head.
func TestProjectionRebuildEquivalence(t *testing.T) {
	root := t.TempDir()
	copyScaleCorpus(t, root)

	checkpointed, err := OpenReadOnly(root)
	if err != nil {
		t.Fatal(err)
	}
	defer checkpointed.Close()
	if !checkpointed.SnapshotReplay().Loaded {
		t.Fatal("checkpoint anchor unavailable")
	}
	checkpointDigest := querySurfaceDigest(t, checkpointed)

	coldRoot := t.TempDir()
	copyCorpusFile(t, checkpointed.log.file.Name(), filepath.Join(coldRoot, storeFilename))
	copyCorpusTree(t, filepath.Join(root, blobDirectory), filepath.Join(coldRoot, blobDirectory))
	copyCorpusTree(t, filepath.Join(root, segmentDirectory), filepath.Join(coldRoot, segmentDirectory))
	cold, err := OpenReadOnly(coldRoot)
	if err != nil {
		t.Fatal(err)
	}
	defer cold.Close()
	if cold.SnapshotReplay().Loaded {
		t.Fatal("cold path unexpectedly used an anchor")
	}
	if coldDigest := querySurfaceDigest(t, cold); coldDigest != checkpointDigest {
		t.Fatal("cold replay answers differently from the checkpoint anchor")
	}

	// Independent rebuild: each projection re-derives its state from
	// decoded journal records alone, with no other facet loaded, and
	// must serialize to exactly the checkpoint the anchored open holds.
	for _, registered := range projections(&checkpointed.state) {
		expected, err := registered.view.checkpoint()
		if err != nil {
			t.Fatal(err)
		}
		independent := newCatalogState()
		target := projections(&independent)[indexOfProjection(t, registered.name)]
		if target.name != registered.name {
			t.Fatalf("projection registry order drifted: %s vs %s", target.name, registered.name)
		}
		log, _, err := openRecordLog(coldRoot, true, replayAnchor{}, func(record logRecord) error {
			batch, payloadHash, locators, err := decodeRecord(record)
			if err != nil {
				return err
			}
			target.view.applyCommit(batch, locators, record.sequence)
			independent.commits.add(committedBatch{
				key: batch.Key, id: record.id, payload: payloadHash, sequence: record.sequence,
				coordinate: commitCoordinate{
					segment: record.segment, offset: record.offset - frameHeaderBytes, size: uint32(len(record.payload)),
				},
			})
			return nil
		})
		if err != nil {
			t.Fatalf("independent rebuild of %s: %v", registered.name, err)
		}
		_ = log.Close()
		rebuilt, err := target.view.checkpoint()
		if err != nil {
			t.Fatal(err)
		}
		if string(rebuilt) != string(expected) {
			t.Fatalf("projection %s rebuilt independently differs from the anchored state", registered.name)
		}
	}
}

func indexOfProjection(t *testing.T, name string) int {
	t.Helper()
	state := newCatalogState()
	index := slices.IndexFunc(projections(&state), func(registered registeredProjection) bool {
		return registered.name == name
	})
	if index >= 0 {
		return index
	}
	t.Fatalf("projection %q is not registered", name)
	return -1
}
