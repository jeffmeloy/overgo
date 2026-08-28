package overgodb

import (
	"context"
	"fmt"
	"io"
	"sort"
	"testing"

	"overgo/internal/artifact"
)

// repositoryView reduces one Repository to a deterministic listing
// through the public boundary alone: every corpus artifact, its
// content bytes, aliases, and lineage in both directions.
func repositoryView(t *testing.T, repository artifact.Repository) string {
	t.Helper()
	ctx := context.Background()
	var view []string
	for ordinal := range scaleCorpusCommits {
		content := scaleContent(ordinal)
		id, err := artifact.IdentifyBytes(artifact.KindRun, content)
		if err != nil {
			t.Fatal(err)
		}
		descriptor, found, err := repository.Artifact(ctx, id)
		if err != nil || !found {
			t.Fatalf("artifact %d: found=%v err=%v", ordinal, found, err)
		}
		view = append(view, fmt.Sprintf("artifact/%s/%d/%s", id, descriptor.Size, descriptor.MediaType))
		parents, err := repository.Parents(ctx, id)
		if err != nil {
			t.Fatal(err)
		}
		for _, edge := range parents {
			view = append(view, fmt.Sprintf("parent/%s/%s/%s", edge.Child, edge.Parent, edge.Relation))
		}
		children, err := repository.Children(ctx, id)
		if err != nil {
			t.Fatal(err)
		}
		for _, edge := range children {
			view = append(view, fmt.Sprintf("child/%s/%s/%s", edge.Child, edge.Parent, edge.Relation))
		}
		if ordinal%scaleAliasStride == 0 {
			target, found, err := repository.ResolveAlias(ctx, aliasName(ordinal))
			if err != nil || !found {
				t.Fatalf("alias %d: found=%v err=%v", ordinal, found, err)
			}
			view = append(view, fmt.Sprintf("alias/%s/%s", aliasName(ordinal), target))
			_, reader, found, err := repository.OpenContent(ctx, id)
			if err != nil || !found {
				t.Fatalf("content %d: found=%v err=%v", ordinal, found, err)
			}
			data, err := io.ReadAll(reader)
			if err != nil || len(data) != scaleContentBytes {
				t.Fatalf("content %d bytes=%d err=%v", ordinal, len(data), err)
			}
		}
	}
	head, sequence := repository.Head()
	view = append(view, fmt.Sprintf("head/%s/%d", head, sequence))
	sort.Strings(view)
	var joined string
	for _, line := range view {
		joined += line + "\n"
	}
	return joined
}

// TestModularStoreRepositoryParity proves the modular core presents
// identical Repository behavior on every open path: the snapshot-
// anchored open and the journal-only open answer the complete public
// read surface identically on the deterministic corpus.
func TestModularStoreRepositoryParity(t *testing.T) {
	root := t.TempDir()
	if _, _, _, contentBytes := buildScaleCorpus(t, root); contentBytes == 0 {
		t.Fatal("corpus carried no content")
	}
	snapshotOpen, err := OpenReadOnly(root)
	if err != nil {
		t.Fatal(err)
	}
	defer snapshotOpen.Close()
	if !snapshotOpen.SnapshotReplay().Loaded {
		t.Fatal("snapshot open path unavailable")
	}
	journalRoot := t.TempDir()
	copyCorpusFile(t, snapshotOpen.log.file.Name(), journalRoot+"/"+storeFilename)
	journalOpen, err := OpenReadOnly(journalRoot)
	if err != nil {
		t.Fatal(err)
	}
	defer journalOpen.Close()
	if journalOpen.SnapshotReplay().Loaded {
		t.Fatal("journal open path unexpectedly used a snapshot")
	}
	if snapshotView, journalView := repositoryView(t, snapshotOpen), repositoryView(t, journalOpen); snapshotView != journalView {
		t.Fatal("repository behavior differs between snapshot and journal open paths")
	}
}
