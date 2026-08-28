package dataset

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestBoundedInteractionSelection(t *testing.T) {
	id := func(name string) artifact.ID { return testutil.ArtifactID(t, artifact.KindEvidence, name) }
	head := artifact.CommitID{1}
	root := id("root")
	sources := []InteractionSelectionSource{
		{Source: id("recent"), CausalRoot: root, Tokens: 3, Bytes: 12, Documents: 2, Depth: 1},
		{Source: id("middle"), CausalRoot: root, Tokens: 4, Bytes: 14, Documents: 2, Depth: 2},
		{Source: id("old"), CausalRoot: root, Tokens: 5, Bytes: 16, Documents: 2, Depth: 3},
	}
	bounds := InteractionSelectionBounds{MaxTokens: 7, MaxBytes: 30, MaxDocuments: 4, MaxDepth: 3, MaxResults: 2}
	first, err := SelectInteractions(bounds, head, sources, nil)
	if err != nil || len(first.Sources) != 2 || !first.Truncated || first.Cursor == nil || first.Usage.Tokens != 7 {
		t.Fatalf("first selection = (%+v, %v)", first, err)
	}
	secondBounds := bounds
	secondBounds.MaxTokens, secondBounds.MaxDocuments, secondBounds.MaxResults = 8, 3, 1
	if _, err := SelectInteractions(secondBounds, head, sources, first.Cursor); err == nil {
		t.Fatal("cursor crossed selection contracts")
	}
	second, err := SelectInteractions(bounds, head, sources, first.Cursor)
	if err != nil || len(second.Sources) != 1 || second.Sources[0].Source != sources[2].Source {
		t.Fatalf("continued selection = (%+v, %v)", second, err)
	}
	otherHead := head
	otherHead[1] = 1
	if _, err := SelectInteractions(bounds, otherHead, sources, first.Cursor); err == nil {
		t.Fatal("cursor crossed journal heads")
	}
	changed := slices.Clone(sources)
	changed[2].Source = id("replacement")
	if _, err := SelectInteractions(bounds, head, changed, first.Cursor); err == nil {
		t.Fatal("cursor crossed cited source sets")
	}
	changed = slices.Clone(sources)
	changed[2].Tokens++
	if _, err := SelectInteractions(bounds, head, changed, first.Cursor); err == nil {
		t.Fatal("cursor crossed changed source accounting")
	}
	tampered := *first.Cursor
	tampered.AfterDepth++
	if _, err := SelectInteractions(bounds, head, sources, &tampered); err == nil {
		t.Fatal("cursor identity accepted mutated continuation state")
	}
}
