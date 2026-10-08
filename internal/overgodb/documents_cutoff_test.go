package overgodb

import (
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
)

func TestDocumentHistoricalContentCutoff(t *testing.T) {
	t.Parallel()
	store, contract, contents := documentQueryFixture(t)
	defer store.Close()
	delayed, err := contract.ContentBytes([]byte(`{"value":"delayed evidence"}`))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), artifact.Batch{Key: "cutoff/descriptor", Artifacts: []artifact.Descriptor{delayed.Descriptor}}); err != nil {
		t.Fatal(err)
	}
	_, cutoff := store.Head()
	query := DocumentQuery{Contracts: []artifact.DocumentContract{contract}, Order: DocumentOldestFirst, MaxResults: len(contents), ThroughSequence: cutoff}
	read := func() []artifact.ID {
		t.Helper()
		var ids []artifact.ID
		page, err := store.VisitDocuments(t.Context(), query, func(view DocumentView) error {
			ids = append(ids, view.Content.Descriptor.ID)
			return nil
		})
		if err != nil || page.Truncated || len(ids) != len(contents) {
			t.Fatalf("historical page=%+v err=%v", page, err)
		}
		return ids
	}
	want := []artifact.ID{contents[0].Descriptor.ID, contents[1].Descriptor.ID}
	if got := read(); !slices.Equal(got, want) {
		t.Fatalf("before later content=%v", got)
	}
	if _, err := store.Commit(t.Context(), artifact.Batch{Key: "cutoff/content", Contents: []artifact.Content{delayed}}); err != nil {
		t.Fatal(err)
	}
	if got := read(); !slices.Equal(got, want) {
		t.Fatalf("later content leaked through earlier descriptor=%v", got)
	}
	query.MaxResults = 1
	page, err := store.VisitDocuments(t.Context(), query, func(DocumentView) error { return nil })
	if err != nil || !page.Truncated || page.Next == nil {
		t.Fatalf("bounded page=%+v err=%v", page, err)
	}
	query.Cursor = page.Next
	query.ThroughSequence--
	if _, err := store.VisitDocuments(t.Context(), query, func(DocumentView) error { return nil }); err == nil {
		t.Fatal("cursor crossed historical cutoffs")
	}
	query.Cursor = nil
	_, current := store.Head()
	query.ThroughSequence = current + 1
	if _, err := store.VisitDocuments(t.Context(), query, func(DocumentView) error { return nil }); err == nil {
		t.Fatal("accepted a future cutoff")
	}
	query.ThroughSequence = cutoff
	query.AliasPrefixes = []string{"fixture/active/"}
	if _, err := store.VisitDocuments(t.Context(), query, func(DocumentView) error { return nil }); err == nil {
		t.Fatal("current aliases contaminated a historical selection")
	}
	query.AliasPrefixes = nil
	query.MaxResults = len(contents)
	root := store.root
	if _, err := store.Snapshot(t.Context()); err != nil {
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
	if got := read(); !slices.Equal(got, want) {
		t.Fatalf("snapshot replay changed historical population=%v", got)
	}
}

func TestDocumentContentByteBudget(t *testing.T) {
	t.Parallel()
	store, contract, contents := documentQueryFixture(t)
	defer store.Close()
	var total uint64
	for _, content := range contents {
		total += content.Descriptor.Size
	}
	query := DocumentQuery{Contracts: []artifact.DocumentContract{contract}, Order: DocumentOldestFirst, MaxContentBytes: total - 1}
	visited := 0
	visit := func(DocumentView) error { visited++; return nil }
	if _, err := store.VisitDocuments(t.Context(), query, visit); err == nil || !strings.Contains(err.Error(), "byte budget") || visited != 0 {
		t.Fatalf("oversized page was not refused before consumption: visited=%d err=%v", visited, err)
	}
	query.MaxContentBytes = total
	page, err := store.VisitDocuments(t.Context(), query, visit)
	if err != nil || page.ContentBytes != total || page.Truncated || visited != len(contents) {
		t.Fatalf("exact budget page=%+v visited=%d err=%v", page, visited, err)
	}
	query.MaxResults = 1
	query.MaxContentBytes = contents[0].Descriptor.Size
	page, err = store.VisitDocuments(t.Context(), query, func(DocumentView) error { return nil })
	if err != nil || page.ContentBytes != contents[0].Descriptor.Size || page.Next == nil || !page.Truncated {
		t.Fatalf("record limit did not bound materialized bytes: page=%+v err=%v", page, err)
	}
	query.Cursor = page.Next
	query.MaxContentBytes = contents[1].Descriptor.Size
	page, err = store.VisitDocuments(t.Context(), query, func(DocumentView) error { return nil })
	if err != nil || page.ContentBytes != contents[1].Descriptor.Size || page.Truncated {
		t.Fatalf("next page byte accounting=%+v err=%v", page, err)
	}
}
