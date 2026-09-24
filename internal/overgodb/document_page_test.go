package overgodb

import (
	"fmt"
	"slices"
	"testing"

	"overgo/internal/artifact"
)

// TestQueryScansProportionalToItsPage holds a document read to its own
// schema and page. One schema's documents are interleaved, commit by commit,
// with many more of another schema. The read's candidates are exactly the
// queried schema's records, however much else the store holds. Paging by
// cursor, oldest-first and newest-first, returns every document exactly
// once in commit order, truncating only while more remain. The explicit
// census counts them all.
func TestQueryScansProportionalToItsPage(t *testing.T) {
	t.Parallel()
	store, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	contract := func(schema string) artifact.DocumentContract {
		return artifact.DocumentContract{Kind: artifact.KindEvidence, MediaType: "application/json", Schema: schema}
	}
	queried, other := contract("test/paged/v1"), contract("test/crowd/v1")
	document := func(c artifact.DocumentContract, body string) artifact.Content {
		data := []byte(fmt.Sprintf("{%q:%q}", c.Schema, body))
		id, err := artifact.IdentifyBytes(c.Kind, data)
		if err != nil {
			t.Fatal(err)
		}
		return artifact.Content{Descriptor: artifact.Descriptor{ID: id, Size: uint64(len(data)), MediaType: c.MediaType, Schema: c.Schema}, Data: data}
	}
	const pagedCount, crowdPerCommit = 7, 20
	var want []artifact.ID
	for commit := range pagedCount {
		batch := artifact.Batch{Key: fmt.Sprintf("paged/%d", commit)}
		for index := range crowdPerCommit {
			batch.Contents = append(batch.Contents, document(other, fmt.Sprintf("%d-%d", commit, index)))
		}
		paged := document(queried, fmt.Sprint(commit))
		batch.Contents = append(batch.Contents, paged)
		want = append(want, paged.Descriptor.ID)
		if _, err := store.Commit(t.Context(), batch); err != nil {
			t.Fatal(err)
		}
	}
	store.mu.RLock()
	candidates := len(store.state.artifacts.documentPositions([]artifact.DocumentContract{queried}))
	store.mu.RUnlock()
	if candidates != pagedCount {
		t.Fatalf("the read's candidates are %d records, want the queried schema's %d of %d", candidates, pagedCount, pagedCount*(crowdPerCommit+1))
	}
	for _, order := range []DocumentOrder{DocumentOldestFirst, DocumentNewestFirst} {
		query := DocumentQuery{Contracts: []artifact.DocumentContract{queried}, Order: order, MaxResults: 3}
		var got []artifact.ID
		for pages := 0; ; pages++ {
			var page []artifact.ID
			result, err := store.VisitDocuments(t.Context(), query, func(view DocumentView) error {
				page = append(page, view.Content.Descriptor.ID)
				return nil
			})
			if err != nil || len(page) > query.MaxResults || pages > pagedCount {
				t.Fatalf("order %v page %d = %v, %v", order, pages, page, err)
			}
			got = append(got, page...)
			if !result.Truncated {
				break
			}
			query.Cursor = result.Next
		}
		expected := slices.Clone(want)
		if order == DocumentNewestFirst {
			slices.Reverse(expected)
		}
		if !slices.Equal(got, expected) {
			t.Fatalf("order %v paged %v, want %v", order, got, expected)
		}
	}
	count, err := store.CountDocuments(t.Context(), DocumentQuery{Contracts: []artifact.DocumentContract{queried}, Order: DocumentOldestFirst})
	if err != nil || count != pagedCount {
		t.Fatalf("census = %d, %v; want %d", count, err, pagedCount)
	}
}
