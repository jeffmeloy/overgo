//go:build modeltest

package main

import (
	"maps"
	"slices"
	"testing"

	"overgo/internal/testutil"
)

// TestReviewedDeltaRegistryEvidence re-reads every receipt and holds each kind
// to refusing a delta its evidence does not answer for.
func TestReviewedDeltaRegistryEvidence(t *testing.T) {
	t.Parallel()
	root := testutil.RepoRoot(t)
	registry, err := loadMediaDeltaRegistry(root)
	if err != nil {
		t.Fatal(err)
	}
	receipts := &mediaReceiptStore{root: root}
	t.Cleanup(func() {
		if err := receipts.close(); err != nil {
			t.Error(err)
		}
	})
	kinds := map[string]bool{}
	for _, delta := range registry.Deltas {
		kinds[delta.Receipt] = true
		broken := delta
		broken.Changes = maps.Clone(delta.Changes)
		switch delta.Receipt {
		case mediaReceiptGoTest, mediaReceiptSourceBound:
			broken.Required = map[string][]string{"overgo/cmd/validate": {"TestAbsentFromReceipt"}}
		case mediaReceiptRotary:
			path := slices.Sorted(maps.Keys(delta.Changes))[0]
			change := broken.Changes[path]
			change.Before = change.After
			broken.Changes[path] = change
		case mediaReceiptDocument:
			continue // bound by its document's identity when read
		}
		if checkMediaDeltaReceipt(receipts, registry.Paths, broken) == nil {
			t.Fatalf("%s: evidence accepted for a delta it does not answer for", delta.ID)
		}
		if delta.Source != "" {
			moved := delta
			moved.Source = registry.Base
			if checkMediaDeltaReceipt(receipts, registry.Paths, moved) == nil {
				t.Fatalf("%s: a source commit without the after states accepted", delta.ID)
			}
		}
	}
	for _, kind := range []string{mediaReceiptGoTest, mediaReceiptSourceBound, mediaReceiptRotary, mediaReceiptDocument} {
		if !kinds[kind] {
			t.Fatalf("receipt kind %s is unused", kind)
		}
	}
	if _, err := mediaDocumentChanges([]byte(`{"scope":"no changes"}`)); err == nil {
		t.Fatal("an evidence document without changes accepted")
	}
}
