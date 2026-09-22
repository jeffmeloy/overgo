package gate

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestPackageObligationsAreOneDocumentPerLanding holds a preparation to
// publishing one obligations document for all its packages and no obligation
// record, and holds reuse to still hitting across landings on that document:
// a later preparation of the same packages at the same inputs finds the
// earlier receipt by the obligation it computes, and changed inputs publish a
// second document rather than rewriting the first.
func TestPackageObligationsAreOneDocumentPerLanding(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	g, inputs := terminalEvidenceFixture(t, root)
	packages := []string{"fixture/good", "fixture/pending"}
	ledger, err := g.openPackageEvidence()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ledger.store.Close() })
	if err := ledger.prepare(t.Context(), packages, "complete", inputs, g.retryCache); err != nil {
		t.Fatal(err)
	}
	if listings, records := storedPackageObligationCounts(t, ledger.store); listings != 1 || records != 0 {
		t.Fatalf("after one preparation: listings=%d obligation records=%d, want 1 and 0", listings, records)
	}
	for _, pkg := range packages {
		if ledger.listings[pkg] != ledger.listings[packages[0]] || !ledger.listings[pkg].Valid() {
			t.Fatalf("%s is listed in %s, not the preparation's one document", pkg, ledger.listings[pkg])
		}
	}
	if err := ledger.record(t.Context(), "fixture/good", true, map[string]string{"TestGood": "pass"}); err != nil {
		t.Fatal(err)
	}

	// A later landing: the same packages meet the same inputs and find the
	// receipt through the obligation they compute; the listing's identity is
	// the same, so the store holds one document still.
	later, _ := terminalEvidenceFixture(t, root)
	next, err := later.openPackageEvidence()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = next.store.Close() })
	if err := next.prepare(t.Context(), packages, "complete", inputs, later.retryCache); err != nil {
		t.Fatal(err)
	}
	witness := next.prepared["fixture/good"]
	if !witness.Receipt.Valid() || !witness.Passed || witness.Obligation != ledger.obligations["fixture/good"].ID {
		t.Fatalf("reuse across landings did not hit: %+v", witness)
	}
	if pending, reused, err := later.packageCachePartition(packages, "complete", inputs); err != nil || reused != 1 || len(pending) != 1 || pending[0] != "fixture/pending" {
		t.Fatalf("partition after reuse: pending=%v reused=%d error=%v", pending, reused, err)
	}
	if listings, records := storedPackageObligationCounts(t, next.store); listings != 1 || records != 0 {
		t.Fatalf("after the same preparation again: listings=%d obligation records=%d, want 1 and 0", listings, records)
	}

	// Changed inputs are a new obligation in a new document; the old listing
	// is retained and credits nothing.
	inputs["fixture/good"] = testutil.ArtifactID(t, artifact.KindEvidence, "changed source")
	if err := next.prepare(t.Context(), packages, "complete", inputs, later.retryCache); err != nil {
		t.Fatal(err)
	}
	if next.prepared["fixture/good"].Receipt.Valid() {
		t.Fatal("changed inputs were credited with the earlier receipt")
	}
	if listings, records := storedPackageObligationCounts(t, next.store); listings != 2 || records != 0 {
		t.Fatalf("after changed inputs: listings=%d obligation records=%d, want 2 and 0", listings, records)
	}
}

// storedPackageObligationCounts counts the obligations documents and the
// per-package obligation records the store holds.
func storedPackageObligationCounts(t *testing.T, store *overgodb.Store) (listings, records int) {
	t.Helper()
	if err := store.Refresh(t.Context()); err != nil {
		t.Fatal(err)
	}
	raw := func(data []byte) ([]byte, error) { return data, nil }
	count := func(contract artifact.DocumentContract) int {
		total := 0
		_, err := overgodb.VisitDecodedDocuments(t.Context(), store, overgodb.DocumentQuery{Contracts: []artifact.DocumentContract{contract}, Order: overgodb.DocumentOldestFirst},
			raw, func(overgodb.DocumentView, []byte) error { total++; return nil })
		if err != nil {
			t.Fatal(err)
		}
		return total
	}
	listings = count(artifact.DocumentContract{Kind: artifact.KindEvidence, MediaType: packageObligationsMediaType, Schema: packageObligationsSchema})
	records = count(artifact.DocumentContract{Kind: artifact.KindEvidence, MediaType: runrecord.AgentObligationMediaType, Schema: runrecord.AgentObligationSchema})
	return listings, records
}

// requireStoredBatchObligation holds a verification checkpoint's obligation to
// being a record of its own: the gate reads it back when it re-plans a batch,
// unlike a package obligation, which only its listing retains.
func requireStoredBatchObligation(t *testing.T, root string, id artifact.ID) {
	t.Helper()
	store, err := overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := runrecord.RequireAgentObligation(t.Context(), store, id); err != nil {
		t.Fatal(err)
	}
}
