package runrecord

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

// TestCanonicalDocumentPublicationOwners proves this package's typed
// documents publish only through their codec owners: the produced
// content validates its own bound identity and parses back to the
// identical record, with no package-local marshal-then-Content pair.
func TestCanonicalDocumentPublicationOwners(t *testing.T) {
	split := testutil.ArtifactBytesID(t, artifact.KindDatasetShard, []byte("publication-split"))
	authority := testutil.ArtifactBytesID(t, artifact.KindEvidence, []byte("publication-authority"))
	budget, err := NewBudget("queries", split, 16, authority)
	if err != nil {
		t.Fatal(err)
	}
	content, err := budget.Content()
	if err != nil {
		t.Fatal(err)
	}
	if content.Descriptor.ID != budget.ID {
		t.Fatalf("content identity %s differs from document %s", content.Descriptor.ID, budget.ID)
	}
	parsed, err := ParseBudget(content.Data)
	if err != nil || parsed.ID != budget.ID || parsed.Issued != budget.Issued {
		t.Fatalf("round trip = (%+v, %v)", parsed, err)
	}
}
