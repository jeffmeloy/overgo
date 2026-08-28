package recipe

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

// TestCanonicalDocumentPublicationOwners proves this package's typed
// documents publish only through their codec owners: produced content
// carries the document's own identity, parses back identically, and
// batches with the owner-derived lineage closure.
func TestCanonicalDocumentPublicationOwners(t *testing.T) {
	subject := testutil.ArtifactID(t, artifact.KindRecipe, "publication-subject")
	derivation := testutil.ArtifactID(t, artifact.KindEvidence, "publication-derivation")
	decision, err := NewDecision(
		subject, DecisionRefused, EvidenceParity, "publication owner check",
		Decider{CodeCommit: decisionTestCommit, Derivation: derivation},
		[]artifact.ID{derivation},
	)
	if err != nil {
		t.Fatal(err)
	}
	content, err := decision.Content()
	if err != nil {
		t.Fatal(err)
	}
	if content.Descriptor.ID != decision.ID {
		t.Fatalf("content identity %s differs from document %s", content.Descriptor.ID, decision.ID)
	}
	parsed, err := ParseDecision(content.Data)
	if err != nil || parsed.ID != decision.ID {
		t.Fatalf("round trip = (%+v, %v)", parsed, err)
	}
	batch, err := decision.Batch("publication/decision")
	if err != nil || len(batch.Contents) != 1 || len(batch.Lineage) == 0 {
		t.Fatalf("batch = (%+v, %v)", batch, err)
	}
}
