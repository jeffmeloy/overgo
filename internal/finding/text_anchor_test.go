package finding

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
)

// TestTextBatchBindsArtifactIdentitiesAsThemselves holds an evidence value
// that is already an artifact identity to binding that artifact, not the text
// of its name: a census disposition must name the census records it joins.
func TestTextBatchBindsArtifactIdentitiesAsThemselves(t *testing.T) {
	t.Parallel()
	census, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte("a census"))
	if err != nil {
		t.Fatal(err)
	}
	document, batch, err := NewTextBatch("census moved", SeverityLow,
		[]string{"cmd/compatibility"}, []string{census.String(), "the live denominator grew"}, "rebind", "go test ./cmd/compatibility")
	if err != nil {
		t.Fatal(err)
	}
	if len(document.Evidence) != 2 || !slices.Contains(document.Evidence, census) {
		t.Fatalf("evidence = %v, want the census identity among them", document.Evidence)
	}
	for _, content := range batch.Contents {
		if content.Descriptor.ID == census {
			t.Fatal("the census identity was written as a text anchor")
		}
	}
}
