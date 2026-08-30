package dataset

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func incrementalSelectionFixture(t *testing.T, names ...string) InteractionSelection {
	t.Helper()
	sources := make([]InteractionSelectionSource, 0, len(names))
	for index, name := range names {
		sources = append(sources, InteractionSelectionSource{
			Source:     testutil.ArtifactID(t, artifact.KindEvidence, "arc-"+name),
			CausalRoot: testutil.ArtifactID(t, artifact.KindEvidence, "root-"+name),
			Tokens:     uint64(10 * (index + 1)), Bytes: uint64(100 * (index + 1)),
			Documents: 1, Depth: uint32(index + 1),
		})
	}
	selection, err := SelectInteractions(InteractionSelectionBounds{
		MaxTokens: 1000, MaxBytes: 10000, MaxDocuments: 10, MaxDepth: 10, MaxResults: 10,
	}, artifact.CommitID{0x11}, sources, nil)
	if err != nil {
		t.Fatal(err)
	}
	return selection
}

// TestIncrementalContextPreservesEvidence pins the selection-layer delta:
// held arcs are referenced by identity without duplicating bytes, fresh arcs
// carry full citations and their exact transfer cost, the decomposition
// covers every admitted source exactly once, and a delta that omitted an
// admitted stimulus can never validate.
func TestIncrementalContextPreservesEvidence(t *testing.T) {
	prior := incrementalSelectionFixture(t, "alpha", "beta")
	current := incrementalSelectionFixture(t, "beta", "gamma")
	delta, err := SelectIncremental(prior, current)
	if err != nil {
		t.Fatal(err)
	}
	if len(delta.Reused) != 1 || len(delta.Fresh) != 1 {
		t.Fatalf("delta decomposition = %+v", delta)
	}
	if delta.Fresh[0].Source != current.Sources[1].Source && delta.Fresh[0].Source != current.Sources[0].Source {
		t.Fatalf("fresh arc lost its citation: %+v", delta.Fresh)
	}
	if delta.TransferTokens != delta.Fresh[0].Tokens || delta.TransferBytes != delta.Fresh[0].Bytes {
		t.Fatalf("transfer accounting = %+v", delta)
	}
	if err := delta.Validate(prior, current); err != nil {
		t.Fatal(err)
	}

	omitted := delta
	omitted.Reused = nil
	if err := omitted.Validate(prior, current); err == nil ||
		!strings.Contains(err.Error(), "differs from its admitted boundaries") {
		t.Fatalf("omitted admitted stimulus validated: %v", err)
	}
	foreign := delta
	foreign.SourceSet = testutil.ArtifactID(t, artifact.KindEvidence, "foreign-set")
	if err := foreign.Validate(prior, current); err == nil {
		t.Fatal("foreign source set validated")
	}

	identical, err := SelectIncremental(prior, prior)
	if err != nil {
		t.Fatal(err)
	}
	if len(identical.Fresh) != 0 || identical.TransferBytes != 0 || len(identical.Reused) != len(prior.Sources) {
		t.Fatalf("unchanged context still transfers bytes: %+v", identical)
	}
}
