package runrecord

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/overgodbtest"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestCausalityProjectionRebuild pins the typed-record boundary: record batch
// construction emits the projection fact from the canonical causal context,
// without turning artifact lineage into operational causality.
func TestCausalityProjectionRebuild(t *testing.T) {
	rootID := testutil.ArtifactID(t, artifact.KindEvidence, "typed-causal-root")
	motivation := testutil.ArtifactID(t, artifact.KindEvidence, "typed-causal-motivation")
	causal := testcheck.Must(NewCausalRoot(TriggerControllerProposal, rootID, motivation))(t)
	value := servingObservationFixture(t)
	value.Causal = &causal
	observation := testcheck.Must(NewServingObservation(value))(t)
	batch, err := observation.Batch("causality/typed-serving")
	if err != nil || len(batch.Causality) != 1 || batch.Causality[0].Execution != observation.ID {
		t.Fatalf("typed causal batch = (%+v, %v)", batch.Causality, err)
	}
	seen := map[artifact.ID]bool{observation.ID: true}
	for _, edge := range batch.Lineage {
		seen[edge.Parent] = true
	}
	seen[rootID], seen[motivation] = true, true
	for id := range seen {
		if id != observation.ID {
			batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: id})
		}
	}
	store := overgodbtest.Open(t, t.TempDir())
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	result, err := store.QueryCausality(t.Context(), overgodb.CausalityQuery{Root: &rootID, MaxResults: 1})
	if err != nil || result.Matched != 1 || result.Links[0].Execution != observation.ID ||
		result.Links[0].Trigger != string(TriggerControllerProposal) {
		t.Fatalf("typed causal projection = (%+v, %v)", result, err)
	}
}
