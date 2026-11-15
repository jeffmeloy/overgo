package plan

import (
	"path/filepath"
	"testing"

	"overgo/internal/testcheck"
)

// TestOnlyAConsolidationCarriesACount holds a saved plan to counts on consolidations alone (consolidation-counts/do): a
// feature row's count asks nothing a landing holds it to, since only a consolidation is held to its count, so Save
// drops it and keeps the consolidation's.
func TestOnlyAConsolidationCarriesACount(t *testing.T) {
	t.Parallel()
	path := filepath.Join(t.TempDir(), "plan.json")
	step := []Step{{ID: "do", Title: "do", Verify: "true"}}
	testcheck.NoError(t, Save(path, Plan{Campaign: "fixture", Items: []Item{
		{ID: "feature", Title: "feature", Budget: Budget{Nodes: 300, Reason: "an estimate"}, Steps: step},
		{ID: "shrink", Title: "shrink", Class: ClassConsolidation, Budget: Budget{Nodes: -40, Reason: "it must shrink"}, Steps: step},
	}}))
	saved := testcheck.Must(Load(path))(t)
	if feature, shrink := saved.Items[0].Budget, saved.Items[1].Budget; feature != (Budget{}) || shrink != (Budget{Nodes: -40, Reason: "it must shrink"}) {
		t.Fatalf("a saved plan's counts are %+v on the feature and %+v on the consolidation; want none and the consolidation's", feature, shrink)
	}
}
