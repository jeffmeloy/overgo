package plan

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/testcheck"
)

// TestThePlanHoldsNoAnsweredDecision holds the plan to rows and open decisions: on 2026-11-08 it held 28 decisions, all
// answered, most of its 29 KB, which every reader of decisions passes over. The repository's plan holds no answered
// decision; a plan saved with one holds it only as the record the next landing's message states, and an open one
// stays a decision.
func TestThePlanHoldsNoAnsweredDecision(t *testing.T) {
	t.Parallel()
	held := testcheck.Must(Load(filepath.Join("..", "..", filepath.FromSlash(Path))))(t)
	for _, decision := range held.Decisions {
		if decision.Answer != "" {
			t.Errorf("%s holds the answered decision %s", Path, decision.ID)
		}
	}
	document := Plan{Campaign: "fixture", Decisions: []Decision{
		{ID: "answered", Question: "Keep both?", Options: []string{"yes", "no"}, Recommendation: "yes", Asked: "2026-11-01", Answer: "yes"},
		{ID: "open", Question: "Keep one?", Options: []string{"yes", "no"}, Recommendation: "no", Asked: "2026-11-01"},
	}}
	path := filepath.Join(t.TempDir(), "plan.json")
	testcheck.NoError(t, Save(path, document))
	saved := testcheck.Must(Load(path))(t)
	if len(saved.Decisions) != 1 || saved.Decisions[0].ID != "open" || len(saved.Answered) != 1 || saved.Answered[0].Answer != "yes" {
		t.Fatalf("a plan saved with an answered and an open decision holds decisions %+v and answered %+v; want the open one a decision and the answered one recorded",
			saved.Decisions, saved.Answered)
	}
	_ = os.Remove(path)
}
