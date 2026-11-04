package landing

import (
	"strings"
	"testing"

	"overgo/internal/plan"
)

// TestAStepIsLandedOnlyByItsOwnWholeTrailer files row/do as a step the trunk's plan does not hold, over a history whose
// messages hold its trailer's text twice without landing it: a landed step whose name begins with it, and prose quoting
// it. Neither is its landing, so the plan files it; once a commit's trailer names row/do whole, the plan that files it
// again is refused, naming that commit.
func TestAStepIsLandedOnlyByItsOwnWholeTrailer(t *testing.T) {
	t.Parallel()
	fixture := newRowFixture(t, map[string]string{"a.txt": "base\n"})
	commit := func(message string) string {
		t.Helper()
		fixture.git("commit", "-q", "--allow-empty", "-m", message)
		return fixture.git("rev-parse", "HEAD")
	}
	commit("row/do-more: a longer step\n\nCause: a fixture.\n\n" + plan.LandedRowTrailer + ": row/do-more\n")
	commit("prose: a quoted trailer\n\nThe message before this one wrote " + plan.LandedRowTrailer + ": row/do in a sentence.\n")
	before := plan.Plan{Campaign: "fixture", Items: []plan.Item{{ID: "row", Title: "row", Status: plan.StatusOpen}}}
	filed := plan.Plan{Campaign: "fixture", Items: []plan.Item{{ID: "row", Title: "row", Status: plan.StatusOpen,
		Steps: []plan.Step{{ID: "do", Title: "row", Status: plan.StatusOpen, Verify: "true"}}}}}
	l := &landing{Options: Options{Repo: fixture.repo}}
	if err := l.landedAgain(t.Context(), before, filed); err != nil {
		t.Fatalf("row/do, landed by no trailer of its own, was refused: %v", err)
	}
	landed := commit("row/do: the step\n\nCause: a fixture.\n\n" + plan.LandedRowTrailer + ": row/do\n")
	if err := l.landedAgain(t.Context(), before, filed); err == nil || !strings.Contains(err.Error(), landed[:12]) {
		t.Fatalf("row/do, landed as %.12s, filed again ended as %v, want it refused naming that landing", landed, err)
	}
}
