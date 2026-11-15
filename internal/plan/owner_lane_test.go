package plan

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestARowOfThePlansLaneIsSavedWithNoOwner holds the plan to saving no owner for a row its own lane owns, which the
// plan's lane says already, and to reading such a row as the lane's again (single-source/owner-lane): all 45 rows of
// the live plan named the plan's lane as their owner, and the step that removes the field lands under a gate that
// must read the plan it writes. A row another lane owns keeps its owner.
func TestARowOfThePlansLaneIsSavedWithNoOwner(t *testing.T) {
	t.Parallel()
	document := Plan{Campaign: "x", Lane: "lane", Items: []Item{
		{ID: "ours", Title: "Ours", Owner: "lane", Status: StatusOpen, Steps: []Step{{ID: "do", Title: "Do", Status: StatusOpen, Verify: "go test ./..."}}},
		{ID: "theirs", Title: "Theirs", Owner: "other", Status: StatusOpen, Steps: []Step{{ID: "do", Title: "Do", Status: StatusOpen, Verify: "go test ./..."}}},
	}}
	saved := filepath.Join(t.TempDir(), "plan.json")
	testcheck.NoError(t, Save(saved, document))
	if text := string(testcheck.Must(os.ReadFile(saved))(t)); strings.Count(text, `"owner"`) != 1 || !strings.Contains(text, `"owner": "other"`) {
		t.Fatalf("the plan is saved with owners %s; want only the other lane's", text)
	}
	read := testcheck.Must(Load(saved))(t)
	if read.Items[0].Owner != "lane" || read.Items[1].Owner != "other" {
		t.Fatalf("the saved plan reads its rows as owned by %q and %q; want lane and other", read.Items[0].Owner, read.Items[1].Owner)
	}
}
