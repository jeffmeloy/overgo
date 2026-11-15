package plan

import (
	"encoding/json"
	"maps"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// stepFields are what a step of the live plan holds beside its id: its title, its check (verify), the files it may
// write, its order (depends_on), its cause and what it serves (rationale), and whether its work needs the card.
var stepFields = []string{"id", "title", "verify", "files", "depends_on", "rationale", "needs_card"}

// TestAStepHoldsItsSixFields holds every step of the live plan, as the plan reads it, to its id and six fields, and
// every item to no owner and no status (single-source/step-fields): all 76 open steps held status "open" and all 45
// items status "open" and owner "master-lead", the plan's own lane, so neither field told one row from another, while
// dispatch read them on every call.
func TestAStepHoldsItsSixFields(t *testing.T) {
	t.Parallel()
	document := testcheck.Must(Load(filepath.Join(testutil.RepoRoot(t), filepath.FromSlash(Path))))(t)
	for _, item := range document.Items {
		var held map[string]json.RawMessage
		testcheck.NoError(t, json.Unmarshal(testcheck.Must(json.Marshal(item))(t), &held))
		for _, gone := range []string{"owner", "status"} {
			if _, kept := held[gone]; kept {
				t.Errorf("item %s holds a %s", item.ID, gone)
			}
		}
		for _, step := range item.Steps {
			var fields map[string]json.RawMessage
			testcheck.NoError(t, json.Unmarshal(testcheck.Must(json.Marshal(step))(t), &fields))
			for _, name := range slices.Sorted(maps.Keys(fields)) {
				if !slices.Contains(stepFields, name) {
					t.Errorf("step %s/%s holds %s, beside its id and six fields", item.ID, step.ID, name)
				}
			}
		}
	}
}
