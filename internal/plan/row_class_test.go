package plan

import "testing"

// TestRowsDispatchByDeclaredClass holds a row's kind to its declared class,
// never its id: a declared merge row lands the plan-only lane commit whatever
// it is named, a merge-named row that declares no class does not, and the
// plan refuses a class it does not know.
func TestRowsDispatchByDeclaredClass(t *testing.T) {
	t.Parallel()
	lane := Plan{Lane: "hatchet", Items: []Item{
		{ID: "sync-topic", Class: ClassMerge},
		{ID: "merge-0123456789ab"},
	}}
	if err := PlanOnlyCommitRefusal(lane, "sync-topic/do", []string{Path}, false); err != nil {
		t.Fatalf("a declared merge row was refused its plan-only commit: %v", err)
	}
	if err := PlanOnlyCommitRefusal(lane, "merge-0123456789ab/do", []string{Path}, false); err == nil {
		t.Fatal("a merge-named row without the merge class landed a plan-only commit")
	}
	for class, valid := range map[string]bool{"": true, ClassConsolidation: true, ClassMerge: true, "automation": false} {
		document := Plan{Campaign: "c", Items: []Item{{ID: "row", Title: "t", Class: class, Status: StatusOpen,
			Steps: []Step{{ID: "do", Title: "t", Status: StatusOpen, Verify: "go test ./x -run '^TestX$'"}}}}}
		if err := Validate(document); (err == nil) != valid {
			t.Fatalf("class %q validated %v, want valid=%t", class, err, valid)
		}
	}
}
