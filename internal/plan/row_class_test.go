package plan

import "testing"

// TestRowsDispatchByDeclaredClass holds a row's kind to its declared class:
// the plan refuses a class it does not know.
func TestRowsDispatchByDeclaredClass(t *testing.T) {
	t.Parallel()
	for class, valid := range map[string]bool{"": true, ClassConsolidation: true, ClassMerge: true, "automation": false} {
		document := Plan{Campaign: "c", Items: []Item{{ID: "row", Title: "t", Class: class, Status: StatusOpen,
			Steps: []Step{{ID: "do", Title: "t", Status: StatusOpen, Verify: "go test ./x -run '^TestX$'"}}}}}
		if err := Validate(document); (err == nil) != valid {
			t.Fatalf("class %q validated %v, want valid=%t", class, err, valid)
		}
	}
}
