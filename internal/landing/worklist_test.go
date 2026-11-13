package landing

import (
	"slices"
	"testing"

	"overgo/internal/testcheck"
)

// TestRefusalsAreListedByTheRuleThatRefusedThem holds the worklist to the refusals the record keeps: two attempts
// refused by the pins rule, one of whose refusals said its edit, are listed as the rule with 2 (1 with an edit), ahead
// of a test that failed once, and the refusal the record kept no cause for, as it kept none before
// session-leftovers/ended-session, is not listed.
func TestRefusalsAreListedByTheRuleThatRefusedThem(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	for _, refused := range []string{
		"gate: land: invariants: failed tests/packages: overgo/internal/p: TestPHolds, overgo/internal/p",
		"gate: land: rules: changes bytes a record pins: internal/p/p.go (pinned by docs/verification/p.json)",
		"gate: land: rules: changes bytes a record pins: internal/p/q.go (pinned by docs/verification/p.json; edit: sed -i s/a/b/ docs/verification/p.json)",
	} {
		testcheck.Must(RecordRefusal(root, Classify("row/do", refused)))(t)
	}
	testcheck.Must(RecordRefusal(root, Refusal{Reference: "row/do", Class: "lanes"}))(t)
	listed := testcheck.Must(Worklist(root))(t)
	got := make([][3]any, 0, len(listed))
	for _, item := range listed {
		got = append(got, [3]any{item.By, item.Refusals, item.Edits})
	}
	want := [][3]any{{"pins", 2, 1}, {"overgo/internal/p: TestPHolds", 1, 0}}
	if !slices.Equal(got, want) {
		t.Fatalf("the worklist is %v, want %v", got, want)
	}
}
