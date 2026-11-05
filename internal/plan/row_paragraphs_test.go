package plan

import (
	"slices"
	"testing"

	"overgo/internal/testcheck"
)

// TestARowParagraphLeavesWithItsLastRow advances the steps of two rows under paragraphs that govern one row, both rows,
// no row that is left, the standing doctrine and the campaign: a row's paragraph stays while any row it governs is in
// the plan and leaves with the last of them, one governing only rows already gone leaves at the next advance, and the
// standing and campaign paragraphs stay. Every session reads the paragraphs, so one that outlives its rows misleads
// every session after them.
func TestARowParagraphLeavesWithItsLastRow(t *testing.T) {
	t.Parallel()
	step := func(id string) Step { return Step{ID: id, Status: StatusOpen, Verify: "go test ./..."} }
	document := Plan{
		Items: []Item{
			{ID: "first", Status: StatusOpen, Steps: []Step{step("one"), step("two")}},
			{ID: "second", Status: StatusOpen, Steps: []Step{step("one")}},
		},
		Doctrine: []Paragraph{
			{Governs: governsRow, Rows: []string{"first"}, Text: "first"},
			{Governs: governsRow, Rows: []string{"first", "second"}, Text: "both"},
			{Governs: governsRow, Rows: []string{"gone"}, Text: "gone"},
			{Governs: governsStanding, Text: "standing"},
			{Governs: governsCampaign, Text: "campaign"},
		},
	}
	for _, advance := range []struct {
		item, step string
		left       []string
	}{
		{"first", "one", []string{"first", "both", "standing", "campaign"}},
		{"first", "two", []string{"both", "standing", "campaign"}},
		{"second", "one", []string{"standing", "campaign"}},
	} {
		advanced, err := Advance(document, advance.item, advance.step)
		testcheck.NoError(t, err)
		var left []string
		for _, paragraph := range advanced.Doctrine {
			left = append(left, paragraph.Text)
		}
		if !slices.Equal(left, advance.left) {
			t.Fatalf("after %s/%s the paragraphs are %v, want %v", advance.item, advance.step, left, advance.left)
		}
		document = advanced
	}
}
