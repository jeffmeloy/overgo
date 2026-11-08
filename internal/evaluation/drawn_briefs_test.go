package evaluation

import (
	"fmt"
	"slices"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestDrawnBriefsAreWrittenBySeededCode holds the drawn-text suite to as many pictures as it asks for: on 2026-11-08
// its 20 held briefs capped it at 20, at most 20 pairs a paired test could read. A suite of 200 holds the 20 held briefs
// first, then 180 written by seeded code, each named once and asking five questions, a title among its three swaps and
// four figures among their turned neighbours, as a held brief does; one seed writes the same suite every time, and
// another writes another.
func TestDrawnBriefsAreWrittenBySeededCode(t *testing.T) {
	t.Parallel()
	held := testcheck.Must(drawnBriefs())(t)
	const asked = 200
	items := testcheck.Must(PickDrawnText(asked, 17))(t)
	again := testcheck.Must(PickDrawnText(asked, 17))(t)
	other := testcheck.Must(PickDrawnText(asked, 18))(t)
	if len(items) != asked || fmt.Sprint(items) != fmt.Sprint(again) || fmt.Sprint(items[len(held):]) == fmt.Sprint(other[len(held):]) {
		t.Fatalf("seed 17 picked %d briefs, the same twice %t, and seed 18 the same written ones %t; want %d, the same, and others",
			len(items), fmt.Sprint(items) == fmt.Sprint(again), fmt.Sprint(items[len(held):]) == fmt.Sprint(other[len(held):]), asked)
	}
	var ids []string
	for index, item := range items {
		isHeld := slices.ContainsFunc(held, func(brief drawnBrief) bool { return brief.ID == item.ID })
		if isHeld != (index < len(held)) || slices.Contains(ids, item.ID) || len(item.Questions) != nearTexts+titleSwaps {
			t.Fatalf("brief %d, %s, is held %t, repeated %t, with %d questions; want the held ones first, each once, with %d questions",
				index, item.ID, isHeld, slices.Contains(ids, item.ID), len(item.Questions), nearTexts+titleSwaps)
		}
		for _, question := range item.Questions {
			if len(question.Choices) != nearTexts+1 || !slices.Contains(question.Choices, question.Answer) || !strings.Contains(item.Caption, question.Answer) {
				t.Fatalf("brief %s asks %+v, want its answer among four choices and in its prompt", item.ID, question)
			}
		}
		ids = append(ids, item.ID)
	}
}
