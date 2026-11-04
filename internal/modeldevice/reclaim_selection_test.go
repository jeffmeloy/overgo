package modeldevice

import (
	"cmp"
	"container/list"
	"slices"
	"testing"
)

// TestAReclaimSortsNoMoreExpertsThanItMoves holds the victims a reclaim takes to what a stable sort of every card expert
// by coldness would put first, ties in list order, and a reclaim that moves one expert of a stride's thousand to one
// comparison an expert: sorting them all took about 2 ms a reclaim, 37 of them before a 16K read's first kernel.
func TestAReclaimSortsNoMoreExpertsThanItMoves(t *testing.T) {
	t.Parallel()
	order := list.New()
	var entries []*list.Element
	// Uses in a few values, so most experts tie with others.
	const experts, uses = 1000, 7
	for at := range experts {
		entries = append(entries, order.PushBack(at*at%uses))
	}
	compared := 0
	colder := func(a, b *list.Element) int {
		compared++
		return cmp.Compare(a.Value.(int), b.Value.(int))
	}
	sorted := slices.Clone(entries)
	slices.SortStableFunc(sorted, colder)
	for _, k := range []int{0, 1, 2, 5, 9, 10, 64, experts, experts + 1} {
		got := coldestFirst(slices.Clone(entries), k, colder)
		if want := sorted[:min(k, experts)]; !slices.Equal(got, want) {
			t.Fatalf("the %d coldest differ from a stable sort's first %d", k, len(want))
		}
	}
	compared = 0
	coldestFirst(slices.Clone(entries), 1, colder)
	if compared > experts {
		t.Fatalf("finding one victim among %d experts compared %d times, want at most one an expert", experts, compared)
	}
}
