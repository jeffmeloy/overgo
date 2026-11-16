package tensorstats

import "testing"

// TestGroupedSignTailReadsGroupsAsUnits holds the grouped tail to its null, every group's sign flipped at once:
// where each group holds one pair it is the exact paired test's tail, 5 wins and 1 loss 7/64, summed here with no
// rounding; eight losses read as eight independent pairs give 1/256 and as two clips of four questions each 1/4, the
// two signs alone; nets of 3 and -1 total 2, which half the four sign choices reach; and groups that net nothing say
// nothing, a tail of 1.
func TestGroupedSignTailReadsGroupsAsUnits(t *testing.T) {
	t.Parallel()
	for _, table := range []struct {
		name string
		nets []int64
		want float64
	}{
		{"one pair a group", []int64{1, 1, 1, 1, 1, -1}, 7.0 / 64},
		{"eight pairs alone", []int64{1, 1, 1, 1, 1, 1, 1, 1}, 1.0 / 256},
		{"eight pairs in two groups", []int64{4, 4}, 1.0 / 4},
		{"nets of 3 and -1", []int64{3, -1}, 1.0 / 2},
		{"nothing netted", []int64{0, 0}, 1},
	} {
		if got := GroupedSignTail(table.nets); got != table.want {
			t.Errorf("%s: the grouped tail of %v is %v, want %v", table.name, table.nets, got, table.want)
		}
	}
}
