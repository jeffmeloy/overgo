package representation

import (
	"slices"
	"testing"

	"overgo/internal/testcheck"
)

func TestRelativePositionBuckets(t *testing.T) {
	got := testcheck.Must(RelativePositionBuckets(2, 2, 4, 8, true))(t)
	if !slices.Equal(got, []int{0, 3, 1, 0}) {
		t.Fatalf("buckets=%v", got)
	}
}
