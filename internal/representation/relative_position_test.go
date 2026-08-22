package representation

import (
	"slices"
	"testing"
)

func TestRelativePositionBuckets(t *testing.T) {
	got, err := RelativePositionBuckets(2, 2, 4, 8, true)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, []int{0, 3, 1, 0}) {
		t.Fatalf("buckets=%v", got)
	}
}
