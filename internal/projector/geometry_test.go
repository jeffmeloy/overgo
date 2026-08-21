package projector

import (
	"slices"
	"testing"

	"overgo/internal/media"
)

func TestPixelMergePlan(t *testing.T) {
	const (
		mergeSize     = 2
		blocksPerAxis = 2
		gridSide      = blocksPerAxis * mergeSize
		valueWidth    = 1
	)
	plan, err := newPixelMergePlan(gridSide, gridSide, mergeSize)
	if err != nil {
		t.Fatal(err)
	}
	wantSets := [][]uint32{{0, 2, 8, 10}, {1, 3, 9, 11}, {4, 6, 12, 14}, {5, 7, 13, 15}}
	for index := range wantSets {
		if !slices.Equal(plan.indexSets[index], wantSets[index]) {
			t.Fatalf("index set %d = %v, want %v", index, plan.indexSets[index], wantSets[index])
		}
	}
	values := make([]float32, gridSide*gridSide*valueWidth)
	for index := range values {
		values[index] = float32(index)
	}
	merged, err := plan.shuffle(values, valueWidth)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15}
	if !slices.Equal(merged, want) {
		t.Fatalf("merged = %v, want %v", merged, want)
	}
}

func TestPixelMergePlanRejectsInvalidGeometry(t *testing.T) {
	const (
		mergeSize     = 2
		alignedSide   = 2 * mergeSize
		unalignedSide = alignedSide - 1
		identityMerge = 1
	)
	maxInt := int(^uint(0) >> 1)
	tests := []struct {
		name                 string
		height, width, merge int
	}{
		{name: "unaligned", height: unalignedSide, width: alignedSide, merge: mergeSize},
		{name: "overflow", height: maxInt, width: mergeSize, merge: identityMerge},
	}
	for _, test := range tests {
		if _, err := newPixelMergePlan(test.height, test.width, test.merge); err == nil {
			t.Fatalf("%s merge geometry was accepted", test.name)
		}
	}
}

func TestSplitTemporalPatchPairs(t *testing.T) {
	const (
		rows      = 1
		patchArea = 2
	)
	values := []float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
	first, second, width, err := splitTemporalPatchPairs(values, rows, patchArea)
	if err != nil {
		t.Fatal(err)
	}
	if width != media.RGBChannels*patchArea || !slices.Equal(first, []float32{0, 1, 4, 5, 8, 9}) ||
		!slices.Equal(second, []float32{2, 3, 6, 7, 10, 11}) {
		t.Fatalf("split = width %d, %v / %v", width, first, second)
	}
	if _, _, _, err := splitTemporalPatchPairs(values[:len(values)-1], rows, patchArea); err == nil {
		t.Fatal("short temporal patch tensor was accepted")
	}
}
