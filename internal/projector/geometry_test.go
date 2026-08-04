package projector

import (
	"slices"
	"testing"
)

func TestPixelMergePlan(t *testing.T) {
	plan, err := newPixelMergePlan(4, 4, 2)
	if err != nil {
		t.Fatal(err)
	}
	wantSets := [][]uint32{{0, 2, 8, 10}, {1, 3, 9, 11}, {4, 6, 12, 14}, {5, 7, 13, 15}}
	for index := range wantSets {
		if !slices.Equal(plan.indexSets[index], wantSets[index]) {
			t.Fatalf("index set %d = %v, want %v", index, plan.indexSets[index], wantSets[index])
		}
	}
	values := make([]float32, 16)
	for index := range values {
		values[index] = float32(index)
	}
	merged, err := plan.shuffle(values, 1)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15}
	if !slices.Equal(merged, want) {
		t.Fatalf("merged = %v, want %v", merged, want)
	}
}

func TestPixelMergePlanRejectsInvalidGeometry(t *testing.T) {
	if _, err := newPixelMergePlan(3, 4, 2); err == nil {
		t.Fatal("unaligned merge geometry was accepted")
	}
	if _, err := newPixelMergePlan(int(^uint(0)>>1), 2, 1); err == nil {
		t.Fatal("overflowing merge geometry was accepted")
	}
}

func TestSplitTemporalPatchPairs(t *testing.T) {
	values := []float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
	first, second, width, err := splitTemporalPatchPairs(values, 1, 2)
	if err != nil {
		t.Fatal(err)
	}
	if width != 6 || !slices.Equal(first, []float32{0, 1, 4, 5, 8, 9}) ||
		!slices.Equal(second, []float32{2, 3, 6, 7, 10, 11}) {
		t.Fatalf("split = width %d, %v / %v", width, first, second)
	}
	if _, _, _, err := splitTemporalPatchPairs(values[:len(values)-1], 1, 2); err == nil {
		t.Fatal("short temporal patch tensor was accepted")
	}
}
