package representation

import (
	"slices"
	"testing"
)

func TestAssemblePaddedSequence(t *testing.T) {
	got, err := AssemblePaddedSequence([]int{1}, []int{2}, []int{3}, 3, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got.IDs, []int{1, 2, 0, 3}) || !slices.Equal(got.Mask, []bool{true, true, false, true}) || got.PromptRows != 3 {
		t.Fatalf("sequence=%+v", got)
	}
	if _, err := AssemblePaddedSequence(nil, []int{2}, []int{3}, 3, 0); err == nil {
		t.Fatal("accepted empty prefix")
	}
}

func TestActivePrefixRows(t *testing.T) {
	if rows, err := ActivePrefixRows(2, 4); err != nil || rows != 3 {
		t.Fatalf("rows=%d err=%v", rows, err)
	}
	if rows, err := ActivePrefixRows(4, 4); err != nil || rows != 4 {
		t.Fatalf("full rows=%d err=%v", rows, err)
	}
}
