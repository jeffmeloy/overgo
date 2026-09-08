package hostmath

import (
	"math"
	"slices"
	"testing"
)

func TestDepthwiseMemoryDilatedContextAndFuture(t *testing.T) {
	// Two independent channels, with the second equal to ten times the first.
	x := []float32{1, 10, 2, 20, 3, 30, 4, 40}
	past := []float32{.5, 1, .5, 1}
	future := []float32{.25, .5, .25, .5}
	cache := []float32{-1, -10, -2, -20}
	dst := make([]float32, len(x))
	if err := DepthwiseMemoryF64(dst, x, past, future, cache, 4, 2, 2, 1); err != nil {
		t.Fatal(err)
	}
	want := []float32{3.5, 35, 5.75, 57.5, 7.5, 75, 9, 90}
	if !slices.Equal(dst, want) {
		t.Fatalf("memory filter %v, want %v", dst, want)
	}
	if err := DepthwiseMemoryF64(dst[:2], x[:2], past, future, cache, 1, 2, 2, 1); err != nil || !slices.Equal(dst[:2], []float32{1.5, 15}) {
		t.Fatalf("single frame has no future context: %v: %v", dst[:2], err)
	}
}

func TestDepthwiseMemoryCausalChunks(t *testing.T) {
	x := []float32{1, 10, 2, 20, 3, 30, 4, 40}
	weights := []float32{.5, 1, .25, 2}
	want := make([]float32, len(x))
	if err := DepthwiseMemoryF64(want, x, weights, nil, nil, 4, 2, 2, 0); err != nil {
		t.Fatal(err)
	}
	for split := 1; split < 4; split++ {
		got := make([]float32, len(x))
		if err := DepthwiseMemoryF64(got[:split*2], x[:split*2], weights, nil, nil, split, 2, 2, 0); err != nil {
			t.Fatal(err)
		}
		cache := x[max(0, split-2)*2 : split*2]
		if err := DepthwiseMemoryF64(got[split*2:], x[split*2:], weights, nil, cache, 4-split, 2, 2, 0); err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(got, want) {
			t.Fatalf("split %d changed output %v != %v", split, got, want)
		}
	}
}

func TestDepthwiseMemoryRefusals(t *testing.T) {
	for name, run := range map[string]func() error{
		"empty": func() error { return DepthwiseMemoryF64(nil, nil, nil, nil, nil, 0, 0, 0, 0) },
		"alias": func() error { x := []float32{1}; return DepthwiseMemoryF64(x, x, []float32{1}, nil, nil, 1, 1, 1, 0) },
		"context": func() error {
			return DepthwiseMemoryF64(make([]float32, 1), []float32{1}, []float32{1}, nil, []float32{1}, 1, 1, 1, 0)
		},
		"overflow": func() error { return DepthwiseMemoryF64(nil, nil, []float32{1}, nil, nil, math.MaxInt, 2, 1, 0) },
		"non-finite": func() error {
			return DepthwiseMemoryF64(make([]float32, 1), []float32{1}, []float32{float32(math.Inf(1))}, nil, nil, 1, 1, 1, 0)
		},
	} {
		t.Run(name, func(t *testing.T) {
			if run() == nil {
				t.Fatal("invalid filter accepted")
			}
		})
	}
}
