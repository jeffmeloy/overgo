package speechrecognition

import (
	"context"
	"math"
	"slices"
	"testing"
)

func TestGreedyCTC(t *testing.T) {
	for _, tc := range []struct {
		name         string
		logits       []float32
		frames, want []int
	}{
		{"blank separates repeats", []float32{0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1}, []int{1, 1, 0, 1, 2}, []int{1, 1, 2}},
		{"lowest tie", []float32{0, 1, 1, 1, 1, 1}, []int{1, 0}, []int{1}},
		{"all blank", []float32{1, 0, 0, 1, 0, 0}, []int{0, 0}, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			frames := make([]int, len(tc.frames))
			tokens := make([]int, len(frames))
			got, err := GreedyCTC(t.Context(), frames, tokens, tc.logits, len(frames), 3, 0)
			if err != nil || !slices.Equal(got, tc.want) || !slices.Equal(frames, tc.frames) {
				t.Fatalf("frames=%v tokens=%v error=%v", frames, got, err)
			}
		})
	}
}

func TestGreedyCTCRefusal(t *testing.T) {
	for _, logits := range [][]float32{nil, {float32(math.NaN())}, {float32(math.Inf(1))}} {
		if _, err := GreedyCTC(t.Context(), make([]int, 1), make([]int, 1), logits, 1, 1, 0); err == nil {
			t.Fatal("accepted invalid logits")
		}
	}
	ids := make([]int, 1)
	if _, err := GreedyCTC(t.Context(), ids, ids, []float32{1}, 1, 1, 0); err == nil {
		t.Fatal("accepted alias")
	}
	if _, err := GreedyCTC(nil, ids, make([]int, 1), []float32{1}, 1, 1, 0); err == nil {
		t.Fatal("accepted nil context")
	}
	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(nil)
	if _, err := GreedyCTC(ctx, ids, make([]int, 1), []float32{1}, 1, 1, 0); err == nil {
		t.Fatal("accepted cancellation")
	}
	if _, err := GreedyCTC(t.Context(), ids, make([]int, 1), []float32{1}, 1, 1, 1); err == nil {
		t.Fatal("accepted invalid blank")
	}
}
