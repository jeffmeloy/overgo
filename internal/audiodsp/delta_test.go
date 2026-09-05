package audiodsp

import (
	"context"
	"errors"
	"math"
	"slices"
	"testing"
)

func TestAppendFrameDeltas(t *testing.T) {
	for _, probe := range []struct {
		name          string
		input, want   []float32
		bands, radius int
	}{
		{"ramp", []float32{0, 2, 4}, []float32{0, 1, 2, 2, 4, 1}, 1, 1},
		{"two-bands", []float32{0, 10, 2, 8, 4, 6}, []float32{0, 10, 1, -1, 2, 8, 2, -2, 4, 6, 1, -1}, 2, 1},
		{"wide-window", []float32{0, 10}, []float32{0, 3, 10, 3}, 1, 2},
		{"singleton", []float32{7}, []float32{7, 0}, 1, math.MaxInt},
	} {
		t.Run(probe.name, func(t *testing.T) {
			original := slices.Clone(probe.input)
			out := make([]float32, len(probe.want))
			if err := AppendFrameDeltas(t.Context(), out, probe.input, probe.bands, probe.radius); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(out, probe.want) || !slices.Equal(original, probe.input) {
				t.Fatalf("got %v, want %v; source %v", out, probe.want, probe.input)
			}
			if allocations := testing.AllocsPerRun(100, func() {
				if err := AppendFrameDeltas(t.Context(), out, probe.input, probe.bands, probe.radius); err != nil {
					panic(err)
				}
			}); allocations != 0 {
				t.Fatalf("allocations %g", allocations)
			}
		})
	}
}

func TestAppendFrameDeltasRefusal(t *testing.T) {
	input := []float32{1, 2}
	out := make([]float32, 4)
	for _, probe := range []struct {
		input, output []float32
		bands, radius int
	}{
		{input, out, 0, 1}, {input, out, 3, 1}, {input, out, 1, 0},
		{input, out[:3], 1, 1}, {nil, nil, 1, 1},
		{[]float32{float32(math.NaN()), 0}, out, 1, 1},
		{out[:2], out, 1, 1},
	} {
		if err := AppendFrameDeltas(t.Context(), probe.output, probe.input, probe.bands, probe.radius); err == nil {
			t.Fatal("accepted invalid input")
		}
	}
	if err := AppendFrameDeltas(nil, out, input, 1, 1); err == nil {
		t.Fatal("accepted nil context")
	}
	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(nil)
	if err := AppendFrameDeltas(ctx, out, input, 1, 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation: %v", err)
	}
}
