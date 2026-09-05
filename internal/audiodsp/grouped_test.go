package audiodsp

import (
	"context"
	"math"
	"slices"
	"testing"
)

func TestProcessGroupedComposition(t *testing.T) {
	p, err := NewFrontend(testFrontendConfig(), 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name                                           string
		samples, stack, radius, frames, width, padding int
	}{
		{"even", 8, 2, 1, 2, 12, 0},
		{"odd", 6, 2, 1, 2, 12, 1},
		{"static", 8, 2, 0, 2, 6, 0},
		{"ungrouped", 8, 1, 0, 4, 3, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			input := []float32{1, 2, 3, 4, -1, -2, -3, -4}[:tc.samples]
			original := slices.Clone(input)
			config := GroupedFeatureConfig{StackFrames: tc.stack, DeltaRadius: tc.radius, FinalFrameSamples: 1}
			var w, reference Workspace
			got, frames, width, err := p.ProcessGrouped(t.Context(), input, 8, &w, config)
			if err != nil {
				t.Fatal(err)
			}
			if frames != tc.frames || width != tc.width || len(got) != frames*width || len(w.padding) != tc.padding {
				t.Fatalf("geometry=%d/%d/%d padding=%d", frames, width, len(got), len(w.padding))
			}
			padded := append(slices.Clone(input), make([]float32, tc.padding)...)
			static, _, err := p.Process(t.Context(), [][]float32{padded}, 8, &reference, ProcessOptions{FrameLimit: tc.frames * tc.stack})
			if err != nil {
				t.Fatal(err)
			}
			want := static
			if tc.radius > 0 {
				want = make([]float32, len(static)*2)
				if err := AppendFrameDeltas(t.Context(), want, static, p.bands, tc.radius); err != nil {
					t.Fatal(err)
				}
			}
			if !slices.Equal(got, want) || !slices.Equal(input, original) {
				t.Fatal("composition or input changed")
			}
			if w.groupedChunks[0] != nil || w.groupedChunks[1] != nil || cap(w.joined) != 0 {
				t.Fatal("input retained or unnecessarily copied")
			}
			ctx := t.Context()
			if allocs := testing.AllocsPerRun(100, func() {
				if _, _, _, err := p.ProcessGrouped(ctx, input, 8, &w, config); err != nil {
					panic(err)
				}
			}); allocs != 0 {
				t.Fatalf("warm allocations=%g", allocs)
			}
			for i := range w.padding {
				w.padding[i] = 123
			}
			got, _, _, err = p.ProcessGrouped(ctx, input, 8, &w, config)
			if err != nil || !slices.Equal(got, want) {
				t.Fatalf("padding not reset: %v", err)
			}
		})
	}
}

func TestProcessGroupedRefusalAndBudget(t *testing.T) {
	p, err := NewFrontend(testFrontendConfig(), 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	input := []float32{1, 2, 3, 4, 5, 6}
	config := GroupedFeatureConfig{StackFrames: 2, DeltaRadius: 1, FinalFrameSamples: 1}
	for _, invalid := range []GroupedFeatureConfig{{}, {StackFrames: 1, DeltaRadius: -1}, {StackFrames: 1, FinalFrameSamples: -1}, {StackFrames: math.MaxInt, FinalFrameSamples: 1}} {
		if _, _, _, err := p.ProcessGrouped(t.Context(), input, 8, &Workspace{}, invalid); err == nil {
			t.Fatalf("accepted %v", invalid)
		}
	}
	if _, _, _, err := new(Frontend).ProcessGrouped(t.Context(), input, 8, &Workspace{}, config); err == nil {
		t.Fatal("accepted zero frontend")
	}
	if _, _, _, err := p.ProcessGrouped(nil, input, 8, &Workspace{}, config); err == nil {
		t.Fatal("accepted nil context")
	}
	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(nil)
	if _, _, _, err := p.ProcessGrouped(ctx, input, 8, &Workspace{}, config); err == nil {
		t.Fatal("accepted cancellation")
	}
	if _, _, _, err := p.ProcessGrouped(t.Context(), input, 16, &Workspace{}, config); err == nil {
		t.Fatal("accepted wrong sample rate")
	}
	if _, _, _, err := p.ProcessGrouped(t.Context(), input[:1], 8, &Workspace{}, config); err == nil {
		t.Fatal("accepted empty frame selection")
	}
	var w Workspace
	got, _, _, err := p.ProcessGrouped(t.Context(), input, 8, &w, config)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := p.ProcessGrouped(t.Context(), got, 8, &w, config); err == nil {
		t.Fatal("accepted borrowed output")
	}
	if _, _, err := p.Process(t.Context(), [][]float32{got}, 8, &w, ProcessOptions{}); err == nil {
		t.Fatal("ordinary frontend accepted borrowed grouped output")
	}
	bytes := p.tableBytes + uint64(cap(w.window)+cap(w.real)+cap(w.imaginary)+cap(w.magnitude))*float64Bytes + uint64(cap(w.features)+cap(w.sequence)+cap(w.padding))*float32Bytes
	p.memoryBytes = bytes
	if _, _, _, err := p.ProcessGrouped(t.Context(), input, 8, &w, config); err != nil {
		t.Fatal(err)
	}
	p.memoryBytes--
	if _, _, _, err := p.ProcessGrouped(t.Context(), input, 8, &w, config); err == nil {
		t.Fatal("accepted insufficient byte budget")
	}
	if _, _, err := p.Process(t.Context(), [][]float32{input}, 8, &w, ProcessOptions{FrameLimit: 4}); err == nil {
		t.Fatal("ordinary frontend omitted retained grouped buffers from budget")
	}
	var fresh Workspace
	if _, _, _, err := p.ProcessGrouped(t.Context(), input, 8, &fresh, config); err == nil {
		t.Fatal("accepted insufficient fresh byte budget")
	}
	if cap(fresh.sequence) != 0 || cap(fresh.padding) != 0 || cap(fresh.features) != 0 {
		t.Fatal("allocated before budget refusal")
	}
}
