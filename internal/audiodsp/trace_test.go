package audiodsp

import (
	"errors"
	"math"
	"slices"
	"testing"
)

func TestFrameSelectionPrecedesGlobalTransform(t *testing.T) {
	config := testFrontendConfig()
	config.Window = "rectangular"
	config.Log.DynamicRange = new(float64)
	// Zero range replaces every value with the selected maximum. The loud
	// tail makes trimming after this transformation observably incorrect.
	samples := []float32{1, 1, 1, 1, 1000, 1000, 1000, 1000}
	plan, err := NewFrontend(config, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	var workspace Workspace
	options := ProcessOptions{FrameLimit: 1}
	selected, frames, err := plan.Process(t.Context(), [][]float32{samples}, config.SampleRate, &workspace, options)
	if err != nil {
		t.Fatal(err)
	}
	if frames != options.FrameLimit {
		t.Fatalf("frames=%d", frames)
	}
	selected = slices.Clone(selected)
	all, _, err := plan.Process(t.Context(), [][]float32{samples}, config.SampleRate, &workspace, ProcessOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if slices.Equal(selected, all[:len(selected)]) {
		t.Fatal("selection did not precede global clipping")
	}
	var maximum float32 = -math.MaxFloat32
	options.Observe = func(trace FrameTrace) error {
		maximum = max(maximum, slices.Max(trace.LogMel))
		return nil
	}
	got, _, err := plan.Process(t.Context(), [][]float32{samples}, config.SampleRate, &workspace, options)
	if err != nil {
		t.Fatal(err)
	}
	for _, value := range got {
		if value != maximum {
			t.Fatalf("clipped value=%g, selected maximum=%g", value, maximum)
		}
	}
}

func TestFrameTraceRefusalAndBudget(t *testing.T) {
	config := testFrontendConfig()
	plan, err := NewFrontend(config, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	chunks := [][]float32{{1, 2, 3, 4, 5, 6, 7, 8}}
	var workspace Workspace
	sentinel := errors.New("observer refused")
	for _, options := range []ProcessOptions{
		{FrameLimit: -1}, {FrameLimit: math.MaxInt},
		{Observe: func(FrameTrace) error { return sentinel }},
	} {
		values, frames, err := plan.Process(t.Context(), chunks, config.SampleRate, &workspace, options)
		if err == nil || values != nil || frames != 0 {
			t.Fatalf("refusal returned values=%v frames=%d error=%v", values, frames, err)
		}
		if options.Observe != nil && !errors.Is(err, sentinel) {
			t.Fatalf("lost observer error: %v", err)
		}
	}
	if _, _, err := plan.Process(t.Context(), chunks, config.SampleRate, &workspace, ProcessOptions{}); err != nil {
		t.Fatal(err)
	}
	// Restrict the same plan to its already-reserved numeric bytes, excluding
	// the optional mel trace backing array. Existing trace capacity must count.
	plan.memoryBytes = plan.tableBytes + uint64(cap(workspace.window)+cap(workspace.real)+cap(workspace.imaginary)+cap(workspace.magnitude)+cap(workspace.accum)+cap(workspace.envelope))*float64Bytes + uint64(cap(workspace.features)+cap(workspace.waveform)+cap(workspace.joined)+cap(workspace.resampled))*float32Bytes
	if values, _, err := plan.Process(t.Context(), chunks, config.SampleRate, &workspace, ProcessOptions{}); err == nil || values != nil {
		t.Fatal("untraced call ignored retained trace capacity")
	}
}

func TestFrameTraceSteadyStateAllocations(t *testing.T) {
	plan, err := NewFrontend(testFrontendConfig(), 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	chunks := [][]float32{{1, 2, 3, 4, 5, 6, 7, 8}}
	var workspace Workspace
	options := ProcessOptions{Observe: func(FrameTrace) error { return nil }}
	ctx := t.Context()
	run := func() {
		if _, _, err := plan.Process(ctx, chunks, plan.config.SampleRate, &workspace, options); err != nil {
			t.Fatal(err)
		}
	}
	run()
	if allocations := testing.AllocsPerRun(100, run); allocations != 0 {
		t.Fatalf("trace allocations=%g", allocations)
	}
}
