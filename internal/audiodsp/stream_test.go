package audiodsp

import (
	"context"
	"encoding/json"
	"math"
	"reflect"
	"slices"
	"testing"
)

func streamConfig() FrontendConfig {
	c := testFrontendConfig()
	c.FrameSpan, c.WindowOffset, c.PadLeft, c.PadRight = 4, 0, 0, 0
	c.Condition = &FrameCondition{Gain: 2, RemoveMean: true, Preemphasis: .5}
	c.Normalize = &NormalizeConfig{Mode: "fixed", Mean: []float64{1, 2, 3}, InverseStd: []float64{2, 3, 4}}
	return c
}

func TestStreamFrontendChunkRestart(t *testing.T) {
	c := streamConfig()
	offline, err := NewFrontend(c, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	stream, err := NewStreamFrontend(c, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	samples := []float32{1, 2, 3, 4, -1, -2, -3, -4, 2, 1, 0}
	var whole Workspace
	want, count, err := offline.Process(t.Context(), [][]float32{samples}, c.SampleRate, &whole, ProcessOptions{})
	if err != nil {
		t.Fatal(err)
	}
	for chunk := 1; chunk <= len(samples); chunk++ {
		var state StreamState
		var w StreamWorkspace
		var got []float32
		for start := 0; start < len(samples); start += chunk {
			end := min(start+chunk, len(samples))
			encoded, err := json.Marshal(state)
			if err != nil {
				t.Fatal(err)
			}
			var restart StreamState
			if err := json.Unmarshal(encoded, &restart); err != nil {
				t.Fatal(err)
			}
			before := slices.Clone(restart.Tail)
			// A fresh workspace at every boundary proves no hidden framing state.
			w = StreamWorkspace{}
			features, _, next, err := stream.Process(t.Context(), samples[start:end], c.SampleRate, restart, false, &w)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(before, restart.Tail) {
				t.Fatal("mutated persisted state")
			}
			got, state = append(got, features...), next
			if cap(w.tail) != c.FrameSpan-1 {
				t.Fatal("unbounded retained prefix")
			}
		}
		if !slices.Equal(got, want) || state.Frames != uint64(count) || state.Samples != uint64(len(samples)) {
			t.Fatalf("chunk %d differs", chunk)
		}
		features, frames, final, err := stream.Process(t.Context(), nil, c.SampleRate, state, true, &w)
		if err != nil || len(features) != 0 || frames != 0 || !final.Final {
			t.Fatalf("final flush: %v", err)
		}
		if _, _, _, err := stream.Process(t.Context(), nil, c.SampleRate, final, true, &w); err == nil {
			t.Fatal("accepted repeated finalization")
		}
	}
	// Reusing borrowed tail must also preserve overlap under in-place copy.
	var w StreamWorkspace
	var state StreamState
	var got []float32
	for _, value := range samples {
		features, _, next, err := stream.Process(t.Context(), []float32{value}, c.SampleRate, state, false, &w)
		if err != nil {
			t.Fatal(err)
		}
		got, state = append(got, features...), next
	}
	if !slices.Equal(got, want) {
		t.Fatal("borrowed overlap continuation differs")
	}
}

func TestStreamFrontendRefusals(t *testing.T) {
	c := streamConfig()
	for _, name := range []string{"padding", "global-log", "global-normalization", "resampling", "gapped-hop", "budget"} {
		t.Run(name, func(t *testing.T) {
			bad, budget := c, uint64(1<<20)
			switch name {
			case "padding":
				bad.PadLeft = 1
			case "global-log":
				value := 1.
				bad.Log.DynamicRange = &value
			case "global-normalization":
				bad.Normalize = &NormalizeConfig{Mode: "all", Epsilon: 1}
			case "resampling":
				bad.ResampleTaps = []float64{1}
			case "gapped-hop":
				bad.Geometry.HopSamples = uint64(c.FrameSpan + 1)
			case "budget":
				budget = 1
			}
			if _, err := NewStreamFrontend(bad, budget); err == nil {
				t.Fatal("accepted unsupported live declaration")
			}
		})
	}
	p, err := NewStreamFrontend(c, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	for _, state := range []StreamState{{Samples: 1}, {Frames: 1}, {Tail: []float32{1}}, {Samples: 1, Tail: []float32{float32(math.NaN())}}, {Final: true}} {
		var w StreamWorkspace
		if _, _, _, err := p.Process(t.Context(), []float32{1}, c.SampleRate, state, false, &w); err == nil {
			t.Fatal("accepted inconsistent restart")
		}
	}
	var w StreamWorkspace
	_, _, state, err := p.Process(t.Context(), []float32{1}, c.SampleRate, StreamState{}, false, &w)
	if err != nil {
		t.Fatal(err)
	}
	before := slices.Clone(state.Tail)
	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(context.Canceled)
	_, _, result, err := p.Process(ctx, []float32{2}, c.SampleRate, state, false, &w)
	if err == nil || !reflect.DeepEqual(result, StreamState{}) || !slices.Equal(state.Tail, before) {
		t.Fatal("canceled admission changed state")
	}
	if _, _, _, err := p.Process(t.Context(), state.Tail, c.SampleRate, state, false, &w); err == nil {
		t.Fatal("accepted input alias")
	}
}
