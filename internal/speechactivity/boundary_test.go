package speechactivity

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"
)

func TestBoundaryRefusalsAndCancellation(t *testing.T) {
	config := BoundaryConfig{Smoothing: 3, Threshold: .5, MinSpeech: 2, MaxSpeech: 6, MinSilence: 2}
	p, err := NewBoundaryPolicy(config, 3*8)
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"window", "threshold", "speech", "silence", "maximum", "budget"} {
		t.Run(name, func(t *testing.T) {
			bad, budget := config, uint64(3*8)
			switch name {
			case "window":
				bad.Smoothing = 0
			case "threshold":
				bad.Threshold = math.NaN()
			case "speech":
				bad.MinSpeech = 0
			case "silence":
				bad.MinSilence = 0
			case "maximum":
				bad.MaxSpeech = 1
			case "budget":
				budget--
			}
			if _, err := NewBoundaryPolicy(bad, budget); err == nil {
				t.Fatal("accepted invalid boundary declaration")
			}
		})
	}
	for _, state := range []BoundaryState{{Frame: 1}, {Frame: 1, Window: []float64{math.NaN()}}, {Phase: boundaryPhase(4)}, {Final: true}, {Next: 1}, {Sum: math.Inf(1)}, {Start: 1}} {
		if err := p.Process(t.Context(), nil, &state, false, nil); err == nil {
			t.Fatal("accepted inconsistent checkpoint")
		}
	}
	for _, value := range []float32{-1, 2, float32(math.NaN()), float32(math.Inf(1))} {
		var state BoundaryState
		if err := p.Process(t.Context(), []float32{value}, &state, false, nil); err == nil || !reflect.DeepEqual(state, BoundaryState{}) {
			t.Fatal("invalid probability advanced state")
		}
	}
	ctx, cancel := context.WithCancelCause(t.Context())
	var state BoundaryState
	err = p.Process(ctx, []float32{1, 1, 1}, &state, false, func(Boundary) error { cancel(context.Canceled); return nil })
	if !errors.Is(err, context.Canceled) || state.Frame != 1 {
		t.Fatal("cancellation did not stop frame processing")
	}
}

func TestOfflineShortSignalClipsExtension(t *testing.T) {
	config := OfflineConfig{Smoothing: 1, Threshold: .5, MaxSpeech: 4, ExtendSpeech: 100}
	got, err := OfflineDecisions(t.Context(), []float32{1}, config, 2)
	if err != nil || len(got) != 1 || !got[0] {
		t.Fatalf("short signal extension=%v: %v", got, err)
	}
	if _, err := OfflineDecisions(t.Context(), []float32{1}, config, 1); err == nil {
		t.Fatal("accepted insufficient decision budget")
	}
	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(context.Canceled)
	if _, err := OfflineDecisions(ctx, []float32{1}, config, 2); !errors.Is(err, context.Canceled) {
		t.Fatal("ignored cancellation")
	}
}
