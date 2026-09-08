package speechrecognition

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"reflect"
	"testing"

	"overgo/internal/recipecontract"
)

func TestSpeakerBoundaryIndependentOracle(t *testing.T) {
	data, err := os.ReadFile("testdata/speaker_boundaries.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Name          string
		Probabilities []float32
		MinFrames     uint64 `json:"min_frames"`
		PadFrames     uint64 `json:"pad_frames"`
		Spans         []recipecontract.SampleSpan
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	if len(cases) != 5 {
		t.Fatal("oracle denominator differs")
	}
	for _, test := range cases {
		t.Run(test.Name, func(t *testing.T) {
			config := SpeakerBoundary{Threshold: .5, FrameSamples: 1280, TickSamples: 160, PadSamples: test.PadFrames * 1280, MinimumSamples: test.MinFrames * 1280, FinalBoundary: "last-tick-clipped-to-source"}
			turns, err := speakerTurns(t.Context(), test.Probabilities, len(test.Probabilities), 1, uint64(len(test.Probabilities))*1280, config)
			if err != nil {
				t.Fatal(err)
			}
			if len(turns) != len(test.Spans) {
				t.Fatalf("turns %+v want %+v", turns, test.Spans)
			}
			for i, turn := range turns {
				if turn.Span != test.Spans[i] || turn.Speaker != "speaker-0" {
					t.Fatalf("turn %+v want %+v", turn, test.Spans[i])
				}
			}
		})
	}
}

func TestSpeakerBoundary(t *testing.T) {
	base := SpeakerBoundary{Threshold: .5, FrameSamples: 8, TickSamples: 1, FinalBoundary: "last-tick-clipped-to-source"}
	for _, test := range []struct {
		name         string
		values       []float32
		samples      uint64
		pad, minimum uint64
		want         []recipecontract.SampleSpan
	}{
		{"threshold_equality_retains_state", []float32{.5, .6, .5, .4}, 32, 0, 0, []recipecontract.SampleSpan{{Start: 8, End: 24}}},
		{"final_tick", []float32{.4, .6}, 16, 0, 0, []recipecontract.SampleSpan{{Start: 8, End: 15}}},
		{"source_clip", []float32{.4, .6}, 12, 0, 0, []recipecontract.SampleSpan{{Start: 8, End: 12}}},
		{"pad_merge_then_filter", []float32{.6, .4, .6, .4}, 32, 4, 20, []recipecontract.SampleSpan{{Start: 0, End: 28}}},
		{"minimum_refuses", []float32{.6, .4}, 16, 0, 9, nil},
		{"inactive", []float32{0, .5}, 16, 0, 0, nil},
	} {
		t.Run(test.name, func(t *testing.T) {
			config := base
			config.PadSamples = test.pad
			config.MinimumSamples = test.minimum
			turns, err := speakerTurns(t.Context(), test.values, len(test.values), 1, test.samples, config)
			if err != nil {
				t.Fatal(err)
			}
			var spans []recipecontract.SampleSpan
			for _, turn := range turns {
				if turn.Speaker != "speaker-0" || turn.Confidence != 0 {
					t.Fatal(turn)
				}
				spans = append(spans, turn.Span)
			}
			if !reflect.DeepEqual(spans, test.want) {
				t.Fatalf("spans=%v want=%v", spans, test.want)
			}
		})
	}
	for _, values := range [][]float32{{float32(math.NaN())}, {float32(math.Inf(1))}, {-.1}, {1.1}} {
		if _, err := speakerTurns(t.Context(), values, 1, 1, 8, base); err == nil {
			t.Fatal("invalid probability accepted")
		}
	}
	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(context.Canceled)
	if _, err := speakerTurns(ctx, []float32{.6}, 1, 1, 8, base); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	overflow := base
	overflow.FrameSamples = math.MaxUint64
	overflow.TickSamples = 1
	if _, err := speakerTurns(t.Context(), []float32{.6, .6}, 2, 1, 16, overflow); err == nil {
		t.Fatal("overflow accepted")
	}
}
