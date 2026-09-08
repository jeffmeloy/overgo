package speechrecognition

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"slices"

	"overgo/internal/checked"
	"overgo/internal/recipecontract"
)

// Speaker interval behavior is ported from audio.cpp
// 3497b7cc44753e2c141d8fe60ac42cec433e3281, Copyright 2026 ShugoAI LLC,
// Apache-2.0 (licenses/audio.cpp.txt). This implementation uses exact declared
// sample-grid arithmetic without repeated score arrays or decimal round trips.
// Unlike the source, published spans cannot extend beyond the actual audio.
func speakerTurns(ctx context.Context, probabilities []float32, frames, speakers int, samples uint64, config SpeakerBoundary) ([]recipecontract.SpeechTurn, error) {
	if ctx == nil || frames <= 0 || speakers <= 0 || samples == 0 {
		return nil, errors.New("speaker boundary: invalid extent")
	}
	if err := config.validate(); err != nil {
		return nil, err
	}
	count, ok := checked.MulInt(frames, speakers)
	if !ok || len(probabilities) != count {
		return nil, errors.New("speaker boundary: probability shape differs")
	}
	extent, ok := checked.Mul64(uint64(frames), config.FrameSamples)
	if !ok {
		return nil, errors.New("speaker boundary: frame extent overflows")
	}
	if _, ok := checked.Add64(extent, config.PadSamples); !ok {
		return nil, errors.New("speaker boundary: padded extent overflows")
	}
	for _, p := range probabilities {
		if !checked.UnitInterval64(float64(p)) {
			return nil, errors.New("speaker boundary: probability outside unit interval")
		}
	}
	var turns []recipecontract.SpeechTurn
	for speaker := range speakers {
		var spans []recipecontract.SampleSpan
		active := false
		var start uint64
		appendSpan := func(end uint64) {
			s := recipecontract.SampleSpan{Start: start - min(start, config.PadSamples), End: end + config.PadSamples}
			if s.End <= s.Start {
				return
			}
			if len(spans) > 0 && spans[len(spans)-1].End >= s.Start {
				spans[len(spans)-1].End = max(spans[len(spans)-1].End, s.End)
			} else {
				spans = append(spans, s)
			}
		}
		for frame := range frames {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			p := probabilities[frame*speakers+speaker]
			at := uint64(frame) * config.FrameSamples
			if active && p < config.Threshold {
				appendSpan(at)
				active = false
			} else if !active && p > config.Threshold {
				start = at
				active = true
			}
		}
		if active {
			appendSpan(extent - config.TickSamples)
		}
		for _, span := range spans {
			if span.End-span.Start < config.MinimumSamples {
				continue
			}
			span.End = min(span.End, samples)
			if span.Start >= span.End {
				continue
			}
			turns = append(turns, recipecontract.SpeechTurn{Span: span, Speaker: fmt.Sprintf("speaker-%d", speaker)})
		}
	}
	slices.SortStableFunc(turns, func(a, b recipecontract.SpeechTurn) int {
		return cmp.Or(cmp.Compare(a.Span.Start, b.Span.Start), cmp.Compare(a.Speaker, b.Speaker))
	})
	return turns, ctx.Err()
}
