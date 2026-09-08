package evaluation

import (
	"context"
	"errors"
	"slices"

	"overgo/internal/checked"
	"overgo/internal/recipecontract"
)

// SpeechBoundaryError reports directional nearest same-kind (start or end),
// same-mapped-speaker endpoint distances. Several endpoints may select the
// same target; this is not a one-to-one turn accuracy score. Every raw input
// endpoint is either matched or explicitly unmatched. TotalErrorSamples and
// MaximumErrorSamples cover only matched endpoints; the denominator is Matched,
// never Matched+Unmatched. Fragmentation is visible in the reverse direction.
type SpeechBoundaryError struct {
	Matched             uint64 `json:"matched"`
	Unmatched           uint64 `json:"unmatched"`
	TotalErrorSamples   uint64 `json:"total_error_samples"`
	MaximumErrorSamples uint64 `json:"maximum_error_samples"`
}

func scoreSpeechBoundaries(ctx context.Context, from, to []recipecontract.SpeechTurn, mapping map[string]string) (SpeechBoundaryError, error) {
	type endpoints struct{ starts, ends []uint64 }
	counts := make(map[string]int)
	for _, turn := range to {
		counts[turn.Speaker]++
	}
	index := make(map[string]*endpoints, len(counts))
	for speaker, count := range counts {
		index[speaker] = &endpoints{make([]uint64, 0, count), make([]uint64, 0, count)}
	}
	for _, turn := range to {
		points := index[turn.Speaker]
		points.starts = append(points.starts, turn.Span.Start)
		points.ends = append(points.ends, turn.Span.End)
	}
	for _, points := range index {
		slices.Sort(points.starts)
		slices.Sort(points.ends)
	}
	var result SpeechBoundaryError
	for _, turn := range from {
		if err := ctx.Err(); err != nil {
			return SpeechBoundaryError{}, err
		}
		points := index[mapping[turn.Speaker]]
		if points == nil {
			result.Unmatched += 2
			continue
		}
		for i, value := range [...]uint64{turn.Span.Start, turn.Span.End} {
			ordered := points.starts
			if i == 1 {
				ordered = points.ends
			}
			at, _ := slices.BinarySearch(ordered, value)
			delta := ^uint64(0)
			if at < len(ordered) {
				delta = ordered[at] - value
			}
			if at > 0 {
				delta = min(delta, value-ordered[at-1])
			}
			var ok bool
			result.TotalErrorSamples, ok = checked.Add64(result.TotalErrorSamples, delta)
			if !ok {
				return SpeechBoundaryError{}, errors.New("speaker evaluation: boundary error overflows")
			}
			result.MaximumErrorSamples = max(result.MaximumErrorSamples, delta)
			result.Matched++
		}
	}
	return result, ctx.Err()
}
