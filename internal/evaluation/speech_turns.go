package evaluation

import (
	"cmp"
	"context"
	"errors"
	"math"
	"slices"

	"overgo/internal/binaryschema"
	"overgo/internal/checked"
	"overgo/internal/recipecontract"
)

const (
	// Binary64 exactly represents consecutive integers through 2^53.
	speakerExactIntegerLimit = 1 << 53
	// Event storage: sample, index, delta and padded boolean, each bounded by
	// one uint64 slot. Interval duration and endpoint-index samples add one each.
	speakerEventScalarSlots    = 4
	speakerIntervalScalarSlots = 1
	speakerBoundaryScalarSlots = 1
	// Six primal-dual arrays, one assignment result and two active-count arrays.
	speakerMatchingScalarArrays = 9
)

// SpeakerMatch records a global, evaluation-only permutation. Reference names
// must never be applied to the model's input or presented as recognized people.
type SpeakerMatch struct {
	Predicted      string `json:"predicted"`
	Reference      string `json:"reference"`
	OverlapSamples uint64 `json:"overlap_samples"`
}

// SpeechTurnScore measures zero-collar, overlap-inclusive diarization error.
// The denominator is reference speaker-time, not recording time. Silence and
// unmapped predicted speakers still contribute false alarms. Same-speaker
// overlapping intervals are treated as a union rather than counted twice.
type SpeechTurnScore struct {
	ScoredSamples            uint64              `json:"scored_samples"`
	ReferenceSpeakerSamples  uint64              `json:"reference_speaker_samples"`
	MissedSpeakerSamples     uint64              `json:"missed_speaker_samples"`
	FalseAlarmSpeakerSamples uint64              `json:"false_alarm_speaker_samples"`
	ConfusedSpeakerSamples   uint64              `json:"confused_speaker_samples"`
	ReferenceOverlapSamples  uint64              `json:"reference_overlap_samples"`
	DiarizationErrorRate     float64             `json:"diarization_error_rate"`
	ReferenceSpeakers        int                 `json:"reference_speakers"`
	PredictedSpeakers        int                 `json:"predicted_speakers"`
	Mapping                  []SpeakerMatch      `json:"mapping"`
	ReferenceBoundaries      SpeechBoundaryError `json:"reference_boundaries"`
	PredictedBoundaries      SpeechBoundaryError `json:"predicted_boundaries"`
}

type speakerEvent struct {
	sample       uint64
	index, delta int
	predicted    bool
}
type speakerInterval struct {
	samples              uint64
	reference, predicted []int
}

// ScoreSpeechTurns scores every sample in span, with one maximum-overlap
// speaker permutation for the entire interval. It refuses source mismatches,
// out-of-span turns and a zero reference denominator. memoryBytes bounds
// numeric event/matching storage, not string metadata or allocator overhead.
func ScoreSpeechTurns(ctx context.Context, reference, predicted recipecontract.SpeechTurns, span recipecontract.SampleSpan, memoryBytes uint64) (SpeechTurnScore, error) {
	var result SpeechTurnScore
	if ctx == nil {
		return result, errors.New("speaker evaluation: missing context")
	}
	if err := reference.Validate(); err != nil {
		return result, err
	}
	if err := predicted.Validate(); err != nil {
		return result, err
	}
	if err := span.Validate(); err != nil {
		return result, err
	}
	if reference.Source != predicted.Source {
		return result, errors.New("speaker evaluation: source differs")
	}
	names := func(turns []recipecontract.SpeechTurn) []string {
		out := make([]string, 0, len(turns))
		for _, turn := range turns {
			out = append(out, turn.Speaker)
		}
		slices.Sort(out)
		return slices.Compact(out)
	}
	rNames, pNames := names(reference.Turns), names(predicted.Turns)
	n := max(len(rNames), len(pNames))
	if n == 0 {
		return result, errors.New("speaker evaluation: reference speaker-time absent")
	}
	// Float64 matching potentials must retain integer-sample distinctions.
	// n+1 additions each bounded by the interval duration stay exactly integral.
	if uint64(n) >= speakerExactIntegerLimit || span.End-span.Start > speakerExactIntegerLimit/uint64(n+1) {
		return result, errors.New("speaker evaluation: sample extent exceeds exact matching precision")
	}
	square, ok := checked.MulInt(n, n)
	if !ok {
		return result, errors.New("speaker evaluation: assignment extent overflows")
	}
	eventCount, ok := checked.AddInt(len(reference.Turns), len(predicted.Turns))
	if !ok {
		return result, errors.New("speaker evaluation: event extent overflows")
	}
	eventCount, ok = checked.MulInt(eventCount, 2)
	if !ok {
		return result, errors.New("speaker evaluation: event extent overflows")
	}
	// Events occupy four machine-width slots. Each retained interval owns one
	// duration and at most two n-wide active-index arrays. Nine n+1 slots cover
	// assignment scratch, its result and both active-count arrays.
	intervalSlots, ok := checked.Mul64(uint64(n), 2)
	if !ok {
		return result, errors.New("speaker evaluation: interval storage overflows")
	}
	// One additional slot per event covers both directional endpoint indexes.
	intervalSlots, ok = checked.Add64(intervalSlots, speakerEventScalarSlots, speakerIntervalScalarSlots, speakerBoundaryScalarSlots)
	if !ok {
		return result, errors.New("speaker evaluation: interval storage overflows")
	}
	eventSlots, ok := checked.Mul64(uint64(eventCount), intervalSlots)
	if !ok {
		return result, errors.New("speaker evaluation: event storage overflows")
	}
	workSlots, ok := checked.Mul64(uint64(n+1), speakerMatchingScalarArrays)
	if !ok {
		return result, errors.New("speaker evaluation: matching storage overflows")
	}
	slots, ok := checked.Add64(eventSlots, workSlots, uint64(square))
	if !ok || slots > memoryBytes/binaryschema.Uint64Bytes {
		return result, errors.New("speaker evaluation: byte budget exceeded")
	}
	events := make([]speakerEvent, 0, eventCount)
	for _, group := range []struct {
		turns     []recipecontract.SpeechTurn
		names     []string
		predicted bool
	}{{reference.Turns, rNames, false}, {predicted.Turns, pNames, true}} {
		for _, turn := range group.turns {
			if turn.Span.Start < span.Start || turn.Span.End > span.End {
				return result, errors.New("speaker evaluation: turn outside scored source interval")
			}
			index, _ := slices.BinarySearch(group.names, turn.Speaker)
			events = append(events, speakerEvent{turn.Span.Start, index, 1, group.predicted}, speakerEvent{turn.Span.End, index, -1, group.predicted})
		}
	}
	slices.SortFunc(events, func(a, b speakerEvent) int {
		return cmp.Compare(a.sample, b.sample)
	})
	rActive, pActive := make([]int, len(rNames)), make([]int, len(pNames))
	weights := make([]uint64, square)
	intervals := make([]speakerInterval, 0, eventCount)
	prior := span.Start
	for at := 0; at < len(events); {
		if err := ctx.Err(); err != nil {
			return result, err
		}
		position := events[at].sample
		if position > prior {
			interval := speakerInterval{samples: position - prior, reference: make([]int, 0, len(rNames)), predicted: make([]int, 0, len(pNames))}
			for i, count := range rActive {
				if count > 0 {
					interval.reference = append(interval.reference, i)
				}
			}
			for i, count := range pActive {
				if count > 0 {
					interval.predicted = append(interval.predicted, i)
				}
			}
			for _, r := range interval.reference {
				for _, p := range interval.predicted {
					weights[p*n+r] += interval.samples
				}
			}
			intervals = append(intervals, interval)
		}
		for at < len(events) && events[at].sample == position {
			event := events[at]
			if event.predicted {
				pActive[event.index] += event.delta
			} else {
				rActive[event.index] += event.delta
			}
			at++
		}
		prior = position
	}
	assignment, err := speakerAssignment(ctx, weights, n)
	if err != nil {
		return result, err
	}
	result.ScoredSamples = span.End - span.Start
	result.ReferenceSpeakers = len(rNames)
	result.PredictedSpeakers = len(pNames)
	for p := range pNames {
		r := assignment[p]
		if r < len(rNames) && weights[p*n+r] > 0 {
			result.Mapping = append(result.Mapping, SpeakerMatch{pNames[p], rNames[r], weights[p*n+r]})
		}
	}
	add := func(target *uint64, count int, duration uint64) bool {
		value, ok := checked.Mul64(uint64(count), duration)
		if !ok {
			return false
		}
		*target, ok = checked.Add64(*target, value)
		return ok
	}
	for _, interval := range intervals {
		r, p, correct := len(interval.reference), len(interval.predicted), 0
		for _, speaker := range interval.predicted {
			if slices.Contains(interval.reference, assignment[speaker]) {
				correct++
			}
		}
		if !add(&result.ReferenceSpeakerSamples, r, interval.samples) || !add(&result.MissedSpeakerSamples, max(0, r-p), interval.samples) || !add(&result.FalseAlarmSpeakerSamples, max(0, p-r), interval.samples) || !add(&result.ConfusedSpeakerSamples, min(r, p)-correct, interval.samples) {
			return SpeechTurnScore{}, errors.New("speaker evaluation: speaker-time overflows")
		}
		if r > 1 {
			result.ReferenceOverlapSamples += interval.samples
		}
	}
	if result.ReferenceSpeakerSamples == 0 {
		return SpeechTurnScore{}, errors.New("speaker evaluation: reference speaker-time absent")
	}
	errorSamples, ok := checked.Add64(result.MissedSpeakerSamples, result.FalseAlarmSpeakerSamples, result.ConfusedSpeakerSamples)
	if !ok {
		return SpeechTurnScore{}, errors.New("speaker evaluation: error speaker-time overflows")
	}
	result.DiarizationErrorRate = float64(errorSamples) / float64(result.ReferenceSpeakerSamples)
	rToP, pToR := make(map[string]string), make(map[string]string)
	for _, match := range result.Mapping {
		rToP[match.Reference], pToR[match.Predicted] = match.Predicted, match.Reference
	}
	result.ReferenceBoundaries, err = scoreSpeechBoundaries(ctx, reference.Turns, predicted.Turns, rToP)
	if err != nil {
		return SpeechTurnScore{}, err
	}
	result.PredictedBoundaries, err = scoreSpeechBoundaries(ctx, predicted.Turns, reference.Turns, pToR)
	if err != nil {
		return SpeechTurnScore{}, err
	}
	return result, ctx.Err()
}

// Primal-dual assignment minimizes negative overlap. Dummy zero-overlap rows
// and columns handle unequal speaker counts; ordered scans resolve ties.
func speakerAssignment(ctx context.Context, weights []uint64, n int) ([]int, error) {
	u, v := make([]float64, n+1), make([]float64, n+1)
	matched, previous := make([]int, n+1), make([]int, n+1)
	minimum, used := make([]float64, n+1), make([]bool, n+1)
	for row := 1; row <= n; row++ {
		matched[0] = row
		column := 0
		clear(used)
		for j := range minimum {
			minimum[j] = math.Inf(1)
		}
		for {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			used[column] = true
			active := matched[column]
			delta, next := math.Inf(1), 0
			for j := 1; j <= n; j++ {
				if used[j] {
					continue
				}
				cost := -float64(weights[(active-1)*n+j-1]) - u[active] - v[j]
				if cost < minimum[j] {
					minimum[j] = cost
					previous[j] = column
				}
				if minimum[j] < delta {
					delta, next = minimum[j], j
				}
			}
			for j := 0; j <= n; j++ {
				if used[j] {
					u[matched[j]] += delta
					v[j] -= delta
				} else {
					minimum[j] -= delta
				}
			}
			column = next
			if matched[column] == 0 {
				break
			}
		}
		for {
			prior := previous[column]
			matched[column] = matched[prior]
			column = prior
			if column == 0 {
				break
			}
		}
	}
	assignment := make([]int, n)
	for column := 1; column <= n; column++ {
		assignment[matched[column]-1] = column - 1
	}
	return assignment, nil
}
