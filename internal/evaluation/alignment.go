package evaluation

import (
	"errors"
	"slices"

	"overgo/internal/recipecontract"
)

// AlignmentScore measures agreement with independent reference boundaries.
// Samples are in the shared source's units. Annotation precision and provenance
// remain the reference producer's responsibility; agreement is not proof of
// human-verified timing. No word or boundary is dropped from the denominator.
type AlignmentScore struct {
	Words                  int     `json:"words"`
	Boundaries             int     `json:"boundaries"`
	MeanAbsoluteSamples    float64 `json:"mean_absolute_samples"`
	MedianAbsoluteSamples  float64 `json:"median_absolute_samples"`
	MaximumAbsoluteSamples uint64  `json:"maximum_absolute_samples"`
}

// ScoreAlignment compares complete, identically conditioned word sequences.
// A source, transcript, count or text mismatch refuses the comparison rather
// than scoring only matching words. Both interval lists must be canonical.
func ScoreAlignment(reference, predicted recipecontract.TimestampedAlignment) (AlignmentScore, error) {
	if err := reference.Validate(); err != nil {
		return AlignmentScore{}, err
	}
	if err := predicted.Validate(); err != nil {
		return AlignmentScore{}, err
	}
	if reference.Source != predicted.Source || reference.Transcription != predicted.Transcription || len(reference.Items) == 0 || len(reference.Items) != len(predicted.Items) {
		return AlignmentScore{}, errors.New("alignment evaluation: source, transcript or complete word count differs")
	}
	var distances []uint64
	for index, want := range reference.Items {
		got := predicted.Items[index]
		if want.Text != got.Text {
			return AlignmentScore{}, errors.New("alignment evaluation: ordered word text differs")
		}
		for _, pair := range [][2]uint64{{want.Span.Start, got.Span.Start}, {want.Span.End, got.Span.End}} {
			distances = append(distances, max(pair[0], pair[1])-min(pair[0], pair[1]))
		}
	}
	slices.Sort(distances)
	result := AlignmentScore{Words: len(reference.Items), Boundaries: len(distances), MaximumAbsoluteSamples: distances[len(distances)-1]}
	for _, distance := range distances {
		result.MeanAbsoluteSamples += float64(distance) / float64(len(distances))
	}
	// Every word contributes exactly two endpoints, so the count is even.
	result.MedianAbsoluteSamples = float64(distances[len(distances)/2-1])/2 + float64(distances[len(distances)/2])/2
	return result, nil
}
