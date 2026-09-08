package speechrecognition

import (
	"context"
	"errors"
	"slices"
	"sort"

	"overgo/internal/recipecontract"
)

// AttributedWord preserves the exact aligned text, span and confidence. Speakers
// contains every local label with positive sample overlap, sorted and unique.
// An empty list means no attributed speaker; multiple labels remain ambiguous.
// Alignment confidence is not speaker confidence.
type AttributedWord struct {
	Word     recipecontract.AlignedText `json:"word"`
	Speakers []string                   `json:"speakers"`
}

// AttributeWords composes existing source-bound analysis outputs without
// changing words or selecting an unscored winning speaker. Optional activity
// restricts contributing speaker samples, never the word inventory. All inputs
// must refer to exactly the same audio and format profile. No cross-recording
// identity or reference-label mapping is applied.
func AttributeWords(ctx context.Context, alignment recipecontract.TimestampedAlignment, turns recipecontract.SpeechTurns, activity *recipecontract.ActivitySegments) ([]AttributedWord, error) {
	if ctx == nil {
		return nil, errors.New("speaker attribution: missing context")
	}
	if err := alignment.Validate(); err != nil {
		return nil, err
	}
	if err := turns.Validate(); err != nil {
		return nil, err
	}
	if alignment.Source != turns.Source {
		return nil, errors.New("speaker attribution: source differs")
	}
	if activity != nil {
		if err := activity.Validate(); err != nil {
			return nil, err
		}
		if activity.Source != alignment.Source {
			return nil, errors.New("speaker attribution: activity source differs")
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	words := make([]AttributedWord, len(alignment.Items))
	for i, item := range alignment.Items {
		words[i].Word = item
		words[i].Speakers = []string{}
	}
	for _, turn := range turns.Turns {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		first := sort.Search(len(words), func(i int) bool { return words[i].Word.Span.End > turn.Span.Start })
		for i := first; i < len(words) && words[i].Word.Span.Start < turn.Span.End; i++ {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			span := recipecontract.SampleSpan{Start: max(turn.Span.Start, words[i].Word.Span.Start), End: min(turn.Span.End, words[i].Word.Span.End)}
			if activity != nil {
				at := sort.Search(len(activity.Segments), func(j int) bool { return activity.Segments[j].Span.End > span.Start })
				if at == len(activity.Segments) || activity.Segments[at].Span.Start >= span.End {
					continue
				}
			}
			words[i].Speakers = append(words[i].Speakers, turn.Speaker)
		}
	}
	for i := range words {
		slices.Sort(words[i].Speakers)
		words[i].Speakers = slices.Compact(words[i].Speakers)
	}
	return words, ctx.Err()
}
