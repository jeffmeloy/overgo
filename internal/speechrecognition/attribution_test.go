package speechrecognition

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipecontract"
	"overgo/internal/testutil"
)

func TestSpeakerWordAttributionPreservesAlignment(t *testing.T) {
	source := recipecontract.AudioReference{Audio: testutil.ArtifactID(t, artifact.KindFile, "audio"), Profile: testutil.ArtifactID(t, artifact.KindProfile, "format")}
	alignment := recipecontract.TimestampedAlignment{Source: source, Transcription: testutil.ArtifactID(t, artifact.KindOutput, "transcript"), Items: []recipecontract.AlignedText{
		{Span: recipecontract.SampleSpan{Start: 0, End: 10}, Text: "First", Confidence: .75},
		{Span: recipecontract.SampleSpan{Start: 10, End: 20}, Text: "Second!", Confidence: .5},
		{Span: recipecontract.SampleSpan{Start: 20, End: 30}, Text: "uncovered", Confidence: .25},
	}}
	turns := recipecontract.SpeechTurns{Source: source, Turns: []recipecontract.SpeechTurn{
		{Speaker: "z", Span: recipecontract.SampleSpan{Start: 0, End: 12}},
		{Speaker: "a", Span: recipecontract.SampleSpan{Start: 5, End: 20}},
		{Speaker: "z", Span: recipecontract.SampleSpan{Start: 6, End: 10}},
	}}
	for _, test := range []struct {
		name     string
		activity *recipecontract.ActivitySegments
		want     [][]string
	}{
		{"overlap", nil, [][]string{{"a", "z"}, {"a", "z"}, {}}},
		{"activity", &recipecontract.ActivitySegments{Source: source, Segments: []recipecontract.ActivitySegment{{Span: recipecontract.SampleSpan{Start: 12, End: 20}}}}, [][]string{{}, {"a"}, {}}},
		{"empty_activity", &recipecontract.ActivitySegments{Source: source}, [][]string{{}, {}, {}}},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, err := AttributeWords(t.Context(), alignment, turns, test.activity)
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != len(alignment.Items) {
				t.Fatal("lost words")
			}
			for i, word := range got {
				if word.Word != alignment.Items[i] || !reflect.DeepEqual(word.Speakers, test.want[i]) {
					t.Fatalf("word %d: %+v", i, word)
				}
			}
		})
	}
	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(context.Canceled)
	if _, err := AttributeWords(ctx, alignment, turns, nil); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	other := source
	other.Audio = testutil.ArtifactID(t, artifact.KindFile, "other")
	if _, err := AttributeWords(t.Context(), alignment, turns, &recipecontract.ActivitySegments{Source: other}); err == nil {
		t.Fatal("activity source mismatch accepted")
	}
	turns.Source = other
	if _, err := AttributeWords(t.Context(), alignment, turns, nil); err == nil {
		t.Fatal("speaker source mismatch accepted")
	}
}
