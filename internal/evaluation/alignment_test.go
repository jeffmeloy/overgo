package evaluation

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipecontract"
	"overgo/internal/testutil"
)

func TestAlignmentScore(t *testing.T) {
	reference := recipecontract.TimestampedAlignment{
		Source:        recipecontract.AudioReference{Audio: testutil.ArtifactID(t, artifact.KindFile, "speech"), Profile: testutil.ArtifactID(t, artifact.KindProfile, "format")},
		Transcription: testutil.ArtifactID(t, artifact.KindOutput, "transcript"),
		Items:         []recipecontract.AlignedText{{Span: recipecontract.SampleSpan{Start: 10, End: 20}, Text: "one"}, {Span: recipecontract.SampleSpan{Start: 30, End: 50}, Text: "two"}},
	}
	prediction := reference
	prediction.Items = []recipecontract.AlignedText{{Span: recipecontract.SampleSpan{Start: 11, End: 23}, Text: "one"}, {Span: recipecontract.SampleSpan{Start: 32, End: 54}, Text: "two"}}
	score, err := ScoreAlignment(reference, prediction)
	if err != nil || score != (AlignmentScore{Words: 2, Boundaries: 4, MeanAbsoluteSamples: 2.5, MedianAbsoluteSamples: 2.5, MaximumAbsoluteSamples: 4}) {
		t.Fatalf("score=%+v: %v", score, err)
	}
	for _, mutate := range []func(*recipecontract.TimestampedAlignment){
		func(p *recipecontract.TimestampedAlignment) { p.Items = nil },
		func(p *recipecontract.TimestampedAlignment) { p.Items = p.Items[:1] },
		func(p *recipecontract.TimestampedAlignment) {
			p.Source.Audio = testutil.ArtifactID(t, artifact.KindFile, "other")
		},
		func(p *recipecontract.TimestampedAlignment) {
			p.Transcription = testutil.ArtifactID(t, artifact.KindOutput, "other")
		},
		func(p *recipecontract.TimestampedAlignment) {
			p.Items = []recipecontract.AlignedText{{Span: recipecontract.SampleSpan{Start: 10, End: 20}, Text: "changed"}, reference.Items[1]}
		},
	} {
		changed := reference
		mutate(&changed)
		if _, err := ScoreAlignment(reference, changed); err == nil {
			t.Fatal("partial or mismatched comparison accepted")
		}
	}
}
