package evaluation

import (
	"context"
	"errors"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipecontract"
	"overgo/internal/testutil"
)

func TestSpeechTurnScore(t *testing.T) {
	source := recipecontract.AudioReference{Audio: testutil.ArtifactID(t, artifact.KindFile, "speech"), Profile: testutil.ArtifactID(t, artifact.KindProfile, "format")}
	turn := func(name string, start, end uint64) recipecontract.SpeechTurn {
		return recipecontract.SpeechTurn{Speaker: name, Span: recipecontract.SampleSpan{Start: start, End: end}}
	}
	base := []recipecontract.SpeechTurn{turn("A", 0, 10), turn("B", 10, 20)}
	for _, test := range []struct {
		name                                              string
		reference, predicted                              []recipecontract.SpeechTurn
		denominator, miss, falseAlarm, confusion, overlap uint64
	}{
		{"permutation", base, []recipecontract.SpeechTurn{turn("y", 0, 10), turn("x", 10, 20)}, 20, 0, 0, 0, 0},
		{"miss", base, []recipecontract.SpeechTurn{turn("x", 0, 5), turn("y", 10, 20)}, 20, 5, 0, 0, 0},
		{"false_alarm_in_silence", base, []recipecontract.SpeechTurn{turn("x", 0, 10), turn("y", 10, 25)}, 20, 0, 5, 0, 0},
		{"global_not_per_turn_mapping", base, []recipecontract.SpeechTurn{turn("x", 0, 20)}, 20, 0, 0, 10, 0},
		{"overlap", []recipecontract.SpeechTurn{turn("A", 0, 20), turn("B", 5, 15)}, []recipecontract.SpeechTurn{turn("x", 0, 20)}, 30, 10, 0, 0, 10},
		{"same_speaker_union", []recipecontract.SpeechTurn{turn("A", 0, 20), turn("A", 5, 15), turn("B", 5, 15)}, []recipecontract.SpeechTurn{turn("x", 0, 20)}, 30, 10, 0, 0, 10},
		{"no_predictions", base, nil, 20, 20, 0, 0, 0},
	} {
		t.Run(test.name, func(t *testing.T) {
			score, err := ScoreSpeechTurns(t.Context(), recipecontract.SpeechTurns{Source: source, Turns: test.reference}, recipecontract.SpeechTurns{Source: source, Turns: test.predicted}, recipecontract.SampleSpan{End: 30}, 1<<20)
			if err != nil {
				t.Fatal(err)
			}
			if score.ScoredSamples != 30 || score.ReferenceSpeakerSamples != test.denominator || score.MissedSpeakerSamples != test.miss || score.FalseAlarmSpeakerSamples != test.falseAlarm || score.ConfusedSpeakerSamples != test.confusion || score.ReferenceOverlapSamples != test.overlap || score.DiarizationErrorRate != float64(test.miss+test.falseAlarm+test.confusion)/float64(test.denominator) {
				t.Fatalf("score=%+v", score)
			}
			if score.ReferenceBoundaries.Matched+score.ReferenceBoundaries.Unmatched != uint64(2*len(test.reference)) || score.PredictedBoundaries.Matched+score.PredictedBoundaries.Unmatched != uint64(2*len(test.predicted)) {
				t.Fatal("lost boundary denominator")
			}
			if test.name == "miss" && (score.ReferenceBoundaries.TotalErrorSamples != 5 || score.PredictedBoundaries.TotalErrorSamples != 5 || score.ReferenceBoundaries.MaximumErrorSamples != 5) {
				t.Fatalf("boundary distance: %+v", score)
			}
		})
	}
	reference := recipecontract.SpeechTurns{Source: source, Turns: base}
	predicted := reference
	for _, test := range []struct {
		name   string
		change func(*recipecontract.SpeechTurns, *recipecontract.SampleSpan, *uint64)
	}{
		{"source", func(p *recipecontract.SpeechTurns, _ *recipecontract.SampleSpan, _ *uint64) {
			p.Source.Audio = testutil.ArtifactID(t, artifact.KindFile, "different")
		}},
		{"extent", func(_ *recipecontract.SpeechTurns, s *recipecontract.SampleSpan, _ *uint64) { s.End = 15 }},
		{"budget", func(_ *recipecontract.SpeechTurns, _ *recipecontract.SampleSpan, m *uint64) { *m = 1 }},
	} {
		t.Run(test.name, func(t *testing.T) {
			p := predicted
			s := recipecontract.SampleSpan{End: 30}
			m := uint64(1 << 20)
			test.change(&p, &s, &m)
			if _, err := ScoreSpeechTurns(t.Context(), reference, p, s, m); err == nil {
				t.Fatal("invalid comparison accepted")
			}
		})
	}
	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(context.Canceled)
	if _, err := ScoreSpeechTurns(ctx, reference, predicted, recipecontract.SampleSpan{End: 30}, 1<<20); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if _, err := ScoreSpeechTurns(t.Context(), recipecontract.SpeechTurns{Source: source}, predicted, recipecontract.SampleSpan{End: 30}, 1<<20); err == nil {
		t.Fatal("zero reference denominator accepted")
	}
}

func TestSpeakerAssignment(t *testing.T) {
	assignment, err := speakerAssignment(t.Context(), []uint64{9, 8, 8, 0}, 2)
	if err != nil || !slices.Equal(assignment, []int{1, 0}) {
		t.Fatalf("non-greedy assignment=%v err=%v", assignment, err)
	}
	// Exhaust all three-speaker binary overlap matrices and compare the dual
	// algorithm with an independent enumeration of all six permutations.
	for bits := range 1 << 9 {
		weights := make([]uint64, 9)
		for i := range weights {
			weights[i] = uint64(bits >> i & 1)
		}
		assignment, err := speakerAssignment(t.Context(), weights, 3)
		if err != nil {
			t.Fatal(err)
		}
		var best uint64
		for _, p := range [][3]int{{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}} {
			best = max(best, weights[p[0]]+weights[3+p[1]]+weights[6+p[2]])
		}
		if weights[assignment[0]]+weights[3+assignment[1]]+weights[6+assignment[2]] != best {
			t.Fatalf("assignment differs for matrix %v", weights)
		}
	}
}
