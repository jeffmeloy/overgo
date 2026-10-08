package evaluation

import (
	"math"
	"testing"

	"overgo/internal/trainingprogram"
)

// TestHeldoutReadingStatesWhatAConstantAnswerScores holds what an answer that
// reads no input scores on a held-out measure, where the measure has one.
// Under label accuracy it is always answering the commonest label, or the
// commonest class of a record's rows, each record scored as its report scores
// it: a report of labels two thirds of which are one label says two thirds, a
// model that answers nothing but that label scores exactly it, and a report of
// rows says the mean share of the commonest class. A forecast error has no
// such answer. A suite whose every case is a choice with one answer has
// chance; one a case of which is answered in words, or weighs several
// alternatives, has none.
func TestHeldoutReadingStatesWhatAConstantAnswerScores(t *testing.T) {
	t.Parallel()
	table := newHeldoutFixture(t, trainingprogram.MetricTableAccuracy)
	if len(table.view.Records) < 3 {
		t.Fatalf("the fixture holds out %d records", len(table.view.Records))
	}
	score := func(label func(index int) string, classes func(index int) []float64, answer string, predicted []float64) HeldoutReport {
		t.Helper()
		var cases []HeldoutCase
		var observations []HeldoutObservation
		for index, record := range table.view.Records {
			target := HeldoutCase{Record: record.ID}
			observation := HeldoutObservation{Record: record.ID}
			if label != nil {
				target.Label, observation.Label = label(index), answer
			} else {
				target.Values, observation.Values = classes(index), predicted
			}
			cases, observations = append(cases, target), append(observations, observation)
		}
		plan, err := CompileHeldoutPlan(table.objective, table.view, cases)
		if err != nil {
			t.Fatal(err)
		}
		report, err := plan.Score(observations)
		if err != nil {
			t.Fatal(err)
		}
		return report
	}
	// Every third record expects bad and the rest good.
	good := 0
	labels := score(func(index int) string {
		if index%3 == 0 {
			return "bad"
		}
		good++
		return "good"
	}, nil, "good", nil)
	want := float64(good) / float64(len(table.view.Records))
	value, found := reportMetric(labels.metrics, string(numericLabelAccuracy))
	if labels.constant == nil || *labels.constant != want || !found || value.Value != want {
		t.Fatalf("labels of which %d of %d are good read a constant answer of %v, and always answering good scores %v: want both %g", good, len(table.view.Records), labels.constant, value.Value, want)
	}
	// Each record has three rows, two of class 1 and one of class 0.
	rows := score(nil, func(int) []float64 { return []float64{1, 0, 1} }, "", []float64{0, 0, 0})
	if rows.constant == nil || math.Abs(*rows.constant-2.0/3) > 1e-15 {
		t.Fatalf("rows two thirds of which are one class read a constant answer of %v", rows.constant)
	}
	forecast := newHeldoutFixture(t, trainingprogram.MetricForecastMAE)
	if report := forecast.score(t, 1, "falling"); report.constant != nil {
		t.Fatalf("a forecast error reads a constant answer of %g", *report.constant)
	}

	for name, test := range map[string]struct {
		suite string
		want  float64
	}{
		"options of four and of ten": {`{"cases":[{"options":["a","b","c","d"],"answer":1},{"options":["a","b","c","d","e","f","g","h","i","j"],"answer":0}]}`, (0.25 + 0.1) / 2},
		"candidates of two":          {`{"cases":[{"candidates":["a","b"],"answer":0}]}`, 0.5},
		"a case answered in words":   {`{"cases":[{"options":["a","b"],"answer":0},{"prompt":"say it","answers":["it"]}]}`, 0},
		"a case that weighs several": {`{"cases":[{"candidates":["a","b","c"],"positive":[true,true,false]}]}`, 0},
		"no cases":                   {`{"cases":[]}`, 0},
	} {
		if got := choiceChance([]byte(test.suite)); math.Abs(got-test.want) > 1e-15 {
			t.Errorf("%s: chance reads %g, want %g", name, got, test.want)
		}
	}
}
