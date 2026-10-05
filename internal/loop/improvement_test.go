package loop

import (
	"errors"
	"fmt"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/scorecard"
	"overgo/internal/scorecard/metric"
	"overgo/internal/testutil"
)

// improvingWorld is an emptied plan whose scorecard run publishes the
// scripted record against the scripted baseline.
type improvingWorld struct {
	fakeWorld
	published, baseline *scorecard.Record
	measureErr          error
	promoted            []string
	proposed            []scorecard.Gap
	proposals           int
	// filled holds the commit each fill ran at and what was promoted and
	// proposed when it ran.
	filled []string
}

func (w *improvingWorld) Fill(commit string) {
	w.filled = append(w.filled, fmt.Sprintf("%s after promoted %v, proposals %d", commit, w.promoted, w.proposals))
}

func (w *improvingWorld) Measure() (*scorecard.Record, *scorecard.Record, error) {
	return w.published, w.baseline, w.measureErr
}

func (w *improvingWorld) Promote(commit string) error {
	w.promoted = append(w.promoted, commit)
	return nil
}

func (w *improvingWorld) Propose(_ scorecard.Record, gaps []scorecard.Gap) (int, error) {
	w.proposed = gaps
	w.proposals++
	return 0, nil
}

// triggeredWorld is an improving world whose trigger answers from a script: due says, for each
// look before a step, whether a cycle is due, and worth whether the cycle an empty plan would
// start has anything to measure.
type triggeredWorld struct {
	improvingWorld
	due   []bool
	looks int
	worth bool
}

func (w *triggeredWorld) CycleDue() (bool, string, error) {
	w.looks++
	return w.looks <= len(w.due) && w.due[w.looks-1], "scripted", nil
}

func (w *triggeredWorld) CycleWorthRunning() (bool, string, error) { return w.worth, "scripted", nil }

// TestDriverStartsATriggeredCycleBeforeTheNextStep holds a cycle the trigger calls due to run
// between steps, before the worker is dispatched, and once; to run no cycle while the loop is
// stopped; and an empty plan's cycle to run only when it is worth running.
func TestDriverStartsATriggeredCycleBeforeTheNextStep(t *testing.T) {
	t.Parallel()
	advance := func(w *fakeWorld, _ Step, _ string) string { w.steps = nil; return "" }
	published := &scorecard.Record{Commit: "b"}
	for name, test := range map[string]struct {
		due                    []bool
		worth, paused          bool
		wantFills, wantWorkers int
	}{
		"due before the step, nothing moved after":  {due: []bool{true}, wantFills: 1, wantWorkers: 1},
		"not due, an empty plan with nothing moved": {due: nil, wantFills: 0, wantWorkers: 1},
		"not due, an empty plan that is worth it":   {due: nil, worth: true, wantFills: 1, wantWorkers: 1},
		"due while the loop is stopped":             {due: []bool{true}, paused: true, wantFills: 0, wantWorkers: 0},
	} {
		world := &triggeredWorld{
			improvingWorld: improvingWorld{fakeWorld: fakeWorld{steps: []Step{{Item: "row", ID: "do"}}, behavior: []func(*fakeWorld, Step, string) string{advance}, paused: test.paused}, published: published},
			due:            test.due, worth: test.worth,
		}
		if _, err := Run(world, Config{MaxAttemptsPerStep: 1, MaxInvocations: 3}); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if len(world.filled) != test.wantFills || world.call != test.wantWorkers {
			t.Errorf("%s: %d cycles ran and %d workers, want %d and %d", name, len(world.filled), world.call, test.wantFills, test.wantWorkers)
		}
	}
}

// TestLoopPromotesOnlyOnHeldOutFitness empties the plan against scripted
// scorecards and holds promotion to the held-out quality cells alone, judged
// like for like against a baseline measured on main: a fall in the metric's
// own direction on the same suite refuses it, whatever else improved, while
// slower throughput, a failed attempt or a newly measured metric promote.
// A quality nothing compares -- another suite, or a baseline row the record
// lost -- is unknown and refuses it too. No baseline, a row without held-out
// quality and a failed validation promote nothing, and the outcome says why:
// a failed validation by its own refusal, naming what it did not acquire.
// Every run proposes its gaps: a refusal for a fall proposes the fall, and a
// refusal for an unknown quality proposes measuring it again. Every run that
// published fills the validations promotion does not compare once, at the
// commit its scorecard measured, after its decision and its proposals: the
// fill records its claims apart from the checkout the proposals write.
func TestLoopPromotesOnlyOnHeldOutFitness(t *testing.T) {
	t.Parallel()
	source := testutil.ArtifactID(t, artifact.KindEvidence, "loop-fitness")
	model := testutil.ArtifactID(t, artifact.KindModel, "loop-fitness")
	accuracy := func(value float64, suite string) scorecard.Cell {
		return scorecard.Cell{Value: value, Label: "accuracy, " + suite, Source: source, Direction: runrecord.DirectionMaximize,
			Contract: scorecard.Contract{Metric: "accuracy", Origin: "evaluation", Suite: suite, Protocol: "chat-template", Comparison: testutil.ArtifactID(t, artifact.KindProfile, "chat-acceptance/"+suite)}}
	}
	perplexity := func(value float64) scorecard.Cell {
		return scorecard.Cell{Value: value, Label: "perplexity, store/dna", Source: source, Direction: runrecord.DirectionMinimize,
			Contract: scorecard.Contract{Metric: "perplexity", Origin: "evaluation", Suite: "store/dna", Protocol: "raw-completion", Comparison: testutil.ArtifactID(t, artifact.KindProfile, "dna-raw-acceptance")}}
	}
	record := func(commit string, quality scorecard.Cell, throughput, failures float64) *scorecard.Record {
		cells := map[metric.Name]scorecard.Cell{}
		for _, column := range metric.Columns {
			cells[column] = scorecard.Cell{Value: 1, Source: source}
		}
		cells[metric.Quality] = quality
		cells[metric.Throughput] = scorecard.Cell{Value: throughput, Unit: "tok/s", Source: source}
		cells[metric.Failures] = scorecard.Cell{Value: failures, Label: "attempts", Source: source}
		return &scorecard.Record{Commit: commit, Rows: []scorecard.Row{{Model: model, Name: "model", Task: recipe.TaskInference, Cells: cells}}}
	}
	unmeasured := func(record *scorecard.Record, column metric.Name) *scorecard.Record {
		record.Rows[0].Cells[column] = scorecard.Cell{Reason: "not measured on main"}
		return record
	}
	const fell, unknown = "fell below the baseline", "on the baseline scorecard's models and contracts"
	for _, scenario := range []struct {
		name                string
		baseline, published *scorecard.Record
		measureErr          error
		withheld            string
	}{
		{name: "held quality, slower and failing", baseline: record("a", accuracy(0.5, "mmlu"), 40, 0), published: record("b", accuracy(0.5, "mmlu"), 30, 0.1)},
		{name: "accuracy fell while throughput rose", baseline: record("a", accuracy(0.5, "mmlu"), 40, 0), published: record("b", accuracy(0.4, "mmlu"), 50, 0), withheld: fell},
		{name: "perplexity rose", baseline: record("a", perplexity(900), 40, 0), published: record("b", perplexity(950), 40, 0), withheld: fell},
		{name: "perplexity fell", baseline: record("a", perplexity(900), 40, 0), published: record("b", perplexity(850), 40, 0)},
		{name: "another suite scores lower", baseline: record("a", accuracy(0.5, "mmlu"), 40, 0), published: record("b", accuracy(0.2, "mmlu-pro"), 40, 0), withheld: unknown},
		{name: "a vanished baseline row", baseline: record("a", accuracy(0.5, "mmlu"), 40, 0), published: &scorecard.Record{Commit: "b"}, withheld: unknown},
		{name: "a new optional metric", baseline: unmeasured(record("a", accuracy(0.5, "mmlu"), 40, 0), metric.Load), published: record("b", accuracy(0.5, "mmlu"), 40, 0)},
		{name: "no scorecard of main", published: record("b", accuracy(0.9, "mmlu"), 40, 0), withheld: "no earlier scorecard is recorded as held"},
		{name: "held-out quality missing", baseline: record("a", accuracy(0.5, "mmlu"), 40, 0), published: record("b", scorecard.Cell{Reason: "no held-out suite ran"}, 40, 0), withheld: "has no inference held-out quality: no held-out suite ran"},
		{name: "failed validation", baseline: record("a", accuracy(0.5, "mmlu"), 40, 0), published: record("b", accuracy(0.6, "mmlu"), 40, 0),
			measureErr: errors.New("validate -run: cycle lane models: overgo/internal/x: TestY skipped without declaring itself inapplicable"),
			withheld:   "a validation failed, so the scorecard's cells are stale: validate -run: cycle lane models: overgo/internal/x: TestY skipped"},
	} {
		world := &improvingWorld{published: scenario.published, baseline: scenario.baseline, measureErr: scenario.measureErr}
		outcome, err := Run(world, Config{MaxAttemptsPerStep: 1, MaxInvocations: 1})
		if (err != nil) != (scenario.measureErr != nil) || outcome.Reason != ReasonPlanComplete {
			t.Fatalf("%s: outcome=%+v err=%v", scenario.name, outcome, err)
		}
		promotes := scenario.withheld == ""
		if promoted := slices.Equal(world.promoted, []string{"b"}); promoted != promotes || (outcome.Promoted == "b") != promoted {
			t.Fatalf("%s: promoted %v (outcome %q), want promotion %v", scenario.name, world.promoted, outcome.Promoted, promotes)
		}
		if decided := fmt.Sprintf("b after promoted %v, proposals 1", world.promoted); !slices.Equal(world.filled, []string{decided}) {
			t.Fatalf("%s: filled %q, want one fill at the measured commit after the decision and the proposals (%s)", scenario.name, world.filled, decided)
		}
		if promotes != (outcome.Withheld == "") || !strings.Contains(outcome.Withheld, scenario.withheld) {
			t.Fatalf("%s: withheld %q, want %q", scenario.name, outcome.Withheld, scenario.withheld)
		}
		for prefix, reason := range map[string]string{"scorecard-regressed-": fell, "scorecard-unknown-": unknown} {
			proposed := slices.ContainsFunc(world.proposed, func(gap scorecard.Gap) bool { return strings.HasPrefix(gap.Key, prefix) })
			if proposed != (scenario.withheld == reason) {
				t.Fatalf("%s: proposed %+v; a refusal for %q proposes its %s gap, nothing else does", scenario.name, world.proposed, reason, prefix)
			}
		}
	}
	world := &improvingWorld{measureErr: errors.New("validate -scorecard -publish failed")}
	if outcome, err := Run(world, Config{MaxAttemptsPerStep: 1, MaxInvocations: 1}); err == nil || len(world.promoted) != 0 || world.proposed != nil || world.filled != nil {
		t.Fatalf("a run that published no scorecard: outcome=%+v err=%v promoted=%v proposed=%v filled=%v", outcome, err, world.promoted, world.proposed, world.filled)
	}
}
