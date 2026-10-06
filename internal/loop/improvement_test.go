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
// scripted record against the scripted reference.
type improvingWorld struct {
	fakeWorld
	published, reference *scorecard.Record
	measureErr           error
	proposed             []scorecard.Gap
	proposals            int
	// filled holds the commit each fill ran at and how many proposals had been
	// made when it ran.
	filled []string
}

func (w *improvingWorld) Fill(commit string) {
	w.filled = append(w.filled, fmt.Sprintf("%s after proposals %d", commit, w.proposals))
}

func (w *improvingWorld) Measure() (*scorecard.Record, *scorecard.Record, error) {
	return w.published, w.reference, w.measureErr
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

// TestCycleFilesDeclinesAndAcceptsNothing empties the plan against scripted
// scorecards and holds the cycle to what it does: each cell is compared with its
// reference, and a decline beyond the column's noise, in held-out quality in the
// metric's own direction or in throughput, is proposed as a gap of its own, whatever
// else improved; a quality nothing compares (another suite, a row the record lost, no
// reference at all) is reported and proposes no decline, and no run proposes an
// unknown. A failed validation does not discard the cells that finished: they are
// compared, their declines proposed, and its failure returned for the repair it asks.
// Every run that published fills the validations the scorecard does not compare once, at
// the commit it measured, after its proposals: the fill records its claims apart from
// the checkout the proposals write. Nothing here promotes a commit.
func TestCycleFilesDeclinesAndAcceptsNothing(t *testing.T) {
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
		cells[metric.Throughput] = scorecard.Cell{Value: throughput, Unit: "tok/s", Source: source, Contract: scorecard.Contract{Metric: "decode rate", Origin: "serving"}}
		cells[metric.Failures] = scorecard.Cell{Value: failures, Label: "attempts", Source: source}
		return &scorecard.Record{Commit: commit, Rows: []scorecard.Row{{Model: model, Name: "model", Task: recipe.TaskInference, Cells: cells}}}
	}
	unmeasured := func(record *scorecard.Record, column metric.Name) *scorecard.Record {
		record.Rows[0].Cells[column] = scorecard.Cell{Reason: "not measured before"}
		return record
	}
	for _, scenario := range []struct {
		name                 string
		reference, published *scorecard.Record
		measureErr           error
		// declined says a decline is proposed; unmeasured, that a measurement to take is.
		declined, unmeasured bool
	}{
		{name: "held quality, a little slower, failing", reference: record("a", accuracy(0.5, "mmlu"), 40, 0), published: record("b", accuracy(0.5, "mmlu"), 38, 0.1)},
		{name: "accuracy fell while throughput rose", reference: record("a", accuracy(0.5, "mmlu"), 40, 0), published: record("b", accuracy(0.4, "mmlu"), 50, 0), declined: true},
		{name: "throughput fell beyond its noise", reference: record("a", accuracy(0.5, "mmlu"), 40, 0), published: record("b", accuracy(0.5, "mmlu"), 30, 0), declined: true},
		{name: "perplexity rose", reference: record("a", perplexity(900), 40, 0), published: record("b", perplexity(950), 40, 0), declined: true},
		{name: "perplexity fell", reference: record("a", perplexity(900), 40, 0), published: record("b", perplexity(850), 40, 0)},
		{name: "another suite scores lower", reference: record("a", accuracy(0.5, "mmlu"), 40, 0), published: record("b", accuracy(0.2, "mmlu-pro"), 40, 0)},
		{name: "a vanished reference row", reference: record("a", accuracy(0.5, "mmlu"), 40, 0), published: &scorecard.Record{Commit: "b"}},
		{name: "a new optional metric", reference: unmeasured(record("a", accuracy(0.5, "mmlu"), 40, 0), metric.Load), published: record("b", accuracy(0.5, "mmlu"), 40, 0)},
		{name: "no earlier scorecard", published: record("b", accuracy(0.9, "mmlu"), 40, 0)},
		{name: "held-out quality missing", reference: record("a", accuracy(0.5, "mmlu"), 40, 0), published: record("b", scorecard.Cell{Reason: "no held-out suite ran"}, 40, 0), unmeasured: true},
		{name: "failed validation, the finished cells compared", reference: record("a", accuracy(0.5, "mmlu"), 40, 0), published: record("b", accuracy(0.4, "mmlu"), 40, 0),
			measureErr: errors.New("validate -run: cycle lane models: overgo/internal/x: TestY skipped without declaring itself inapplicable"), declined: true},
	} {
		world := &improvingWorld{published: scenario.published, reference: scenario.reference, measureErr: scenario.measureErr}
		outcome, err := Run(world, Config{MaxAttemptsPerStep: 1, MaxInvocations: 1})
		if (err != nil) != (scenario.measureErr != nil) || outcome.Reason != ReasonPlanComplete {
			t.Fatalf("%s: outcome=%+v err=%v", scenario.name, outcome, err)
		}
		if filled := []string{"b after proposals 1"}; !slices.Equal(world.filled, filled) {
			t.Fatalf("%s: filled %q, want one fill at the measured commit after the proposals %q", scenario.name, world.filled, filled)
		}
		has := func(prefix string) bool {
			return slices.ContainsFunc(world.proposed, func(gap scorecard.Gap) bool { return strings.HasPrefix(gap.Key, prefix) })
		}
		if has("scorecard-regressed-") != scenario.declined || has("scorecard-unmeasured-") != scenario.unmeasured || has("scorecard-unknown-") {
			t.Fatalf("%s: proposed %+v; want a decline %t, a measurement to take %t and no unknown", scenario.name, world.proposed, scenario.declined, scenario.unmeasured)
		}
	}
	world := &improvingWorld{measureErr: errors.New("validate -scorecard -publish failed")}
	if outcome, err := Run(world, Config{MaxAttemptsPerStep: 1, MaxInvocations: 1}); err == nil || world.proposed != nil || world.filled != nil {
		t.Fatalf("a run that published no scorecard: outcome=%+v err=%v proposed=%v filled=%v", outcome, err, world.proposed, world.filled)
	}
}
