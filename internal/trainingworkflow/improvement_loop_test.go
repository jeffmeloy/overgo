package trainingworkflow

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/composition"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// driverTargets labels count targets; the driver hands each target to the
// enumerate stage and reads nothing else of it.
func driverTargets(count int) []CompositionTarget {
	targets := make([]CompositionTarget, 0, count)
	for index := range count {
		targets = append(targets, CompositionTarget{Label: "driver-target-" + string(rune('a'+index))})
	}
	return targets
}

// driverSpy is what the fake stages saw: how many composites they scored
// and, per stage call after enumeration, the label of the target handed in.
type driverSpy struct {
	scored   int
	received []string
}

// driverStages fakes the stage owners over the store. Every target
// enumerates the same donor and component, the pair the driver must never
// map back to a target. Realize assembles a composite from a published
// recipe and leaves its publication to the driver; Promote commits the
// promotion evidence it returns.
func driverStages(
	t *testing.T, store *overgodb.Store, fit func(attempt int) bool, stopped *bool,
) (CompositionDriverStages, *driverSpy) {
	t.Helper()
	spy, realizations := &driverSpy{}, 0
	donor := testutil.ArtifactID(t, artifact.KindModel, "driver-donor")
	testutil.PublishArtifact(t, store, testutil.ArtifactID(t, artifact.KindModel, "driver-base"))
	stages := CompositionDriverStages{
		Enumerate: func(context.Context, CompositionTarget) ([]composition.CompositionCandidate, error) {
			return []composition.CompositionCandidate{{
				Donor: donor, Component: "blk.10.ffn_gate.weight",
				Seam: composition.SeamLayerBoundary, Adapter: composition.AdapterLinear, Residual: 0.3, Fitness: 0.8,
			}}, nil
		},
		Realize: func(_ context.Context, target CompositionTarget, candidate composition.CompositionCandidate) (CandidateRealization, error) {
			spy.received = append(spy.received, "realize "+target.Label)
			realizations++
			recipeID := testutil.ArtifactID(t, artifact.KindRecipe, fmt.Sprintf("driver-recipe-%d", realizations))
			testutil.PublishArtifact(t, store, recipeID)
			composite, err := composition.NewComposedModel(composition.ComposedModelDocument{
				Architecture: "llama",
				Recipe:       recipeID,
				Parents:      []artifact.ID{testutil.ArtifactID(t, artifact.KindModel, "driver-base"), candidate.Donor},
			})
			return CandidateRealization{Candidate: candidate, Composite: composite}, err
		},
		Score: func(_ context.Context, target CompositionTarget, realization CandidateRealization) (composition.CompositeScore, error) {
			spy.received = append(spy.received, "score "+target.Label)
			spy.scored++
			quality := 0.6
			if !fit(spy.scored) {
				quality = 0.4
			}
			return composition.CompositeScore{
				Composite:  realization.Composite.ID,
				Evaluation: testutil.ArtifactID(t, artifact.KindEvidence, "driver-evaluation"),
				Dimensions: map[string]composition.CompositeDimension{
					"quality": {Baseline: 0.5, Candidate: quality, Direction: runrecord.DirectionMaximize},
				},
			}, nil
		},
		Promote: func(ctx context.Context, target CompositionTarget, _ composition.CompositeFitnessVerdict, realization CandidateRealization) (artifact.ID, error) {
			spy.received = append(spy.received, "promote "+target.Label)
			promotion := testutil.ArtifactID(t, artifact.KindEvidence, "driver-promotion-"+realization.Composite.ID.String())
			_, err := artifact.Publish(ctx, store, artifact.Batch{
				Key: "composition/driver-fixture/promotion/" + promotion.String(), Artifacts: []artifact.Descriptor{{ID: promotion}},
			})
			return promotion, err
		},
		OperatorStop: func() bool { return *stopped },
	}
	return stages, spy
}

func driverAttemptRecords(t *testing.T, store *overgodb.Store) int {
	t.Helper()
	count := 0
	if _, err := overgodb.VisitDecodedDocuments(
		t.Context(), store, overgodb.DocumentQuery{
			Contracts: []artifact.DocumentContract{{
				Kind: artifact.KindEvidence, MediaType: DriverAttemptMediaType, Schema: DriverAttemptSchema,
			}}, Order: overgodb.DocumentOldestFirst,
		},
		func(data []byte) (CompositionDriverAttempt, error) {
			var attempt CompositionDriverAttempt
			return attempt, json.Unmarshal(data, &attempt)
		},
		func(_ overgodb.DocumentView, _ CompositionDriverAttempt) error {
			count++
			return nil
		},
	); err != nil {
		t.Fatal(err)
	}
	return count
}

// TestCompositionImprovementLoopClosesUnderBudget pins the driver
// closure: the run sequences enumerate, realize, select, and promote for
// each target, hands every stage the target the attempt selected though
// all targets enumerate the same donor and component, publishes each
// realization before its attempt cites it and every attempt before moving
// on, and stops exactly at the attempt budget derived from measured
// history — the ceiling of attempts per historical promotion — naming the
// stop. A promotion the store never received is refused, not cited.
func TestCompositionImprovementLoopClosesUnderBudget(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	testutil.PublishArtifact(t, store, testutil.ArtifactID(t, artifact.KindModel, "driver-donor"))
	stopped := false
	stages, spy := driverStages(t, store, func(int) bool { return true }, &stopped)
	// Measured history: six attempts, two promotions — one success has
	// historically cost three attempts, so this run may spend three.
	history := []bool{false, true, false, false, true, false}
	targets := driverTargets(5)
	slices.Reverse(targets)
	report, err := RunCompositionImprovementLoop(ctx, store, targets, stages, history)
	if err != nil {
		t.Fatal(err)
	}
	if report.Budget != 3 || report.Stop != DriverStopBudget || len(report.Attempts) != 3 || spy.scored != 3 {
		t.Fatalf("driver report = %+v scored=%d", report, spy.scored)
	}
	var want []string
	for index, attempt := range report.Attempts {
		label := targets[index].Label
		want = append(want, "realize "+label, "score "+label, "promote "+label)
		if attempt.Target != label || !attempt.Promoted || !attempt.Fit || attempt.Refusal != "" {
			t.Fatalf("attempt %d = %+v, want target %s promoted", index, attempt, label)
		}
		for _, cited := range []artifact.ID{attempt.Composite, attempt.Promotion} {
			if _, found, err := store.Artifact(ctx, cited); err != nil || !found {
				t.Fatalf("attempt %d cites %s, which the store does not hold: %v", index, cited, err)
			}
		}
	}
	if !slices.Equal(spy.received, want) {
		t.Fatalf("stages received targets %q, want %q", spy.received, want)
	}
	if records := driverAttemptRecords(t, store); records != len(report.Attempts) {
		t.Fatalf("published attempt records = %d, want %d", records, len(report.Attempts))
	}

	withheld := stages
	withheld.Promote = func(context.Context, CompositionTarget, composition.CompositeFitnessVerdict, CandidateRealization) (artifact.ID, error) {
		return testutil.ArtifactID(t, artifact.KindEvidence, "driver-withheld-promotion"), nil
	}
	refused, err := RunCompositionImprovementLoop(ctx, store, driverTargets(1), withheld, history)
	if err != nil {
		t.Fatal(err)
	}
	if attempt := refused.Attempts[0]; attempt.Promoted || attempt.Promotion.Valid() ||
		!strings.HasPrefix(attempt.Refusal, "promote: ") || !strings.Contains(attempt.Refusal, "not committed") {
		t.Fatalf("uncommitted promotion attempt = %+v", attempt)
	}
	if records := driverAttemptRecords(t, store); records != len(report.Attempts)+1 {
		t.Fatalf("published attempt records = %d, want the refusal recorded too", records)
	}

	if _, err := RunCompositionImprovementLoop(ctx, store, nil, stages, history); err == nil {
		t.Fatal("targetless driver ran")
	}
	missing := stages
	missing.Promote = nil
	if _, err := RunCompositionImprovementLoop(ctx, store, driverTargets(1), missing, history); err == nil ||
		!strings.Contains(err.Error(), "every stage owner") {
		t.Fatalf("stageless driver ran: %v", err)
	}
}

// TestCompositionLoopSaturationStops pins the two remaining stops: with a
// promotionless measured history the derived bounds are maximally
// conservative and one consecutive fitness failure saturates the run,
// while operator authority stops the loop before any further attempt and
// is never overridden.
func TestCompositionLoopSaturationStops(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	testutil.PublishArtifact(t, store, testutil.ArtifactID(t, artifact.KindModel, "driver-donor"))
	stopped := false
	stages, _ := driverStages(t, store, func(int) bool { return false }, &stopped)
	// History with a recovered two-failure streak: saturation = 3, budget =
	// ceil(5/1) = 5, so saturation is the binding stop for an unfit run.
	history := []bool{false, false, true, false, false}
	report, err := RunCompositionImprovementLoop(ctx, store, driverTargets(6), stages, history)
	if err != nil {
		t.Fatal(err)
	}
	if report.Saturation != 3 || report.Stop != DriverStopSaturation || len(report.Attempts) != 3 {
		t.Fatalf("saturated report = %+v", report)
	}
	for _, attempt := range report.Attempts {
		if attempt.Promoted || attempt.Fit || !strings.Contains(attempt.Refusal, "regressed") {
			t.Fatalf("unfit attempt = %+v", attempt)
		}
	}

	budget, saturation := DeriveDriverBounds(nil)
	if budget != 1 || saturation != 1 {
		t.Fatalf("empty history bounds = (%d, %d); nothing measured supports more", budget, saturation)
	}

	stopped = true
	halted, err := RunCompositionImprovementLoop(ctx, store, driverTargets(2), stages, history)
	if err != nil || halted.Stop != DriverStopOperator || len(halted.Attempts) != 0 {
		t.Fatalf("operator stop = (%+v, %v)", halted, err)
	}
}
