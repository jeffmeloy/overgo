package composition

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func driverTargets(t *testing.T, count int) []CompositionTarget {
	t.Helper()
	targets := make([]CompositionTarget, 0, count)
	for index := 0; index < count; index++ {
		name := "driver-target-" + string(rune('a'+index))
		targets = append(targets, CompositionTarget{
			Label: name,
			Component: descriptorComponent(t, name, "blk.10.ffn_gate.weight",
				descriptorStatistics(0.001, 0.02, 0.01, 0.1, 0.2)),
		})
	}
	return targets
}

func driverStages(
	t *testing.T, store *overgodb.Store, fit func(attempt int) bool, stopped *bool,
) (CompositionDriverStages, *int) {
	t.Helper()
	attempts, realizations := 0, 0
	donor := testutil.ArtifactID(t, artifact.KindModel, "driver-donor")
	promotion := testutil.ArtifactID(t, artifact.KindEvidence, "driver-promotion")
	testutil.PublishArtifact(t, store, testutil.ArtifactID(t, artifact.KindModel, "driver-base"))
	stages := CompositionDriverStages{
		Enumerate: func(context.Context, CompositionTarget) ([]CompositionCandidate, error) {
			return []CompositionCandidate{{
				Donor: donor, Component: "blk.10.ffn_gate.weight",
				Seam: SeamLayerBoundary, Adapter: AdapterLinear, Residual: 0.3, Fitness: 0.8,
			}}, nil
		},
		Realize: func(ctx context.Context, candidate CompositionCandidate) (CandidateRealization, error) {
			realizations++
			recipeID := testutil.ArtifactID(t, artifact.KindRecipe, fmt.Sprintf("driver-recipe-%d", realizations))
			testutil.PublishArtifact(t, store, recipeID)
			composite, err := NewComposedModel(ComposedModelDocument{
				Architecture: "llama",
				Recipe:       recipeID,
				Parents:      []artifact.ID{testutil.ArtifactID(t, artifact.KindModel, "driver-base"), candidate.Donor},
			})
			if err != nil {
				return CandidateRealization{}, err
			}
			batch, err := composite.Batch("composition/driver-fixture/composite/" + composite.ID.String())
			if err != nil {
				return CandidateRealization{}, err
			}
			if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
				return CandidateRealization{}, err
			}
			return CandidateRealization{Candidate: candidate, Composite: composite}, nil
		},
		Score: func(_ context.Context, realization CandidateRealization) (CompositeScore, error) {
			attempts++
			quality := 0.6
			if !fit(attempts) {
				quality = 0.4
			}
			return CompositeScore{
				Composite:  realization.Composite.ID,
				Evaluation: testutil.ArtifactID(t, artifact.KindEvidence, "driver-evaluation"),
				Dimensions: map[string]CompositeDimension{
					"quality": {Baseline: 0.5, Candidate: quality, Direction: runrecord.DirectionMaximize},
				},
			}, nil
		},
		Promote: func(context.Context, CompositeFitnessVerdict, CandidateRealization) (artifact.ID, error) {
			return promotion, nil
		},
		OperatorStop: func() bool { return *stopped },
	}
	return stages, &attempts
}

func driverAttemptRecords(t *testing.T, store *overgodb.Store) int {
	t.Helper()
	count := 0
	if _, err := overgodb.VisitDecodedDocuments(
		context.Background(), store, overgodb.DocumentQuery{
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
// each target, publishes every attempt to the store before moving on,
// and stops exactly at the attempt budget derived from measured history —
// the ceiling of attempts per historical promotion — naming the stop.
func TestCompositionImprovementLoopClosesUnderBudget(t *testing.T) {
	ctx := context.Background()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	testutil.PublishArtifact(t, store, testutil.ArtifactID(t, artifact.KindModel, "driver-donor"))
	testutil.PublishArtifact(t, store, testutil.ArtifactID(t, artifact.KindEvidence, "driver-promotion"))
	stopped := false
	stages, attempts := driverStages(t, store, func(int) bool { return true }, &stopped)
	// Measured history: six attempts, two promotions — one success has
	// historically cost three attempts, so this run may spend three.
	history := []bool{false, true, false, false, true, false}
	report, err := RunCompositionImprovementLoop(ctx, store, driverTargets(t, 5), stages, history)
	if err != nil {
		t.Fatal(err)
	}
	if report.Budget != 3 || report.Stop != DriverStopBudget || len(report.Attempts) != 3 || *attempts != 3 {
		t.Fatalf("driver report = %+v attempts=%d", report, *attempts)
	}
	for _, attempt := range report.Attempts {
		if !attempt.Promoted || !attempt.Fit || attempt.Refusal != "" {
			t.Fatalf("promoted attempt = %+v", attempt)
		}
	}
	if records := driverAttemptRecords(t, store); records != len(report.Attempts) {
		t.Fatalf("published attempt records = %d, want %d", records, len(report.Attempts))
	}

	if _, err := RunCompositionImprovementLoop(ctx, store, nil, stages, history); err == nil {
		t.Fatal("targetless driver ran")
	}
	missing := stages
	missing.Promote = nil
	if _, err := RunCompositionImprovementLoop(ctx, store, driverTargets(t, 1), missing, history); err == nil ||
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
	ctx := context.Background()
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
	report, err := RunCompositionImprovementLoop(ctx, store, driverTargets(t, 6), stages, history)
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
	halted, err := RunCompositionImprovementLoop(ctx, store, driverTargets(t, 2), stages, history)
	if err != nil || halted.Stop != DriverStopOperator || len(halted.Attempts) != 0 {
		t.Fatalf("operator stop = (%+v, %v)", halted, err)
	}
}
