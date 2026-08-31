package evaluation

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestDeterministicRSIControlPlane proves the evaluation slice of the
// control plane: evidence coverage answers under explicit fact, chunk, and
// raw-byte budgets with missing coverage stated rather than inferred, an
// exhausted budget refuses instead of degrading silently, and the live
// safety window derives only from sufficient admitted history.
func TestDeterministicRSIControlPlane(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "plane recipe")
	causalRoot := testutil.ArtifactID(t, artifact.KindEvidence, "plane causal root")
	coveredResult := testutil.ArtifactID(t, artifact.KindEvidence, "plane covered result")
	missingResult := testutil.ArtifactID(t, artifact.KindEvidence, "plane missing result")
	for _, id := range []artifact.ID{recipeID, causalRoot, coveredResult, missingResult} {
		testutil.PublishArtifact(t, store, id)
	}
	causal, err := runrecord.NewCausalRoot(runrecord.TriggerManual, causalRoot)
	if err != nil {
		t.Fatal(err)
	}
	commit := "0123456789abcdef0123456789abcdef01234567"
	covered, err := runrecord.NewAttemptRecord(runrecord.AttemptRecord{
		PlanItem: "plane", PlanStep: "covered", Result: coveredResult, Recipe: recipeID,
		CodeCommit: commit, Outcome: runrecord.OutcomeSucceeded, WallNS: 1, Causal: &causal,
	})
	if err != nil {
		t.Fatal(err)
	}
	uncovered, err := runrecord.NewAttemptRecord(runrecord.AttemptRecord{
		PlanItem: "plane", PlanStep: "uncovered", Result: missingResult, Recipe: recipeID,
		CodeCommit: commit, Outcome: runrecord.OutcomeSucceeded, WallNS: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	for name, record := range map[string]runrecord.AttemptRecord{
		"covered": covered, "uncovered": uncovered,
	} {
		content, contentErr := record.Content()
		if contentErr != nil {
			t.Fatal(contentErr)
		}
		batch := artifact.Batch{
			Key:       "plane/attempt/" + name,
			Artifacts: []artifact.Descriptor{content.Descriptor}, Contents: []artifact.Content{content},
			Lineage: record.Lineage(),
		}
		if err := runrecord.BindCausality(&batch, record.ID, record.Causal); err != nil {
			t.Fatal(err)
		}
		if _, err := store.Commit(ctx, batch); err != nil {
			t.Fatal(err)
		}
	}

	units := []CoverageUnit{
		{Attempt: covered.ID, Required: []CoverageRequirement{{Axis: CoverageCausal}}},
		{Attempt: uncovered.ID, Required: []CoverageRequirement{{Axis: CoverageCausal}}},
	}
	bounds := CoverageBounds{
		MaxFacts: 64,
		Observation: runrecord.ObservationStreamBounds{
			MaxChunks: 8, MaxRawBytes: 1 << 20,
		},
	}
	query, err := NewEvidenceCoverageQuery(units, nil, bounds)
	if err != nil {
		t.Fatal(err)
	}
	projection, err := query.Project(ctx, store)
	if err != nil {
		t.Fatal(err)
	}
	if projection.Units != 2 || projection.Causal.Observed != 1 || projection.Causal.Missing != 1 {
		t.Fatalf("coverage misread the denominator: %+v", projection)
	}

	// An exhausted fact budget refuses; nothing degrades silently.
	starvedUnits := []CoverageUnit{{
		Attempt: covered.ID,
		EvaluationEvidence: []artifact.ID{
			coveredResult, missingResult,
		},
		Required: []CoverageRequirement{{Axis: CoverageCausal}},
	}}
	starved := bounds
	starved.MaxFacts = 1
	if _, err := NewEvidenceCoverageQuery(starvedUnits, nil, starved); err == nil ||
		!strings.Contains(err.Error(), "fact budget") {
		t.Fatalf("starved budget admitted its denominator: %v", err)
	}

	// Live safety derives its window only from sufficient history.
	history := []float64{0.52, 0.58, 0.55, 0.50, 0.60, 0.54, 0.53, 0.57, 0.51, 0.56}
	window, err := DeriveLiveSafetyWindow(
		"exact-match", runrecord.DirectionMaximize, history, causalRoot,
	)
	if err != nil {
		t.Fatal(err)
	}
	if window.RequiredObservations == 0 {
		t.Fatalf("derived window requires nothing: %+v", window)
	}
	if _, err := DeriveLiveSafetyWindow(
		"exact-match", runrecord.DirectionMaximize, history[:1], causalRoot,
	); err == nil {
		t.Fatal("insufficient history derived a live safety window")
	}
}
