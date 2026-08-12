package runrecord

import (
	"context"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/repodb"
	"overgo/internal/testutil"
)

func TestGateRecordAggregatesStepsAndRoundTrips(t *testing.T) {
	record, err := NewGateRecord(
		testutil.ArtifactID(t, artifact.KindRecipe, "gate-recipe"),
		testutil.ArtifactID(t, artifact.KindEvidence, "gate-environment"), fixtureCodeCommit,
		OutcomeSucceeded, "", 100, []GateStep{
			{Name: "unit", Phase: PhaseTest, Outcome: StepSucceeded, DurationNS: 30},
			{Name: "integration", Phase: PhaseTest, Outcome: StepSucceeded, DurationNS: 40},
			{Name: "vet", Phase: PhaseVet, Outcome: StepSucceeded, DurationNS: 20},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if record.Run.MeasuredNS != 100 || !slices.Equal(record.Run.Phases, []PhaseMetric{
		{Phase: PhaseTest, DurationNS: 70},
		{Phase: PhaseVet, DurationNS: 20},
	}) {
		t.Fatalf("gate timing = (%d, %+v)", record.Run.MeasuredNS, record.Run.Phases)
	}
	content, err := record.Result.Content()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ParseGateResult(content.Data)
	if err != nil || parsed.ID != record.Result.ID {
		t.Fatalf("gate result round trip = (%+v, %v)", parsed, err)
	}
}

func TestCancelledGatePersistsTerminalTruth(t *testing.T) {
	ctx := context.Background()
	repository := t.TempDir()
	store, err := repodb.Open(repository)
	if err != nil {
		t.Fatal(err)
	}
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "gate-recipe")
	environmentID := testutil.ArtifactID(t, artifact.KindEvidence, "gate-environment")
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "fixture/gate-dependencies",
		Artifacts: []artifact.Descriptor{{ID: recipeID}, {ID: environmentID}},
	}); err != nil {
		t.Fatal(err)
	}
	record, err := NewGateRecord(
		recipeID, environmentID, fixtureCodeCommit, OutcomeCancelled, "", 55,
		[]GateStep{
			{Name: "unit", Phase: PhaseTest, Outcome: StepSucceeded, DurationNS: 40},
			{Name: "integration", Phase: PhaseTest, Outcome: StepCancelled, DurationNS: 10},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := record.Batch("fixture/cancelled-gate")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	store, err = repodb.Open(repository)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	content, ok, err := store.Content(ctx, record.Run.ID)
	if err != nil || !ok {
		t.Fatalf("terminal run content = (%t, %v)", ok, err)
	}
	parsed, err := ParseRun(content.Data)
	if err != nil || parsed.Outcome != OutcomeCancelled {
		t.Fatalf("terminal run = (%+v, %v)", parsed, err)
	}
}

func TestGateRecordRejectsOutcomeStepContradictions(t *testing.T) {
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "gate-recipe")
	environmentID := testutil.ArtifactID(t, artifact.KindEvidence, "gate-environment")
	if _, err := NewGateRecord(
		recipeID, environmentID, fixtureCodeCommit, OutcomeSucceeded, "", 10,
		[]GateStep{{Name: "unit", Phase: PhaseTest, Outcome: StepFailed, DurationNS: 10}},
	); err == nil {
		t.Fatal("successful gate with failed step accepted")
	}
	if _, err := NewGateRecord(
		recipeID, environmentID, fixtureCodeCommit, OutcomeFailed, "tests", 10,
		[]GateStep{{Name: "unit", Phase: PhaseTest, Outcome: StepSucceeded, DurationNS: 10}},
	); err == nil {
		t.Fatal("failed gate without failed step accepted")
	}
}

func TestVerifyGateRunRejectsUnboundIdentities(t *testing.T) {
	ctx := context.Background()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "verified-recipe")
	environmentID := testutil.ArtifactID(t, artifact.KindEvidence, "verified-environment")
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "fixture/verification/facts",
		Artifacts: []artifact.Descriptor{{ID: recipeID}, {ID: environmentID}},
	}); err != nil {
		t.Fatal(err)
	}
	records := make([]GateRecord, 2)
	for index, name := range []string{"first", "second"} {
		records[index], err = NewGateRecord(
			recipeID, environmentID, fixtureCodeCommit, OutcomeSucceeded, "", uint64(index+1),
			[]GateStep{{
				Name: name, Phase: PhaseValidate, Outcome: StepSucceeded, DurationNS: uint64(index + 1),
			}},
		)
		if err != nil {
			t.Fatal(err)
		}
		batch, batchErr := records[index].Batch("fixture/verification/" + name)
		if batchErr != nil {
			t.Fatal(batchErr)
		}
		if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := VerifyGateRun(
		ctx, store, recipeID, records[0].Result.ID, records[0].Run.ID,
	); err != nil {
		t.Fatal(err)
	}
	if _, err := VerifyGateRun(
		ctx, store, recipeID, records[0].Result.ID, records[1].Run.ID,
	); err == nil {
		t.Fatal("unbound gate/run pair accepted")
	}
}
