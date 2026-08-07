package runrecord

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/repodb"
)

func TestGateRecordAggregatesStepsAndRoundTrips(t *testing.T) {
	record, err := NewGateRecord(
		fixtureID(t, artifact.KindRecipe, "gate-recipe"),
		fixtureID(t, artifact.KindEvidence, "gate-environment"), fixtureCodeCommit,
		OutcomeSucceeded, "", 100, []GateStep{
			{Name: "unit", Phase: PhaseTest, Outcome: StepSucceeded, DurationNS: 30},
			{Name: "integration", Phase: PhaseTest, Outcome: StepSucceeded, DurationNS: 40},
			{Name: "vet", Phase: PhaseVet, Outcome: StepSucceeded, DurationNS: 20},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if record.Run.UnattributedNS() != 10 || len(record.Run.Phases) != 2 {
		t.Fatalf("gate timing = (%d, %+v)", record.Run.UnattributedNS(), record.Run.Phases)
	}
	content, err := record.Result.ContentBytes()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ParseGateResult(content)
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
	recipeID := fixtureID(t, artifact.KindRecipe, "gate-recipe")
	environmentID := fixtureID(t, artifact.KindEvidence, "gate-environment")
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
	recipeID := fixtureID(t, artifact.KindRecipe, "gate-recipe")
	environmentID := fixtureID(t, artifact.KindEvidence, "gate-environment")
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
