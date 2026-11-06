package runrecord

import (
	"context"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/overgodbtest"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

func TestGateRecordAggregatesStepsAndRoundTrips(t *testing.T) {
	record, err := NewGateRecord(
		testutil.ArtifactID(t, artifact.KindRecipe, "gate-recipe"),
		testutil.ArtifactID(t, artifact.KindEvidence, "gate-environment"), fixtureCodeCommit,
		OutcomeSucceeded, "", 100, []GateStep{
			{Name: "unit", Phase: PhaseTest, Outcome: StepSucceeded, DurationNS: 30},
			{Name: "integration", Phase: PhaseTest, Outcome: StepSucceeded, DurationNS: 40},
			{Name: "vet", Phase: phaseVet, Outcome: StepSucceeded, DurationNS: 20},
		},
	)
	testcheck.NoError(t, err)
	if record.Run.MeasuredNS != 100 || !slices.Equal(record.Run.Phases, []PhaseMetric{
		{Phase: PhaseTest, DurationNS: 70},
		{Phase: phaseVet, DurationNS: 20},
	}) {
		t.Fatalf("gate timing = (%d, %+v)", record.Run.MeasuredNS, record.Run.Phases)
	}
	content, err := record.Result.Content()
	testcheck.NoError(t, err)
	parsed, err := ParseGateResult(content.Data)
	if err != nil || parsed.ID != record.Result.ID {
		t.Fatalf("gate result round trip = (%+v, %v)", parsed, err)
	}
}

func TestProtectionEvidence(t *testing.T) {
	record, err := NewGateRecord(
		testutil.ArtifactID(t, artifact.KindRecipe, "gate-recipe"),
		testutil.ArtifactID(t, artifact.KindEvidence, "gate-environment"), fixtureCodeCommit,
		OutcomeSucceeded, "", 100, []GateStep{{
			Name: "protection", Phase: PhaseValidate, Outcome: StepSucceeded, DurationNS: 1,
			Evidence: "configured:branches=master,jobs=unit+race,hooks=4;host_enforcement=external;activation=unobserved:parent-harness-fact",
		}},
	)
	testcheck.NoError(t, err)
	content, err := record.Result.Content()
	testcheck.NoError(t, err)
	parsed, err := ParseGateResult(content.Data)
	if err != nil || parsed.Steps[0].Evidence != record.Result.Steps[0].Evidence {
		t.Fatalf("gate step evidence round trip = (%+v, %v)", parsed.Steps, err)
	}
}

func TestCancelledGatePersistsTerminalTruth(t *testing.T) {
	ctx := t.Context()
	repository := t.TempDir()
	store, err := overgodb.Create(repository)
	testcheck.NoError(t, err)
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
			{Name: "integration", Phase: PhaseTest, Outcome: stepCancelled, DurationNS: 10},
		},
	)
	testcheck.NoError(t, err)
	batch, err := record.Batch("fixture/cancelled-gate")
	testcheck.NoError(t, err)
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	required, err := RequireGateResult(ctx, store, record.Result.ID)
	if err != nil || required.ID != record.Result.ID || required.Outcome != OutcomeCancelled {
		t.Fatalf("required gate result = (%+v, %v)", required, err)
	}
	testcheck.NoError(t, store.Close())
	store = overgodbtest.Create(t, repository)
	content, ok, err := artifact.ReadContent(ctx, store, record.Run.ID)
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
	ctx := t.Context()
	store := overgodbtest.Open(t, t.TempDir())
	var err error
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "verified-recipe")
	environmentID := publishGateTestEnvironment(t, ctx, store, "verified-environment")
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "fixture/verification/facts",
		Artifacts: []artifact.Descriptor{{ID: recipeID}},
	}); err != nil {
		t.Fatal(err)
	}
	records := make([]gateRecord, 2)
	for index, name := range []string{"first", "second"} {
		records[index], err = NewGateRecord(
			recipeID, environmentID, fixtureCodeCommit, OutcomeSucceeded, "", uint64(index+1),
			[]GateStep{{
				Name: name, Phase: PhaseValidate, Outcome: StepSucceeded, DurationNS: uint64(index + 1),
			}},
		)
		testcheck.NoError(t, err)
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

func TestVerifyFailedGateRun(t *testing.T) {
	ctx := t.Context()
	store := overgodbtest.Open(t, t.TempDir())
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "failed-recipe")
	environmentID := publishGateTestEnvironment(t, ctx, store, "failed-environment")
	testutil.PublishArtifact(t, store, recipeID)
	record, err := NewGateRecord(
		recipeID, environmentID, fixtureCodeCommit, OutcomeFailed, "exact-mismatch", 1,
		[]GateStep{{Name: "verify", Phase: PhaseTest, Outcome: StepFailed, DurationNS: 1}},
	)
	testcheck.NoError(t, err)
	batch, err := record.Batch("fixture/failed-verification")
	testcheck.NoError(t, err)
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		t.Fatal(err)
	}
	if _, err := VerifyFailedGateRun(ctx, store, recipeID, record.Result.ID, record.Run.ID); err != nil {
		t.Fatal(err)
	}
	if _, err := VerifyGateRun(ctx, store, recipeID, record.Result.ID, record.Run.ID); err == nil {
		t.Fatal("failed evidence accepted as successful")
	}
}

func TestReferenceAdmissionRequiresSemanticRelevance(t *testing.T) {
	ctx := t.Context()
	store := overgodbtest.Open(t, t.TempDir())

	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "relevant-environment-recipe")
	foreignRecipeID := testutil.ArtifactID(t, artifact.KindRecipe, "foreign-environment-recipe")
	testutil.PublishArtifact(t, store, recipeID)
	testutil.PublishArtifact(t, store, foreignRecipeID)
	exactEnvironment := publishGateTestEnvironment(t, ctx, store, "relevant-environment")
	foreignEnvironment := publishGateTestEnvironment(t, ctx, store, "foreign-environment")
	exact, err := NewGateRecord(
		recipeID, exactEnvironment, fixtureCodeCommit, OutcomeSucceeded, "", 1,
		[]GateStep{{Name: "verify", Phase: PhaseValidate, Outcome: StepSucceeded, DurationNS: 1}},
	)
	testcheck.NoError(t, err)
	exactBatch, err := exact.Batch("fixture/reference-relevance/exact")
	testcheck.NoError(t, err)
	if _, err := artifact.CommitBatch(ctx, store, exactBatch); err != nil {
		t.Fatal(err)
	}
	if _, err := VerifyGateRun(ctx, store, recipeID, exact.Result.ID, exact.Run.ID); err != nil {
		t.Fatalf("exact typed environment was rejected: %v", err)
	}
	foreignSubject, err := NewGateRecord(
		foreignRecipeID, exactEnvironment, fixtureCodeCommit, OutcomeSucceeded, "", 1,
		[]GateStep{{Name: "verify", Phase: PhaseValidate, Outcome: StepSucceeded, DurationNS: 1}},
	)
	testcheck.NoError(t, err)
	foreignSubjectBatch, err := foreignSubject.Batch("fixture/reference-relevance/foreign-subject")
	testcheck.NoError(t, err)
	if _, err := artifact.CommitBatch(ctx, store, foreignSubjectBatch); err != nil {
		t.Fatal(err)
	}
	if _, err := VerifyGateRun(ctx, store, recipeID, foreignSubject.Result.ID, foreignSubject.Run.ID); err == nil {
		t.Fatal("well-formed gate and run for a foreign subject were admitted")
	}

	descriptorOnly := testutil.ArtifactID(t, artifact.KindEvidence, "descriptor-only-environment")
	testutil.PublishArtifact(t, store, descriptorOnly)
	untyped, err := NewGateRecord(
		recipeID, descriptorOnly, fixtureCodeCommit, OutcomeSucceeded, "", 1,
		[]GateStep{{Name: "verify", Phase: PhaseValidate, Outcome: StepSucceeded, DurationNS: 1}},
	)
	testcheck.NoError(t, err)
	untypedBatch, err := untyped.Batch("fixture/reference-relevance/descriptor-only")
	testcheck.NoError(t, err)
	if _, err := artifact.CommitBatch(ctx, store, untypedBatch); err != nil {
		t.Fatal(err)
	}
	if _, err := VerifyGateRun(ctx, store, recipeID, untyped.Result.ID, untyped.Run.ID); err == nil {
		t.Fatal("descriptor-only environment was admitted")
	}

	foreignRun, err := NewBoundRun(
		recipeID, OutcomeSucceeded, nil, []artifact.ID{exact.Result.ID}, "", fixtureCodeCommit,
		foreignEnvironment, exact.Run.MeasuredNS, exact.Run.Phases,
	)
	testcheck.NoError(t, err)
	foreignRunBatch, err := foreignRun.Batch("fixture/reference-relevance/foreign-environment")
	testcheck.NoError(t, err)
	if _, err := artifact.CommitBatch(ctx, store, foreignRunBatch); err != nil {
		t.Fatal(err)
	}
	if _, err := VerifyGateRun(ctx, store, recipeID, exact.Result.ID, foreignRun.ID); err == nil {
		t.Fatal("well-formed run bound to a foreign environment was admitted")
	}
}

func publishGateTestEnvironment(
	t testing.TB,
	ctx context.Context,
	store artifact.Repository,
	name string,
) artifact.ID {
	t.Helper()
	environment, err := NewEnvironment(Environment{
		Host: name, OS: "test", Arch: "test", Device: "host", Backend: "go", Driver: "test",
	})
	testcheck.NoError(t, err)
	batch, err := environment.Batch("fixture/environment/" + environment.ID.String())
	testcheck.NoError(t, err)
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		t.Fatal(err)
	}
	return environment.ID
}
