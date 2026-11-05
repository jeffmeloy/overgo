package modelrecipe

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
	"overgo/internal/testcheck"
)

func publishVerification(
	t testing.TB,
	store artifact.Repository,
	definitionID artifact.ID,
	key string,
) Verification {
	t.Helper()
	environment := publishVerificationEnvironment(t, store, key+"/environment")
	record, err := runrecord.NewGateRecord(
		definitionID, environment, lifecycleDecisionCommit,
		runrecord.OutcomeSucceeded, "", 1,
		[]runrecord.GateStep{{
			Name: "verify", Phase: runrecord.PhaseValidate,
			Outcome: runrecord.StepSucceeded, DurationNS: 1,
		}},
	)
	testcheck.NoError(t, err)
	batch, err := record.Batch(key)
	testcheck.NoError(t, err)
	if _, err := artifact.CommitBatch(context.Background(), store, batch); err != nil {
		t.Fatal(err)
	}
	return Verification{Gate: record.Result.ID, Run: record.Run.ID}
}

func publishVerificationEnvironment(t testing.TB, store artifact.Repository, key string) artifact.ID {
	t.Helper()
	environment, err := runrecord.NewEnvironment(runrecord.Environment{
		Host: key, OS: "test", Arch: "test", Device: "host", Backend: "go", Driver: "test",
	})
	testcheck.NoError(t, err)
	batch, err := environment.Batch(key)
	testcheck.NoError(t, err)
	if _, err := artifact.CommitBatch(context.Background(), store, batch); err != nil {
		t.Fatal(err)
	}
	return environment.ID
}
