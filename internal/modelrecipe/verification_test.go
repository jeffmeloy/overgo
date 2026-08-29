package modelrecipe

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
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
	if err != nil {
		t.Fatal(err)
	}
	batch, err := record.Batch(key)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(context.Background(), store, batch); err != nil {
		t.Fatal(err)
	}
	return Verification{Gate: record.Result.ID, Run: record.Run.ID}
}

func publishFailedVerification(
	t testing.TB,
	store artifact.Repository,
	definitionID artifact.ID,
	key string,
) Verification {
	t.Helper()
	environment := publishVerificationEnvironment(t, store, key+"/environment")
	record, err := runrecord.NewGateRecord(
		definitionID, environment, lifecycleDecisionCommit,
		runrecord.OutcomeFailed, "exact-mismatch", 1,
		[]runrecord.GateStep{{
			Name: "verify", Phase: runrecord.PhaseValidate,
			Outcome: runrecord.StepFailed, DurationNS: 1,
		}},
	)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := record.Batch(key)
	if err != nil {
		t.Fatal(err)
	}
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
	if err != nil {
		t.Fatal(err)
	}
	batch, err := environment.Batch(key)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(context.Background(), store, batch); err != nil {
		t.Fatal(err)
	}
	return environment.ID
}
