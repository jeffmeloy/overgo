package gate

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
)

func TestReusedCheckpointCompletionAuthorityAcceptance(t *testing.T) {
	first, batch, tree := verificationBatchFixture(t, "pass")
	first.environment = lifecycleTestEnvironment(t)
	invocations := persistenceInvocations(t, first, batch, tree)
	cache := first.loadRetryCache()
	results, err := first.executeChecks(invocations, nil, map[artifact.ID]artifact.ID{}, &cache, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, result := range results {
		if result.Err != nil || result.Evidence.Reused {
			t.Fatalf("initial execution did not run successfully: %+v", result)
		}
	}

	// Reconstruct all gate state and read the persisted cache. No callback
	// state from the first attempt may provide the second attempt's binding.
	second, batch, tree := verificationBatchContext(t, first.repo)
	second.environment = first.environment
	invocations = persistenceInvocations(t, second, batch, tree)
	cache = second.loadRetryCache()
	results, err = second.executeChecks(invocations, nil, map[artifact.ID]artifact.ID{}, &cache, nil)
	if err != nil {
		t.Fatal(err)
	}
	var records []runrecord.GateStep
	reused := 0
	for _, result := range results {
		if result.Err != nil {
			t.Fatal(result.Err)
		}
		name := result.Invocation.Check.Name
		if name == "acceptance" && result.Evidence.Reused {
			t.Fatal("parent integration must execute")
		}
		if result.Evidence.Reused {
			reused++
		}
		records = append(records, gateEvidenceRecord(name, result.Invocation.Check.Phase, result.Evidence, result.Err, second.stepEvidence[name]))
	}
	if reused != len(batch.Checkpoints) {
		t.Fatalf("reused=%d, want %d checkpoints", reused, len(batch.Checkpoints))
	}
	record, err := runrecord.NewGateRecord(second.manifestPlan.ID, second.environment.ID, second.planHead, runrecord.OutcomeSucceeded, "", 1, records)
	if err != nil {
		t.Fatal(err)
	}
	content, err := record.Result.Content()
	if err != nil {
		t.Fatal(err)
	}
	replayed, err := runrecord.ParseGateResult(content.Data)
	if err != nil {
		t.Fatal(err)
	}
	for _, checkpoint := range batch.Checkpoints {
		found := false
		for _, result := range replayed.Steps {
			if result.Name != checkpoint.GateCheckName() {
				continue
			}
			found = true
			if result.Outcome != runrecord.StepReused {
				t.Fatalf("%s lost reuse outcome", checkpoint.ID)
			}
			if err := runrecord.VerifyCompletionAcceptanceEvidence(result.Evidence, second.planRef, checkpoint.Verify); err != nil {
				t.Fatalf("%s lost completion authority after serialization: %v", checkpoint.ID, err)
			}
			if err := runrecord.VerifyCompletionAcceptanceEvidence(result.Evidence, second.planRef, checkpoint.Verify+" -race"); err == nil {
				t.Fatal("changed command accepted old completion binding")
			}
			if err := runrecord.VerifyCompletionAcceptanceEvidence(result.Evidence, "other/step", checkpoint.Verify); err == nil {
				t.Fatal("different parent accepted old completion binding")
			}
		}
		if !found {
			t.Fatalf("missing checkpoint %s", checkpoint.ID)
		}
	}
	t.Logf("fresh checkpoints=%d reused checkpoints=%d parent integration executed=true immutable record roundtrip=true; historical Git recovery not exercised", len(batch.Checkpoints), reused)
}
