package gate

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/runrecord"
)

const checkpointCompletionRepoEnvironment = "OVERGO_TEST_CHECKPOINT_COMPLETION_REPO"

func TestReusedCheckpointCompletionAuthorityAcceptance(t *testing.T) {
	t.Parallel()
	if repo := os.Getenv(checkpointCompletionRepoEnvironment); repo != "" {
		document, err := plan.Load(filepath.Join(repo, plan.Path))
		if err != nil {
			t.Fatal(err)
		}
		store, err := overgodb.OpenReadOnly(filepath.Join(repo, "tmp", "completion-store"))
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		authority, err := plan.ResolveCompletionAuthority(t.Context(), repo, "HEAD", document, store)
		if err != nil {
			t.Fatalf("reopen completed checkpoint history: %v", err)
		}
		if !authority.ProtectsRevision() {
			t.Fatal("reopened completion did not enter protected history")
		}
		if _, _, open := plan.Current(document, "", authority); open {
			t.Fatal("completed fixture task remained dispatchable")
		}
		return
	}
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
	// Publish the actual reused members through the existing completion fixture
	// owner, then reopen its store and resolve the real Git completion trailer.
	// The fixture models publication, not a complete build/device/browser gate.
	document, err := plan.Load(filepath.Join(second.repo, plan.Path))
	if err != nil {
		t.Fatal(err)
	}
	advanced, err := plan.Advance(document, "audio", "dataset")
	if err != nil {
		t.Fatal(err)
	}
	preparation, err := runrecord.NewGatePreparation(candidateTreeKey(tree), second.environment.ID, time.Now())
	if err != nil {
		t.Fatal(err)
	}
	storePath := filepath.Join("tmp", "completion-store")
	preparationCommit := publishInterruptedPreparation(t, second.repo, storePath, second.environment, preparation)
	message, err := plan.CompletionCommitMessageWithMergeAuthority(
		[]byte("Complete executed checkpoint fixture"), document, "audio", "dataset",
		second.manifestPlan.ID, second.manifestPlan.CandidateManifest, preparation.ID, preparationCommit,
		plan.MergeProjectionSemanticUnion, artifact.ID{},
	)
	if err != nil {
		t.Fatal(err)
	}
	messagePath := filepath.Join(t.TempDir(), "message.txt")
	if err := os.WriteFile(messagePath, message, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := plan.Save(filepath.Join(second.repo, plan.Path), advanced); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, second.repo, "add", "--", plan.Path)
	runGitFixture(t, second.repo, "commit", "-q", "-F", messagePath)
	fixture := interruptedCommitFixture{
		repo: second.repo, storePath: storePath, commit: recoveryGit(t, second.repo, "rev-parse", "HEAD"),
		recipe: second.manifestPlan.ID, candidate: second.manifestPlan.CandidateManifest,
		environment: second.environment, preparation: preparation, preparationCommit: preparationCommit,
		manifest: *second.manifestPlan,
	}
	var members []runrecord.GateStep
	for _, result := range records {
		if result.Name != "acceptance" {
			members = append(members, result)
		}
	}
	_, publication := successfulAttemptBatchForReference(t, fixture, "audio", "dataset",
		document.Items[0].Steps[0].Verify, true, "executed-checkpoint-completion", members...)
	store := openRecoveryStore(t, fixture)
	_, publishErr := store.Commit(t.Context(), publication)
	closeErr := store.Close()
	if publishErr != nil || closeErr != nil {
		t.Fatalf("publish completion: %v; close: %v", publishErr, closeErr)
	}
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	reopen := exec.CommandContext(t.Context(), executable, "-test.run=^TestReusedCheckpointCompletionAuthorityAcceptance$", "-test.v")
	reopen.Env = append(os.Environ(), checkpointCompletionRepoEnvironment+"="+second.repo)
	if output, err := reopen.CombinedOutput(); err != nil {
		t.Fatalf("fresh-process completion dispatch: %v\n%s", err, output)
	}
	t.Logf("fresh checkpoints=%d reused checkpoints=%d parent integration executed=true immutable record roundtrip=true fresh-process store reopen=true Git completion resolved=true; publication fixture excludes full build/device/browser gating and historical repair", len(batch.Checkpoints), reused)
}
