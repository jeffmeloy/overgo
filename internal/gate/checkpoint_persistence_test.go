package gate

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/plan"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

const (
	checkpointCrashRepoEnvironment = "OVERGO_TEST_CHECKPOINT_CRASH_REPO"
	checkpointCrashExitCode        = 88
)

// persistenceInvocations: the fixture batch planned and manifest-bound the way
// the gate binds it; returns the bound invocations.
func persistenceInvocations(t *testing.T, g *gateContext, batch *plan.VerificationBatch, tree string) []automationcheck.Invocation {
	t.Helper()
	checks, err := g.batchAcceptanceChecks([]automationcheck.Check{
		gateCheck("acceptance", runrecord.PhaseTest, g.stepAcceptance),
	}, batch)
	if err != nil {
		t.Fatal(err)
	}
	invocations, err := automationcheck.Plan(checks, automationcheck.Impact{})
	if err != nil {
		t.Fatal(err)
	}
	manifest, err := automationcheck.BindManifestPlan(
		testutil.ArtifactID(t, artifact.KindProfile, "persist base"),
		testutil.ArtifactID(t, artifact.KindProfile, "persist candidate"),
		strings.Repeat("a", 64), candidateTreeKey(tree),
		automationcheck.Surface{Identity: "checkpoint persistence fixture"}, automationcheck.Impact{}, invocations,
	)
	if err != nil {
		t.Fatal(err)
	}
	g.manifestPlan = &manifest
	for index, invocation := range invocations {
		invocations[index], err = automationcheck.BindManifestExecution(manifest, invocation, []artifact.ID{manifest.CandidateManifest})
		if err != nil {
			t.Fatal(err)
		}
	}
	return invocations
}

// TestCheckpointPersistenceCrashProcess: helper; runs the fixture checks over
// the repo named by the environment and exits at the first persisted
// checkpoint success.
func TestCheckpointPersistenceCrashProcess(t *testing.T) {
	repo := os.Getenv(checkpointCrashRepoEnvironment)
	if repo == "" {
		return
	}
	g, batch, tree := verificationBatchContext(t, repo)
	batch.Flush = &plan.BatchFlush{Key: "persist", MaxSize: 2, MaxInterval: "1m", MaxBytes: 1 << 20}
	// The lifecycle test environment is deterministic, so the helper and the
	// parent bind the same cache environment.
	g.environment = lifecycleTestEnvironment(t)
	invocations := persistenceInvocations(t, g, batch, tree)
	gateCheckPersistedHook = func(name string) {
		if name == "acceptance-producer" {
			os.Exit(checkpointCrashExitCode)
		}
	}
	cache := g.loadRetryCache()
	if _, err := g.executeChecks(invocations, nil, map[artifact.ID]artifact.ID{}, &cache, nil); err != nil {
		t.Fatal(err)
	}
	t.Fatal("checks returned without reaching the persisted checkpoint boundary")
}

// TestCheckpointPersistenceSurvivesKill pins: a checkpoint success is on disk
// before the next check runs, a process killed right after it leaves that
// entry reusable, a never-run checkpoint stays absent, and a fresh run reuses
// the persisted checkpoint while executing the rest.
func TestCheckpointPersistenceSurvivesKill(t *testing.T) {
	g, batch, tree := verificationBatchFixture(t, "pass")
	batch.Flush = &plan.BatchFlush{Key: "persist", MaxSize: 2, MaxInterval: "1m", MaxBytes: 1 << 20}
	// The lifecycle test environment is deterministic, so the helper and the
	// parent bind the same cache environment.
	g.environment = lifecycleTestEnvironment(t)

	process := exec.Command(os.Args[0], "-test.run=^TestCheckpointPersistenceCrashProcess$", "-test.count=1")
	process.Env = append(os.Environ(), checkpointCrashRepoEnvironment+"="+g.repo)
	output, err := process.CombinedOutput()
	exitError, exitFailure := errors.AsType[*exec.ExitError](err)
	if !exitFailure || exitError.ExitCode() != checkpointCrashExitCode {
		t.Fatalf("crash helper = %v, output:\n%s", err, output)
	}
	if _, err := os.Stat(filepath.Join(g.repo, filepath.FromSlash(gateRetryFile))); err != nil {
		t.Fatalf("retry cache absent after kill: %v", err)
	}

	invocations := persistenceInvocations(t, g, batch, tree)
	cache := g.loadRetryCache()
	var producer, consumer automationcheck.Invocation
	for _, invocation := range invocations {
		switch invocation.Check.Name {
		case "acceptance-producer":
			producer = invocation
		case "acceptance-consumer":
			consumer = invocation
		}
	}
	slot, input, memoised := g.memoSlot(producer)
	if !memoised {
		t.Fatal("producer has no memo slot")
	}
	if evidence, found := cache.Lookup(slot, input); !found || !evidence.Reused {
		t.Fatalf("killed run did not persist the producer checkpoint: %+v %v", evidence, found)
	}
	if consumerSlot, consumerInput, ok := g.memoSlot(consumer); !ok {
		t.Fatal("consumer has no memo slot")
	} else if _, found := cache.Lookup(consumerSlot, consumerInput); found {
		t.Fatal("never-run consumer checkpoint is present in the cache")
	}

	results, err := g.executeChecks(invocations, nil, map[artifact.ID]artifact.ID{}, &cache, nil)
	if err != nil {
		t.Fatal(err)
	}
	reused, executed := 0, 0
	for _, result := range results {
		if result.Err != nil {
			t.Fatal(result.Err)
		}
		for _, checkpoint := range batch.Checkpoints {
			if result.Invocation.Check.Name != checkpoint.GateCheckName() {
				continue
			}
			record := gateEvidenceRecord(checkpoint.GateCheckName(), runrecord.PhaseTest, result.Evidence, result.Err, g.stepEvidence[checkpoint.GateCheckName()])
			if err := runrecord.VerifyCompletionAcceptanceEvidence(record.Evidence, g.planRef, checkpoint.Verify); err != nil {
				t.Fatalf("persisted %s completion binding: %v", checkpoint.ID, err)
			}
		}
		switch result.Invocation.Check.Name {
		case "acceptance-producer":
			if !result.Evidence.Reused {
				t.Fatalf("producer was executed again: %+v", result.Evidence)
			}
			reused++
		case "acceptance-consumer":
			if result.Evidence.Reused || result.Err != nil {
				t.Fatalf("consumer was not executed: %+v %v", result.Evidence, result.Err)
			}
			executed++
		}
	}
	if reused != 1 || executed != 1 {
		t.Fatalf("rerun reused=%d executed=%d", reused, executed)
	}
}
