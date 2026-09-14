package gate

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/codemanifest"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/runrecord"
	"overgo/internal/testevidence"
	"overgo/internal/testutil"
)

const (
	gateRecoveryCrashRepoEnvironment  = "OVERGO_TEST_GATE_RECOVERY_CRASH_REPO"
	gateRecoveryCrashStoreEnvironment = "OVERGO_TEST_GATE_RECOVERY_CRASH_STORE"
	gateRecoveryCrashStepEnvironment  = "OVERGO_TEST_GATE_RECOVERY_CRASH_STEP"
	gateRecoveryCrashExitCode         = 87
)

func TestRecoverInterruptedCommitRestoresParentAndFinalizesCancellation(t *testing.T) {
	// Each parallel recovery fixture owns its repository and store. Tests that
	// mutate process-wide failure hooks remain serial and finish before these.
	t.Parallel()
	fixture := newInterruptedCommitFixture(t)

	preparation, err := recoverInterruptedCommit(fixture.repo, fixture.storePath)
	if err != nil {
		t.Fatal(err)
	}
	if preparation != fixture.preparation.ID {
		t.Fatalf("recovered preparation = %s, want %s", preparation, fixture.preparation.ID)
	}
	if head := recoveryGit(t, fixture.repo, "rev-parse", "HEAD"); head != fixture.parent {
		t.Fatalf("HEAD = %s, want parent %s", head, fixture.parent)
	}
	if got, err := os.ReadFile(filepath.Join(fixture.repo, filepath.FromSlash(plan.Path))); err != nil {
		t.Fatal(err)
	} else if string(got) != string(fixture.planBefore) {
		t.Fatalf("restored plan differs\ngot:\n%s\nwant:\n%s", got, fixture.planBefore)
	}
	if got, err := os.ReadFile(filepath.Join(fixture.repo, fixture.sourcePath)); err != nil {
		t.Fatal(err)
	} else if string(got) != string(fixture.sourceAfter) {
		t.Fatalf("scoped source was not preserved: got %q want %q", got, fixture.sourceAfter)
	}
	if got := recoveryGit(t, fixture.repo, "diff", "--name-only"); got != fixture.sourcePath {
		t.Fatalf("unstaged recovery paths = %q, want %q", got, fixture.sourcePath)
	}
	if got := recoveryGit(t, fixture.repo, "diff", "--cached", "--name-only"); got != "" {
		t.Fatalf("recovery unexpectedly staged paths: %q", got)
	}
	assertNoCommitIntent(t, fixture.repo)

	store := openRecoveryStore(t, fixture)
	defer store.Close()
	finalization, found, err := runrecord.GateFinalizationForPreparation(
		t.Context(), store, fixture.preparation.ID,
	)
	if err != nil {
		t.Fatal(err)
	}
	if !found {
		t.Fatal("finalization is absent")
	}
	if finalization.State != runrecord.GateFinalized || finalization.Outcome != runrecord.OutcomeCancelled ||
		finalization.CodeCommit != fixture.commit || finalization.Result == nil {
		t.Fatalf("cancelled finalization = %+v", finalization)
	}
	gate, err := runrecord.RequireGateResult(t.Context(), store, *finalization.Result)
	if err != nil {
		t.Fatal(err)
	}
	if len(gate.Steps) != 1 || gate.Steps[0].DurationNS <= 1 {
		t.Fatalf("recovery step retained a synthetic duration: %+v", gate.Steps)
	}
}

func TestRecoverInterruptedStateCrashRetryBoundaries(t *testing.T) {
	t.Parallel()
	for _, step := range []string{"publication", "parent", "plan"} {
		t.Run(step, func(t *testing.T) {
			fixture := newInterruptedCommitFixture(t)
			var intent gateCommitIntent
			if err := readJSON(fixture.repo, gateCommitIntentFile, &intent); err != nil {
				t.Fatal(err)
			}
			locks, marker := gateGitLockAuthorityForTest(t, fixture.repo, intent)
			crashInterruptedRecoveryProcess(t, fixture.repo, fixture.storePath, step)
			assertExactGateGitLockResidues(t, locks, marker)
			assertGitReferenceTransactionLocksReleased(t, fixture.repo, intent)

			if head := recoveryGit(t, fixture.repo, "rev-parse", "HEAD"); head != fixture.parent {
				t.Fatalf("HEAD after %s crash = %s, want parent %s", step, head, fixture.parent)
			}
			if _, err := os.Stat(filepath.Join(fixture.repo, filepath.FromSlash(gateCommitIntentFile))); err != nil {
				t.Fatalf("%s crash lost durable recovery intent: %v", step, err)
			}
			store := openRecoveryStore(t, fixture)
			_, finalized, err := runrecord.GateFinalizationForPreparation(
				t.Context(), store, fixture.preparation.ID,
			)
			if closeErr := store.Close(); err != nil || closeErr != nil {
				t.Fatal(errors.Join(err, closeErr))
			}
			if finalized {
				t.Fatalf("%s crash published a cancellation before exact state recovery", step)
			}

			preparation, err := recoverInterruptedCommit(fixture.repo, fixture.storePath)
			if err != nil {
				t.Fatalf("retry after %s crash: %v", step, err)
			}
			if preparation != fixture.preparation.ID {
				t.Fatalf("retry after %s crash recovered %s, want %s", step, preparation, fixture.preparation.ID)
			}
			if got, err := os.ReadFile(filepath.Join(fixture.repo, filepath.FromSlash(plan.Path))); err != nil {
				t.Fatal(err)
			} else if !bytes.Equal(got, fixture.planBefore) {
				t.Fatalf("retry after %s crash restored the wrong plan", step)
			}
			if staged := recoveryGit(t, fixture.repo, "diff", "--cached", "--name-only"); staged != "" {
				t.Fatalf("retry after %s crash left staged paths: %q", step, staged)
			}
			assertGateGitLockPathsAbsent(t, locks)
			assertNoCommitIntent(t, fixture.repo)
		})
	}
}

func TestRecoverInterruptedStateCrashProcess(t *testing.T) {
	repo := os.Getenv(gateRecoveryCrashRepoEnvironment)
	if repo == "" {
		return
	}
	storePath := os.Getenv(gateRecoveryCrashStoreEnvironment)
	step := os.Getenv(gateRecoveryCrashStepEnvironment)
	if storePath == "" || step == "" {
		t.Fatal("gate recovery crash helper has incomplete authority")
	}
	gateRecoveryAfterStepHook = func(repository, recoveredStep string) {
		if repository == repo && recoveredStep == step {
			os.Exit(gateRecoveryCrashExitCode)
		}
	}
	if _, err := recoverInterruptedCommit(repo, storePath); err != nil {
		t.Fatal(err)
	}
	t.Fatalf("gate recovery returned without reaching %s crash boundary", step)
}

func crashInterruptedRecoveryProcess(t *testing.T, repo, storePath, step string) {
	t.Helper()
	process := exec.Command(os.Args[0], "-test.run=^TestRecoverInterruptedStateCrashProcess$", "-test.count=1")
	process.Env = append(
		os.Environ(),
		gateRecoveryCrashRepoEnvironment+"="+repo,
		gateRecoveryCrashStoreEnvironment+"="+storePath,
		gateRecoveryCrashStepEnvironment+"="+step,
	)
	output, err := process.CombinedOutput()
	exitError, exitFailure := errors.AsType[*exec.ExitError](err)
	if !exitFailure || exitError.ExitCode() != gateRecoveryCrashExitCode {
		t.Fatalf("%s crash helper = %v, output:\n%s", step, err, output)
	}
}

func gateGitLockAuthorityForTest(
	t *testing.T,
	repo string,
	intent gateCommitIntent,
) ([]gateGitLockPath, []byte) {
	t.Helper()
	indexPath, locks, err := gateGitManualLockPaths(repo, os.FileMode(intent.IndexMode), intent)
	if err != nil {
		t.Fatal(err)
	}
	if len(locks) != 7 {
		t.Fatalf("manual Git lock path count = %d, want 7", len(locks))
	}
	marker, err := gateGitLockMarkerBytes(repo, indexPath, intent)
	if err != nil {
		t.Fatal(err)
	}
	return locks, marker
}

func assertExactGateGitLockResidues(t *testing.T, locks []gateGitLockPath, marker []byte) {
	t.Helper()
	for _, lock := range locks {
		present, exact, err := inspectGateGitLockMarker(lock.path, marker, lock.mode)
		if err != nil {
			t.Fatal(err)
		}
		if !present || !exact {
			t.Fatalf("hard exit left %s lock marker = (present %t, exact %t)", lock.name, present, exact)
		}
	}
}

func assertGateGitLockPathsAbsent(t *testing.T, locks []gateGitLockPath) {
	t.Helper()
	for _, lock := range locks {
		if _, err := os.Lstat(lock.path); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("resolved manual Git lock %s remains: %v", lock.name, err)
		}
	}
}

func assertGitReferenceTransactionLocksReleased(t *testing.T, repo string, intent gateCommitIntent) {
	t.Helper()
	var paths []string
	for _, name := range []string{"HEAD", intent.HeadReference} {
		path, err := gitMetadataPath(repo, name)
		if err != nil {
			t.Fatal(err)
		}
		paths = append(paths, path+".lock")
	}
	deadline := time.Now().Add(5 * time.Second)
	for {
		var present []string
		for _, path := range paths {
			if _, err := os.Lstat(path); err == nil {
				present = append(present, path)
			} else if !errors.Is(err, os.ErrNotExist) {
				t.Fatal(err)
			}
		}
		if len(present) == 0 {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("Git update-ref child left transaction locks after parent death: %v", present)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestGateGitLockRecoveryReclaimsEveryExactMarkerPrefix(t *testing.T) {
	t.Parallel()
	fixture := newInterruptedCommitFixture(t)
	var intent gateCommitIntent
	if err := readJSON(fixture.repo, gateCommitIntentFile, &intent); err != nil {
		t.Fatal(err)
	}
	locks, marker := gateGitLockAuthorityForTest(t, fixture.repo, intent)
	for count := 1; count <= len(locks); count++ {
		for index := range count {
			if err := publishGateGitLockMarker(locks[index], marker); err != nil {
				t.Fatalf("publish prefix %d lock %s: %v", count, locks[index].name, err)
			}
		}
		if err := installCompletionIndexForTest(fixture.repo, intent); err != nil {
			t.Fatalf("reclaim exact marker prefix %d: %v", count, err)
		}
		assertGateGitLockPathsAbsent(t, locks)
	}
}

func TestGateGitLockRecoveryRefusesMixedExactAndForeignMarkers(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		name   string
		marker func([]byte) []byte
	}{
		{name: "partial", marker: func(marker []byte) []byte { return bytes.Clone(marker[:len(marker)-1]) }},
		{name: "foreign", marker: func(marker []byte) []byte { return bytes.Repeat([]byte("x"), len(marker)) }},
	} {
		t.Run(test.name, func(t *testing.T) {
			fixture := newInterruptedCommitFixture(t)
			var intent gateCommitIntent
			if err := readJSON(fixture.repo, gateCommitIntentFile, &intent); err != nil {
				t.Fatal(err)
			}
			locks, marker := gateGitLockAuthorityForTest(t, fixture.repo, intent)
			if err := publishGateGitLockMarker(locks[0], marker); err != nil {
				t.Fatal(err)
			}
			foreign := test.marker(marker)
			if err := os.WriteFile(locks[1].path, foreign, locks[1].mode); err != nil {
				t.Fatal(err)
			}
			if err := os.Chmod(locks[1].path, locks[1].mode); err != nil {
				t.Fatal(err)
			}

			if err := installCompletionIndexForTest(fixture.repo, intent); err == nil ||
				!strings.Contains(err.Error(), "partial or foreign ownership") {
				t.Fatalf("mixed %s marker recovery = %v", test.name, err)
			}
			present, exact, err := inspectGateGitLockMarker(locks[0].path, marker, locks[0].mode)
			if err != nil {
				t.Fatal(err)
			}
			if !present || !exact {
				t.Fatalf("refusal deleted the exact marker before finding %s contents", test.name)
			}
			if got, err := os.ReadFile(locks[1].path); err != nil {
				t.Fatal(err)
			} else if !bytes.Equal(got, foreign) {
				t.Fatalf("refusal changed %s lock contents", test.name)
			}
		})
	}
}

func TestGateGitStateProcessGuardRefusesNestedTransaction(t *testing.T) {
	fixture := newInterruptedCommitFixture(t)
	var intent gateCommitIntent
	if err := readJSON(fixture.repo, gateCommitIntentFile, &intent); err != nil {
		t.Fatal(err)
	}
	var nestedErr error
	gateIndexCASLockedHook = func(repository string) {
		gateIndexCASLockedHook = nil
		nestedErr = installCompletionIndexForTest(repository, intent)
	}
	t.Cleanup(func() { gateIndexCASLockedHook = nil })

	if err := installCompletionIndexForTest(fixture.repo, intent); err != nil {
		t.Fatal(err)
	}
	if nestedErr == nil || !strings.Contains(nestedErr.Error(), "another Git-state transaction is active") {
		t.Fatalf("nested Git-state transaction = %v", nestedErr)
	}
}

func TestRecoverInterruptedCommitFinalizesMatchingHeartbeat(t *testing.T) {
	t.Parallel()
	fixture := newInterruptedCommitFixture(t)
	heartbeat := runrecord.GateHeartbeat{
		Version: artifact.InitialDocumentVersion, State: runrecord.HeartbeatRunning,
		Preparation: fixture.preparation.ID, TreeKey: fixture.preparation.TreeKey,
		Environment: fixture.preparation.Environment, PID: os.Getpid(), Updated: time.Now().UTC(),
	}
	if err := writeJSON(fixture.repo, gateHeartbeatFile, heartbeat, 0o644); err != nil {
		t.Fatal(err)
	}

	if _, err := recoverInterruptedCommit(fixture.repo, fixture.storePath); err != nil {
		t.Fatal(err)
	}
	var finalized runrecord.GateHeartbeat
	if err := readJSON(fixture.repo, gateHeartbeatFile, &finalized); err != nil {
		t.Fatal(err)
	}
	if finalized.State != runrecord.HeartbeatFinalized || finalized.Preparation != fixture.preparation.ID {
		t.Fatalf("recovered heartbeat = %+v", finalized)
	}
	if err := requireNoPendingGateState(fixture.repo, fixture.storePath); err != nil {
		t.Fatalf("recovered gate state remained blocked: %v", err)
	}
}

func TestRecoverInterruptedCommitClearsStaleIntentAfterSuccessfulAttempt(t *testing.T) {
	t.Parallel()
	fixture := newInterruptedCommitFixture(t)
	attempt := publishSuccessfulInterruptedAttempt(t, fixture, true)

	preparation, err := recoverInterruptedCommit(fixture.repo, fixture.storePath)
	if err != nil {
		t.Fatal(err)
	}
	if preparation != fixture.preparation.ID {
		t.Fatalf("recovered preparation = %s, want %s", preparation, fixture.preparation.ID)
	}
	if head := recoveryGit(t, fixture.repo, "rev-parse", "HEAD"); head != fixture.commit {
		t.Fatalf("HEAD rolled back to %s after successful finalization; want %s", head, fixture.commit)
	}
	if got, err := os.ReadFile(filepath.Join(fixture.repo, filepath.FromSlash(plan.Path))); err != nil {
		t.Fatal(err)
	} else if string(got) != string(fixture.planAfter) {
		t.Fatalf("completed plan was rolled back\ngot:\n%s\nwant:\n%s", got, fixture.planAfter)
	}
	if got := recoveryGit(t, fixture.repo, "status", "--porcelain", "--untracked-files=no"); got != "" {
		t.Fatalf("successful recovery dirtied tracked files: %q", got)
	}
	assertNoCommitIntent(t, fixture.repo)

	store := openRecoveryStore(t, fixture)
	defer store.Close()
	attempts, err := runrecord.AttemptsForPreparation(t.Context(), store, fixture.preparation.ID)
	if err != nil {
		t.Fatal(err)
	}
	if len(attempts) != 1 || attempts[0].ID != attempt.ID {
		t.Fatalf("attempts = %+v, want only %s", attempts, attempt.ID)
	}
	if _, err := runrecord.VerifyAttemptGate(t.Context(), store, attempts[0]); err != nil {
		t.Fatalf("successful interrupted attempt is incomplete: %v", err)
	}
	finalization, found, err := runrecord.GateFinalizationForPreparation(
		t.Context(), store, fixture.preparation.ID,
	)
	if err != nil {
		t.Fatal(err)
	}
	if !found || finalization.Outcome != runrecord.OutcomeSucceeded {
		t.Fatalf("finalization = (%+v, %t), want one success", finalization, found)
	}
}

func TestRecoverInterruptedCommitRollsBackIncompleteSuccessfulFinalization(t *testing.T) {
	t.Parallel()
	fixture := newInterruptedCommitFixture(t)
	var retryIntent gateCommitIntent
	if err := readJSON(fixture.repo, gateCommitIntentFile, &retryIntent); err != nil {
		t.Fatal(err)
	}
	retryIntent.Commit = fixture.commit
	publishSuccessfulInterruptedAttempt(t, fixture, false)

	if _, err := recoverInterruptedCommit(fixture.repo, fixture.storePath); err != nil {
		t.Fatal(err)
	}
	if head := recoveryGit(t, fixture.repo, "rev-parse", "HEAD"); head != fixture.parent {
		t.Fatalf("HEAD = %s, want restored parent %s", head, fixture.parent)
	}
	assertNoCommitIntent(t, fixture.repo)
	store := openRecoveryStore(t, fixture)
	finalization, found, err := runrecord.GateFinalizationForPreparation(
		t.Context(), store, fixture.preparation.ID,
	)
	if err != nil {
		t.Fatal(err)
	}
	if !found || finalization.Outcome != runrecord.OutcomeSucceeded {
		t.Fatalf("unreachable incomplete finalization = (%+v, %t), want the one append-only record", finalization, found)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	if err := writeGateCommitIntent(fixture.repo, retryIntent); err != nil {
		t.Fatal(err)
	}
	if _, err := recoverInterruptedCommit(fixture.repo, fixture.storePath); err != nil {
		t.Fatalf("retry after rollback before intent cleanup: %v", err)
	}
	assertNoCommitIntent(t, fixture.repo)
	if err := writeGateCommitIntent(fixture.repo, retryIntent); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(
		filepath.Join(fixture.repo, filepath.FromSlash(plan.Path)), fixture.planAfter, 0o644,
	); err != nil {
		t.Fatal(err)
	}
	if _, err := recoverInterruptedCommit(fixture.repo, fixture.storePath); err != nil {
		t.Fatalf("retry after parent reset before plan restoration: %v", err)
	}
	if restored, err := planBytesMatch(fixture.repo, fixture.planBefore); err != nil || !restored {
		t.Fatalf("retry did not restore plan: (%v, %v)", restored, err)
	}
	assertNoCommitIntent(t, fixture.repo)
}

func TestReconcileGateDebtRetainsRollbackAuthorityForIncompleteSuccess(t *testing.T) {
	t.Parallel()
	fixture := newInterruptedCommitFixture(t)
	var intent gateCommitIntent
	if err := readJSON(fixture.repo, gateCommitIntentFile, &intent); err != nil {
		t.Fatal(err)
	}
	intent.Commit = fixture.commit
	if err := writeGateCommitIntent(fixture.repo, intent); err != nil {
		t.Fatal(err)
	}
	_, batch := successfulInterruptedAttemptBatch(
		t,
		fixture,
		false,
		"gate/final/"+fixture.preparation.ID.String(),
	)
	gate := gateContext{repo: fixture.repo, preparation: fixture.preparation}
	if err := gate.oweRecord(batch, errors.New("injected record outage")); err == nil {
		t.Fatal("record debt did not retain its triggering failure")
	}
	if preparation, err := reconcileGateDebt(fixture.repo, fixture.storePath); err == nil ||
		preparation != fixture.preparation.ID || !strings.Contains(err.Error(), "commit was rolled back") {
		t.Fatalf("incomplete success reconciliation = (%s, %v)", preparation, err)
	}
	if _, err := os.Stat(filepath.Join(fixture.repo, filepath.FromSlash(gateDebtFile))); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("durable invalid batch remained record debt: %v", err)
	}
	if head := recoveryGit(t, fixture.repo, "rev-parse", "HEAD"); head != fixture.parent {
		t.Fatalf("HEAD = %s, want restored parent %s", head, fixture.parent)
	}
	assertNoCommitIntent(t, fixture.repo)
}

func TestReconcileGateDebtAcceptsExactSuccessfulChain(t *testing.T) {
	t.Parallel()
	fixture := newInterruptedCommitFixture(t)
	var intent gateCommitIntent
	if err := readJSON(fixture.repo, gateCommitIntentFile, &intent); err != nil {
		t.Fatal(err)
	}
	intent.Commit = fixture.commit
	if err := writeGateCommitIntent(fixture.repo, intent); err != nil {
		t.Fatal(err)
	}
	attempt, batch := successfulInterruptedAttemptBatch(
		t,
		fixture,
		true,
		"gate/final/"+fixture.preparation.ID.String(),
	)
	gate := gateContext{repo: fixture.repo, preparation: fixture.preparation}
	if err := gate.oweRecord(batch, errors.New("injected record outage")); err == nil {
		t.Fatal("record debt did not retain its triggering failure")
	}
	heartbeat := runrecord.GateHeartbeat{
		Version: artifact.InitialDocumentVersion, State: runrecord.HeartbeatRecordDebt,
		Preparation: fixture.preparation.ID, TreeKey: fixture.preparation.TreeKey,
		Environment: fixture.preparation.Environment, PID: os.Getpid(), Updated: time.Now().UTC(),
	}
	if err := writeJSON(fixture.repo, gateHeartbeatFile, heartbeat, 0o644); err != nil {
		t.Fatal(err)
	}
	if preparation, err := reconcileGateDebt(fixture.repo, fixture.storePath); err != nil ||
		preparation != fixture.preparation.ID {
		t.Fatalf("exact success reconciliation = (%s, %v)", preparation, err)
	}
	if head := recoveryGit(t, fixture.repo, "rev-parse", "HEAD"); head != fixture.commit {
		t.Fatalf("HEAD = %s, want completed commit %s", head, fixture.commit)
	}
	if _, err := os.Stat(filepath.Join(fixture.repo, filepath.FromSlash(gateDebtFile))); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("reconciled debt remains: %v", err)
	}
	assertNoCommitIntent(t, fixture.repo)
	var finalized runrecord.GateHeartbeat
	if err := readJSON(fixture.repo, gateHeartbeatFile, &finalized); err != nil {
		t.Fatal(err)
	}
	if finalized.State != runrecord.HeartbeatFinalized || finalized.Preparation != fixture.preparation.ID {
		t.Fatalf("reconciled heartbeat = %+v", finalized)
	}
	store := openRecoveryStore(t, fixture)
	defer store.Close()
	if _, err := runrecord.VerifyAttemptGate(t.Context(), store, attempt); err != nil {
		t.Fatalf("reconciled attempt chain: %v", err)
	}
}

func TestRecoverInterruptedCommitRefusesDifferentBranchAtSameCommit(t *testing.T) {
	t.Parallel()
	fixture := newInterruptedCommitFixture(t)
	runGitFixture(t, fixture.repo, "checkout", "-q", "-b", "other-recovery-branch")

	if _, err := recoverInterruptedCommit(fixture.repo, fixture.storePath); err == nil ||
		!strings.Contains(err.Error(), "HEAD reference moved") {
		t.Fatalf("different-branch recovery error = %v", err)
	}
	if head := recoveryGit(t, fixture.repo, "rev-parse", "HEAD"); head != fixture.commit {
		t.Fatalf("different branch moved to %s, want %s", head, fixture.commit)
	}
}

func TestRecoverInterruptedCommitRefusesUnboundAdvancedHead(t *testing.T) {
	t.Parallel()
	fixture := newInterruptedCommitFixture(t)
	var intent gateCommitIntent
	if err := readJSON(fixture.repo, gateCommitIntentFile, &intent); err != nil {
		t.Fatal(err)
	}
	intent.Commit = ""
	if err := writeGateCommitIntent(fixture.repo, intent); err == nil ||
		!strings.Contains(err.Error(), "invalid interrupted commit intent") {
		t.Fatalf("unbound intent rewrite result = %v", err)
	}
	if head := recoveryGit(t, fixture.repo, "rev-parse", "HEAD"); head != fixture.commit {
		t.Fatalf("unbound advanced HEAD moved to %s, want %s", head, fixture.commit)
	}
	var persisted gateCommitIntent
	if err := readJSON(fixture.repo, gateCommitIntentFile, &persisted); err != nil {
		t.Fatal(err)
	}
	if persisted.Commit != fixture.commit {
		t.Fatalf("refused unbound rewrite persisted commit %q, want %q", persisted.Commit, fixture.commit)
	}
}

func TestInterruptedRollbackCASPreservesAdvancedCommit(t *testing.T) {
	t.Parallel()
	fixture := newInterruptedCommitFixture(t)
	var intent gateCommitIntent
	if err := readJSON(fixture.repo, gateCommitIntentFile, &intent); err != nil {
		t.Fatal(err)
	}
	intent.Commit = fixture.commit
	runGitFixture(t, fixture.repo, "commit", "-q", "--allow-empty", "-m", "concurrent advance")
	advanced := recoveryGit(t, fixture.repo, "rev-parse", "HEAD")
	if _, err := recoverInterruptedCommit(fixture.repo, fixture.storePath); err == nil ||
		!strings.Contains(err.Error(), "moved beyond interrupted commit") {
		t.Fatalf("concurrent rollback error = %v", err)
	}
	if head := recoveryGit(t, fixture.repo, "rev-parse", "HEAD"); head != advanced {
		t.Fatalf("concurrent commit was removed: HEAD = %s, want %s", head, advanced)
	}
}

func TestRecoverInterruptedCommitRefusesUnboundPreCASIndex(t *testing.T) {
	t.Parallel()
	fixture := newInterruptedCommitFixture(t)
	var intent gateCommitIntent
	if err := readJSON(fixture.repo, gateCommitIntentFile, &intent); err != nil {
		t.Fatal(err)
	}
	intent.Commit = fixture.commit
	if err := writeGateCommitIntent(fixture.repo, intent); err != nil {
		t.Fatal(err)
	}
	runGitFixture(
		t, fixture.repo, "update-ref", intent.HeadReference, intent.Parent, fixture.commit,
	)
	runGitFixture(t, fixture.repo, "read-tree", intent.IndexTree)
	stagedOnly := []byte("package candidate\n\nconst value = 3\n")
	if err := os.WriteFile(filepath.Join(fixture.repo, fixture.sourcePath), stagedOnly, 0o644); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, fixture.repo, "add", "--", fixture.sourcePath)
	stagedBlob := recoveryGit(t, fixture.repo, "rev-parse", ":"+fixture.sourcePath)
	worktreeOnly := []byte("package candidate\n\nconst value = 4\n")
	if err := os.WriteFile(filepath.Join(fixture.repo, fixture.sourcePath), worktreeOnly, 0o644); err != nil {
		t.Fatal(err)
	}

	if _, err := recoverInterruptedCommit(fixture.repo, fixture.storePath); err == nil ||
		!strings.Contains(err.Error(), "Git index moved beyond the write-ahead states") {
		t.Fatalf("unbound pre-CAS index recovery error = %v", err)
	}
	if head := recoveryGit(t, fixture.repo, "rev-parse", "HEAD"); head != fixture.parent {
		t.Fatalf("refused recovery moved HEAD to %s, want parent %s", head, fixture.parent)
	}
	if currentBlob := recoveryGit(t, fixture.repo, "rev-parse", ":"+fixture.sourcePath); currentBlob != stagedBlob {
		t.Fatalf("refused recovery replaced staged blob %s with %s", stagedBlob, currentBlob)
	}
	if current, err := os.ReadFile(filepath.Join(fixture.repo, fixture.sourcePath)); err != nil {
		t.Fatal(err)
	} else if !bytes.Equal(current, worktreeOnly) {
		t.Fatalf("refused recovery changed worktree bytes to %q", current)
	}
	if _, err := os.Stat(filepath.Join(fixture.repo, filepath.FromSlash(gateCommitIntentFile))); err != nil {
		t.Fatalf("refused recovery removed its durable intent: %v", err)
	}
}

func TestRecoveryPreLockMergeRaceRefusesBeforeMutation(t *testing.T) {
	fixture := newInterruptedCommitFixture(t)
	side := prepareRecoveryMergeRaceBranch(t, fixture)
	var mergeErr error
	var mergeOutput []byte
	gateRecoveryBeforeLockHook = func(repository string) {
		process := exec.Command("git", "merge", "--no-ff", "--no-commit", side)
		process.Dir = repository
		mergeOutput, mergeErr = process.CombinedOutput()
	}
	t.Cleanup(func() { gateRecoveryBeforeLockHook = nil })

	if _, err := recoverInterruptedCommit(fixture.repo, fixture.storePath); err == nil {
		t.Fatal("recovery accepted a merge that won before the exact state lock")
	}
	if mergeErr != nil {
		t.Fatalf("pre-lock merge did not win: %v: %s", mergeErr, mergeOutput)
	}
	if head := recoveryGit(t, fixture.repo, "rev-parse", "HEAD"); head != fixture.commit {
		t.Fatalf("pre-lock refusal moved HEAD to %s, want %s", head, fixture.commit)
	}
	if got, err := os.ReadFile(filepath.Join(fixture.repo, filepath.FromSlash(plan.Path))); err != nil {
		t.Fatal(err)
	} else if !bytes.Equal(got, fixture.planAfter) {
		t.Fatalf("pre-lock refusal changed plan bytes\ngot:\n%s\nwant:\n%s", got, fixture.planAfter)
	}
	if _, err := os.Stat(mustGateValue(gitMetadataPath(fixture.repo, "MERGE_HEAD"))); err != nil {
		t.Fatalf("pre-lock refusal removed the winning merge state: %v", err)
	}
}

func TestRecoveryIndexModeRaceRefusesBeforeHeadMutation(t *testing.T) {
	t.Parallel()
	fixture := newInterruptedCommitFixture(t)
	indexPath, err := gateIndexPath(fixture.repo)
	if err != nil {
		t.Fatal(err)
	}
	before, err := os.Stat(indexPath)
	if err != nil {
		t.Fatal(err)
	}
	changedMode := before.Mode().Perm() &^ 0o200
	if changedMode == before.Mode().Perm() {
		changedMode |= 0o200
	}
	if err := os.Chmod(indexPath, changedMode); err != nil {
		t.Fatal(err)
	}
	after, err := os.Stat(indexPath)
	if err != nil {
		t.Fatal(err)
	}
	if after.Mode().Perm() == before.Mode().Perm() {
		t.Skip("filesystem does not expose index permission-mode changes")
	}

	if _, err := recoverInterruptedCommit(fixture.repo, fixture.storePath); err == nil ||
		!strings.Contains(err.Error(), "Git index moved before exact interrupted-state recovery") {
		t.Fatalf("index mode race recovery error = %v", err)
	}
	if head := recoveryGit(t, fixture.repo, "rev-parse", "HEAD"); head != fixture.commit {
		t.Fatalf("index mode refusal moved HEAD to %s, want %s", head, fixture.commit)
	}
	if got, err := os.ReadFile(filepath.Join(fixture.repo, filepath.FromSlash(plan.Path))); err != nil {
		t.Fatal(err)
	} else if !bytes.Equal(got, fixture.planAfter) {
		t.Fatalf("index mode refusal changed plan bytes\ngot:\n%s\nwant:\n%s", got, fixture.planAfter)
	}
}

func TestRecoveryLockBlocksMidTransactionMerge(t *testing.T) {
	fixture := newInterruptedCommitFixture(t)
	side := prepareRecoveryMergeRaceBranch(t, fixture)
	var mergeErr error
	var mergeOutput []byte
	gateRecoveryAfterStepHook = func(repository, step string) {
		if step != "admission" {
			return
		}
		process := exec.Command("git", "merge", "--no-ff", "--no-commit", side)
		process.Dir = repository
		mergeOutput, mergeErr = process.CombinedOutput()
	}
	t.Cleanup(func() { gateRecoveryAfterStepHook = nil })

	if _, err := recoverInterruptedCommit(fixture.repo, fixture.storePath); err != nil {
		t.Fatal(err)
	}
	if mergeErr == nil || !strings.Contains(string(mergeOutput), "index.lock") {
		t.Fatalf("mid-transaction merge under exact lock = (%v, %q)", mergeErr, mergeOutput)
	}
	if head := recoveryGit(t, fixture.repo, "rev-parse", "HEAD"); head != fixture.parent {
		t.Fatalf("recovered HEAD = %s, want %s", head, fixture.parent)
	}
	if got, err := os.ReadFile(filepath.Join(fixture.repo, filepath.FromSlash(plan.Path))); err != nil {
		t.Fatal(err)
	} else if !bytes.Equal(got, fixture.planBefore) {
		t.Fatalf("recovered plan differs\ngot:\n%s\nwant:\n%s", got, fixture.planBefore)
	}
	if _, err := os.Stat(mustGateValue(gitMetadataPath(fixture.repo, "MERGE_HEAD"))); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("blocked mid-transaction merge left MERGE_HEAD: %v", err)
	}
}

func TestRecoveryParentPublicationHandoffRefusesWinningBranchWriter(t *testing.T) {
	fixture := newInterruptedCommitFixture(t)
	var intent gateCommitIntent
	if err := readJSON(fixture.repo, gateCommitIntentFile, &intent); err != nil {
		t.Fatal(err)
	}
	messagePath := filepath.Join(t.TempDir(), "competing-recovery.txt")
	if err := os.WriteFile(messagePath, []byte("competing recovery publication\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	competing, err := createGateCommit(fixture.repo, gateCommitIntent{
		Parent: intent.Parent,
		Tree:   intent.Tree,
	}, messagePath)
	if err != nil {
		t.Fatal(err)
	}
	var raceErr error
	gateRecoveryAfterStepHook = func(repository, step string) {
		if step != "publication" {
			return
		}
		_, raceErr = gitAuthorityOutput(
			repository, "update-ref", "-m", "competing recovery writer",
			intent.HeadReference, competing, intent.Parent,
		)
	}
	t.Cleanup(func() { gateRecoveryAfterStepHook = nil })

	if _, err := recoverInterruptedCommit(fixture.repo, fixture.storePath); err == nil ||
		!strings.Contains(err.Error(), "guard restored parent authority") {
		t.Fatalf("recovery handoff race error = %v", err)
	}
	if raceErr != nil {
		t.Fatalf("competing branch writer did not win publication handoff: %v", raceErr)
	}
	if head := recoveryGit(t, fixture.repo, "rev-parse", "HEAD"); head != competing {
		t.Fatalf("refused recovery HEAD = %s, want competing commit %s", head, competing)
	}
	index, err := captureGateIndex(fixture.repo)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(index.Data, intent.IndexAfter) {
		t.Fatal("refused recovery changed the index after losing the parent publication handoff")
	}
	if got, err := os.ReadFile(filepath.Join(fixture.repo, filepath.FromSlash(plan.Path))); err != nil {
		t.Fatal(err)
	} else if !bytes.Equal(got, fixture.planAfter) {
		t.Fatal("refused recovery changed the plan after losing the parent publication handoff")
	}
	if _, err := os.Stat(filepath.Join(fixture.repo, filepath.FromSlash(gateCommitIntentFile))); err != nil {
		t.Fatalf("refused recovery removed its durable intent: %v", err)
	}
}

func TestRecoverInterruptedCommitRecognizesExactPriorCancellation(t *testing.T) {
	t.Parallel()
	fixture := newInterruptedCommitFixture(t)
	var intent gateCommitIntent
	if err := readJSON(fixture.repo, gateCommitIntentFile, &intent); err != nil {
		t.Fatal(err)
	}
	if _, err := recoverInterruptedCommit(fixture.repo, fixture.storePath); err != nil {
		t.Fatal(err)
	}
	intent.Commit = fixture.commit
	if err := writeGateCommitIntent(fixture.repo, intent); err != nil {
		t.Fatal(err)
	}
	if _, err := recoverInterruptedCommit(fixture.repo, fixture.storePath); err != nil {
		t.Fatalf("idempotent cancellation recovery: %v", err)
	}
	assertNoCommitIntent(t, fixture.repo)
}

func TestRecoverInterruptedMergeRestoresExactPendingMerge(t *testing.T) {
	t.Parallel()
	repo := t.TempDir()
	if err := os.MkdirAll(filepath.Join(repo, "tmp"), 0o755); err != nil {
		t.Fatal(err)
	}
	storePath := "store"
	retiredPath := "retired.txt"
	planPath := filepath.Join(repo, filepath.FromSlash(plan.Path))
	if err := os.MkdirAll(filepath.Dir(planPath), 0o755); err != nil {
		t.Fatal(err)
	}
	document := plan.Plan{
		Campaign: "interrupted-merge", Doctrine: "test exact merge recovery",
		Items: []plan.Item{{
			ID: "ratchet", Status: plan.StatusOpen,
			Steps: []plan.Step{{
				ID: "merge", Status: plan.StatusOpen, Verify: "go test ./cmd/gate",
			}},
		}},
	}
	if err := plan.Save(planPath, document); err != nil {
		t.Fatal(err)
	}
	planBefore, err := os.ReadFile(planPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo, retiredPath), []byte("tracked before merge\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "init", "-q")
	runGitFixture(t, repo, "config", "user.email", "merge-recovery@example.invalid")
	runGitFixture(t, repo, "config", "user.name", "Merge Recovery Test")
	runGitFixture(t, repo, "config", "core.autocrlf", "false")
	runGitFixture(t, repo, "add", "--", plan.Path, retiredPath)
	runGitFixture(t, repo, "commit", "-q", "-m", "merge base")
	baseBranch := recoveryGit(t, repo, "rev-parse", "--abbrev-ref", "HEAD")

	runGitFixture(t, repo, "checkout", "-q", "-b", "merge-candidate")
	if err := os.Remove(filepath.Join(repo, retiredPath)); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo, "merged.go"), []byte("package merged\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "add", "-A", "--", retiredPath, "merged.go")
	runGitFixture(t, repo, "commit", "-q", "-m", "candidate side")
	runGitFixture(t, repo, "checkout", "-q", baseBranch)
	if err := os.WriteFile(filepath.Join(repo, "parent.go"), []byte("package parent\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "add", "--", "parent.go")
	runGitFixture(t, repo, "commit", "-q", "-m", "parent side")
	parent := recoveryGit(t, repo, "rev-parse", "HEAD")
	runGitFixture(t, repo, "merge", "--no-ff", "--no-commit", "merge-candidate")

	merge, err := capturePendingMerge(repo)
	if err != nil {
		t.Fatal(err)
	}
	if merge == nil {
		t.Fatal("git merge did not produce pending merge metadata")
	}
	mergeHead := append([]byte(nil), merge.Head...)
	mergeMode := append([]byte(nil), merge.Mode...)
	mergeMessage := append([]byte(nil), merge.Message...)
	mergeIndexTree := merge.IndexTree
	mergeIndex, err := captureGateIndex(repo)
	if err != nil {
		t.Fatal(err)
	}
	if mergeIndex.Tree != mergeIndexTree {
		t.Fatalf("captured merge index tree = %s, want %s", mergeIndex.Tree, mergeIndexTree)
	}
	mergeRestore, err := buildGateIndexForTree(repo, mergeIndex.Tree)
	if err != nil {
		t.Fatal(err)
	}

	environment, err := runrecord.NewEnvironment(runrecord.Environment{
		Host: "test", OS: "test", Arch: "test", Device: "host", Backend: "go", Driver: "none", Runtime: "go-test",
	})
	if err != nil {
		t.Fatal(err)
	}
	preparation, err := runrecord.NewGatePreparation(strings.Repeat("b", 64), environment.ID, time.Unix(200, 1))
	if err != nil {
		t.Fatal(err)
	}
	preparationCommit := publishInterruptedPreparation(t, repo, storePath, environment, preparation)

	planAfter, err := advancedPlanBytes(planBefore, "ratchet/merge")
	if err != nil {
		t.Fatal(err)
	}
	recipe := testutil.ArtifactID(t, artifact.KindRecipe, "merge completion manifest")
	candidate := testutil.ArtifactID(t, artifact.KindProfile, "merge code manifest")
	intent := gateCommitIntent{
		Version: artifact.InitialDocumentVersion, Preparation: preparation.ID,
		PreparationCommit: preparationCommit, Recipe: recipe,
		CandidateManifest: candidate, Parent: parent,
		PlanProjection: plan.MergeProjectionSemanticUnion,
		HeadReference:  mustCurrentHeadReference(t, repo), Merge: merge,
		IndexTree: mergeIndexTree,
		PlanRef:   "ratchet/merge", Paths: []string{"merged.go", retiredPath, plan.Path},
		Plan: planBefore, AdvancedPlan: planAfter, PlanMode: 0o644,
	}
	if err := os.WriteFile(planPath, planAfter, 0o644); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "add", "--", plan.Path)
	finalIndex, err := captureGateIndex(repo)
	if err != nil {
		t.Fatal(err)
	}
	intent.Tree = finalIndex.Tree
	intent.IndexBefore, intent.IndexAfter, intent.IndexRestore = mergeIndex.Data, finalIndex.Data, mergeRestore.Data
	intent.IndexMode = uint32(mergeIndex.Mode.Perm())
	message, err := plan.CompletionCommitMessageWithMergeAuthority(
		[]byte("Complete interrupted merge"), document, "ratchet", "merge",
		recipe, candidate, preparation.ID, preparationCommit, plan.MergeProjectionSemanticUnion, artifact.ID{},
	)
	if err != nil {
		t.Fatal(err)
	}
	messagePath := filepath.Join(t.TempDir(), "merge-message.txt")
	if err := os.WriteFile(messagePath, message, 0o600); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "commit", "-q", "-F", messagePath)
	commit := recoveryGit(t, repo, "rev-parse", "HEAD")
	postCommitIndex, err := captureGateIndex(repo)
	if err != nil {
		t.Fatal(err)
	}
	intent.Commit, intent.Tree, intent.IndexAfter = commit, postCommitIndex.Tree, postCommitIndex.Data
	if err := writeGateCommitIntent(repo, intent); err != nil {
		t.Fatal(err)
	}
	if err := readJSON(repo, gateCommitIntentFile, &intent); err != nil {
		t.Fatal(err)
	}
	if intent.PlanProjection != plan.MergeProjectionSemanticUnion {
		t.Fatalf("recovered merge projection = %q", intent.PlanProjection)
	}
	recreated := []byte("recreated after the merge commit\n")
	if err := os.WriteFile(filepath.Join(repo, retiredPath), recreated, 0o644); err != nil {
		t.Fatal(err)
	}

	locks, lockMarker := gateGitLockAuthorityForTest(t, repo, intent)
	crashInterruptedRecoveryProcess(t, repo, storePath, "merge")
	assertExactGateGitLockResidues(t, locks, lockMarker)
	assertGitReferenceTransactionLocksReleased(t, repo, intent)
	if head := recoveryGit(t, repo, "rev-parse", "HEAD"); head != parent {
		t.Fatalf("HEAD after merge restore crash = %s, want merge parent %s", head, parent)
	}
	if _, err := recoverInterruptedCommit(repo, storePath); err != nil {
		t.Fatal(err)
	}
	assertGateGitLockPathsAbsent(t, locks)
	if head := recoveryGit(t, repo, "rev-parse", "HEAD"); head != parent {
		t.Fatalf("HEAD = %s, want merge parent %s", head, parent)
	}
	if got := recoveryGit(t, repo, "write-tree"); got != mergeIndexTree {
		t.Fatalf("restored merge index tree = %s, want %s", got, mergeIndexTree)
	}
	for _, metadata := range []struct {
		name string
		want []byte
	}{
		{name: "MERGE_HEAD", want: mergeHead},
		{name: "MERGE_MODE", want: mergeMode},
		{name: "MERGE_MSG", want: mergeMessage},
	} {
		if got := readRecoveryGitMetadata(t, repo, metadata.name); string(got) != string(metadata.want) {
			t.Fatalf("restored %s = %q, want %q", metadata.name, got, metadata.want)
		}
	}
	if got, err := os.ReadFile(planPath); err != nil {
		t.Fatal(err)
	} else if string(got) != string(planBefore) {
		t.Fatalf("restored merge plan differs\ngot:\n%s\nwant:\n%s", got, planBefore)
	}
	if got, err := os.ReadFile(filepath.Join(repo, retiredPath)); err != nil {
		t.Fatal(err)
	} else if string(got) != string(recreated) {
		t.Fatalf("recreated untracked path was clobbered: got %q want %q", got, recreated)
	}
	if tracked := recoveryGit(t, repo, "ls-files", "--", retiredPath); tracked != "" {
		t.Fatalf("recreated path was staged or tracked: %q", tracked)
	}
	if status := recoveryGit(t, repo, "status", "--porcelain=v1", "--untracked-files=all", "--", retiredPath); status != "D  "+retiredPath+"\n?? "+retiredPath {
		t.Fatalf("recreated path status = %q, want staged merge deletion plus untracked recreation", status)
	}
	assertNoCommitIntent(t, repo)

	store, err := overgodb.Open(filepath.Join(repo, storePath))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	finalization, found, err := runrecord.GateFinalizationForPreparation(t.Context(), store, preparation.ID)
	if err != nil {
		t.Fatal(err)
	}
	if !found || finalization.Outcome != runrecord.OutcomeCancelled || finalization.CodeCommit != commit {
		t.Fatalf("merge recovery finalization = (%+v, %t)", finalization, found)
	}
}

func TestCapturePendingMergeRejectsAutostash(t *testing.T) {
	t.Parallel()
	repo := t.TempDir()
	runGitFixture(t, repo, "init", "-q")
	runGitFixture(t, repo, "config", "user.email", "autostash@example.invalid")
	runGitFixture(t, repo, "config", "user.name", "Autostash Test")
	runGitFixture(t, repo, "config", "core.autocrlf", "false")
	runGitFixture(t, repo, "config", "merge.autostash", "true")
	dirtyPath := filepath.Join(repo, "dirty.txt")
	if err := os.WriteFile(dirtyPath, []byte("baseline\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "add", "--", "dirty.txt")
	runGitFixture(t, repo, "commit", "-q", "-m", "baseline")
	runGitFixture(t, repo, "checkout", "-q", "-b", "autostash-side")
	if err := os.WriteFile(filepath.Join(repo, "side.txt"), []byte("side\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "add", "--", "side.txt")
	runGitFixture(t, repo, "commit", "-q", "-m", "side")
	runGitFixture(t, repo, "checkout", "-q", "master")
	wantDirty := []byte("uncommitted user state\n")
	if err := os.WriteFile(dirtyPath, wantDirty, 0o644); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "merge", "--no-ff", "--no-commit", "autostash-side")

	if _, err := capturePendingMerge(repo); err == nil || !strings.Contains(err.Error(), "MERGE_AUTOSTASH") {
		t.Fatalf("autostash merge capture error = %v", err)
	}
	autostashPath, err := gitMetadataPath(repo, "MERGE_AUTOSTASH")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(autostashPath); err != nil {
		t.Fatalf("autostash rejection removed recovery authority: %v", err)
	}
	runGitFixture(t, repo, "merge", "--abort")
	if restored, err := os.ReadFile(dirtyPath); err != nil {
		t.Fatal(err)
	} else if !bytes.Equal(restored, wantDirty) {
		t.Fatalf("git merge --abort restored %q, want %q", restored, wantDirty)
	}
}

func TestRecoverInitialWriteAheadIntentWithAdvancedPlan(t *testing.T) {
	t.Parallel()
	repo := t.TempDir()
	if err := os.MkdirAll(filepath.Join(repo, "tmp"), 0o755); err != nil {
		t.Fatal(err)
	}
	storePath := "store"
	sourcePath := "candidate.go"
	planPath := filepath.Join(repo, filepath.FromSlash(plan.Path))
	if err := os.MkdirAll(filepath.Dir(planPath), 0o755); err != nil {
		t.Fatal(err)
	}
	document := plan.Plan{
		Campaign: "initial-write-ahead", Doctrine: "test pre-commit recovery",
		Items: []plan.Item{{
			ID: "ratchet", Status: plan.StatusOpen,
			Steps: []plan.Step{{
				ID: "recover", Status: plan.StatusOpen, Verify: "go test ./cmd/gate",
			}},
		}},
	}
	if err := plan.Save(planPath, document); err != nil {
		t.Fatal(err)
	}
	planBefore, err := os.ReadFile(planPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo, sourcePath), []byte("package candidate\n\nconst value = 1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "init", "-q")
	runGitFixture(t, repo, "config", "user.email", "write-ahead@example.invalid")
	runGitFixture(t, repo, "config", "user.name", "Write Ahead Test")
	runGitFixture(t, repo, "config", "core.autocrlf", "false")
	runGitFixture(t, repo, "add", "--", plan.Path, sourcePath)
	runGitFixture(t, repo, "commit", "-q", "-m", "baseline")
	parent := recoveryGit(t, repo, "rev-parse", "HEAD")
	indexBefore, err := captureGateIndex(repo)
	if err != nil {
		t.Fatal(err)
	}

	environment, err := runrecord.NewEnvironment(runrecord.Environment{
		Host: "test", OS: "test", Arch: "test", Device: "host", Backend: "go", Driver: "none", Runtime: "go-test",
	})
	if err != nil {
		t.Fatal(err)
	}
	preparation, err := runrecord.NewGatePreparation(strings.Repeat("c", 64), environment.ID, time.Unix(300, 1))
	if err != nil {
		t.Fatal(err)
	}
	preparationCommit := publishInterruptedPreparation(t, repo, storePath, environment, preparation)
	planAfter, err := advancedPlanBytes(planBefore, "ratchet/recover")
	if err != nil {
		t.Fatal(err)
	}
	sourceAfter := []byte("package candidate\n\nconst value = 2\n")
	if err := os.WriteFile(filepath.Join(repo, sourcePath), sourceAfter, 0o644); err != nil {
		t.Fatal(err)
	}
	gate := gateContext{repo: repo, paths: []string{sourcePath, plan.Path}}
	acceptedTree, err := gate.plannedTree()
	if err != nil {
		t.Fatal(err)
	}
	indexAfter, err := buildAcceptedCompletionIndex(repo, acceptedTree, planAfter)
	if err != nil {
		t.Fatal(err)
	}
	indexRestore, err := buildGateIndexForTree(repo, indexBefore.Tree)
	if err != nil {
		t.Fatal(err)
	}
	intent := gateCommitIntent{
		Version: artifact.InitialDocumentVersion, Preparation: preparation.ID,
		PreparationCommit: preparationCommit,
		Recipe:            testutil.ArtifactID(t, artifact.KindRecipe, "pre-commit recipe"),
		CandidateManifest: testutil.ArtifactID(t, artifact.KindProfile, "pre-commit candidate"),
		Parent:            parent, HeadReference: mustCurrentHeadReference(t, repo),
		IndexTree: indexBefore.Tree, Tree: indexAfter.Tree,
		IndexBefore: indexBefore.Data, IndexAfter: indexAfter.Data, IndexRestore: indexRestore.Data,
		IndexMode: uint32(indexBefore.Mode.Perm()),
		PlanRef:   "ratchet/recover", Paths: []string{sourcePath, plan.Path},
		Plan: planBefore, AdvancedPlan: planAfter, PlanMode: 0o644,
	}
	messagePath := filepath.Join(t.TempDir(), "write-ahead-message.txt")
	if err := os.WriteFile(messagePath, []byte("write-ahead completion\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	intent.Commit, err = createGateCommit(repo, intent, messagePath)
	if err != nil {
		t.Fatal(err)
	}
	if err := writeGateCommitIntent(repo, intent); err != nil {
		t.Fatalf("write initial intent: %v", err)
	}
	if err := readJSON(repo, gateCommitIntentFile, &intent); err != nil {
		t.Fatal(err)
	}
	if err := installCompletionIndexForTest(repo, intent); err != nil {
		t.Fatalf("install write-ahead completion index: %v", err)
	}
	if err := os.WriteFile(planPath, planAfter, 0o644); err != nil {
		t.Fatal(err)
	}

	if _, err := recoverInterruptedCommit(repo, storePath); err != nil {
		t.Fatal(err)
	}
	if head := recoveryGit(t, repo, "rev-parse", "HEAD"); head != parent {
		t.Fatalf("HEAD = %s, want unchanged parent %s", head, parent)
	}
	if got, err := os.ReadFile(planPath); err != nil {
		t.Fatal(err)
	} else if string(got) != string(planBefore) {
		t.Fatalf("pre-commit plan was not restored\ngot:\n%s\nwant:\n%s", got, planBefore)
	}
	if got, err := os.ReadFile(filepath.Join(repo, sourcePath)); err != nil {
		t.Fatal(err)
	} else if string(got) != string(sourceAfter) {
		t.Fatalf("pre-commit candidate was clobbered: got %q want %q", got, sourceAfter)
	}
	if staged := recoveryGit(t, repo, "diff", "--cached", "--name-only"); staged != "" {
		t.Fatalf("pre-commit recovery staged paths: %q", staged)
	}
	assertNoCommitIntent(t, repo)
}

func TestRecoverRefAdvancedBeforePlanPublication(t *testing.T) {
	t.Parallel()
	fixture := newInterruptedCommitFixture(t)
	var intent gateCommitIntent
	if err := readJSON(fixture.repo, gateCommitIntentFile, &intent); err != nil {
		t.Fatal(err)
	}
	intent.Commit = fixture.commit
	if err := writeGateCommitIntent(fixture.repo, intent); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(
		filepath.Join(fixture.repo, filepath.FromSlash(plan.Path)), fixture.planBefore, 0o644,
	); err != nil {
		t.Fatal(err)
	}

	if _, err := recoverInterruptedCommit(fixture.repo, fixture.storePath); err != nil {
		t.Fatalf("recover exact pre-publication plan delta: %v", err)
	}
	if head := recoveryGit(t, fixture.repo, "rev-parse", "HEAD"); head != fixture.parent {
		t.Fatalf("HEAD = %s, want restored parent %s", head, fixture.parent)
	}
	if restored, err := planBytesMatch(fixture.repo, fixture.planBefore); err != nil || !restored {
		t.Fatalf("pre-publication plan restoration = (%v, %v)", restored, err)
	}
	assertNoCommitIntent(t, fixture.repo)
}

func TestRecoverRefAdvancedDuringPlanPublicationDetach(t *testing.T) {
	t.Parallel()
	fixture := newInterruptedCommitFixture(t)
	detachPlanPublicationForTest(t, fixture)

	if _, err := recoverInterruptedCommit(fixture.repo, fixture.storePath); err != nil {
		t.Fatalf("recover plan publication detach: %v", err)
	}
	if head := recoveryGit(t, fixture.repo, "rev-parse", "HEAD"); head != fixture.parent {
		t.Fatalf("HEAD = %s, want restored parent %s", head, fixture.parent)
	}
	if restored, err := planBytesMatch(fixture.repo, fixture.planBefore); err != nil || !restored {
		t.Fatalf("detached plan restoration = (%v, %v)", restored, err)
	}
	assertNoCommitIntent(t, fixture.repo)
}

func TestRecoverPlanPublicationDetachPreservesConcurrentCreation(t *testing.T) {
	fixture := newInterruptedCommitFixture(t)
	planPath := filepath.Join(fixture.repo, filepath.FromSlash(plan.Path))
	detachPlanPublicationForTest(t, fixture)
	concurrent := bytes.Replace(
		fixture.planBefore, []byte("test recovery"), []byte("concurrent recovery edit"), 1,
	)
	gatePlanRecoveryBeforeSwapHook = func(path string) {
		if err := os.WriteFile(path, concurrent, 0o644); err != nil {
			t.Fatalf("concurrent plan creation: %v", err)
		}
	}
	t.Cleanup(func() { gatePlanRecoveryBeforeSwapHook = nil })

	if _, err := recoverInterruptedCommit(fixture.repo, fixture.storePath); err == nil ||
		!strings.Contains(err.Error(), "plan moved before interrupted-state recovery completed") {
		t.Fatalf("concurrent detached-plan recovery error = %v", err)
	}
	got, err := os.ReadFile(planPath)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, concurrent) {
		t.Fatal("refused detached-plan recovery replaced concurrent bytes")
	}
	if _, err := os.Stat(filepath.Join(fixture.repo, filepath.FromSlash(gateCommitIntentFile))); err != nil {
		t.Fatalf("refused detached-plan recovery removed its durable intent: %v", err)
	}
}

func TestRecoverBareMissingPlanRefusesWithoutDetachEvidence(t *testing.T) {
	t.Parallel()
	fixture := newInterruptedCommitFixture(t)
	planPath := filepath.Join(fixture.repo, filepath.FromSlash(plan.Path))
	if err := os.Remove(planPath); err != nil {
		t.Fatal(err)
	}
	if _, err := recoverInterruptedCommit(fixture.repo, fixture.storePath); err == nil ||
		!strings.Contains(err.Error(), "absent without interrupted swap evidence") {
		t.Fatalf("bare missing-plan recovery error = %v", err)
	}
	if _, err := os.Stat(filepath.Join(fixture.repo, filepath.FromSlash(gateCommitIntentFile))); err != nil {
		t.Fatalf("refused missing-plan recovery removed its durable intent: %v", err)
	}
}

func detachPlanPublicationForTest(t *testing.T, fixture interruptedCommitFixture) {
	t.Helper()
	var intent gateCommitIntent
	if err := readJSON(fixture.repo, gateCommitIntentFile, &intent); err != nil {
		t.Fatal(err)
	}
	planPath := filepath.Join(fixture.repo, filepath.FromSlash(plan.Path))
	if err := os.WriteFile(planPath, fixture.planBefore, 0o644); err != nil {
		t.Fatal(err)
	}
	scratch, err := gatePlanScratchPath(fixture.repo, intent.Preparation)
	if err != nil {
		t.Fatal(err)
	}
	next := filepath.Join(scratch, ".atomic-swap-next-test")
	witness := filepath.Join(scratch, ".atomic-swap-witness-test")
	old := filepath.Join(scratch, ".atomic-swap-old-test")
	if err := os.WriteFile(next, fixture.planAfter, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Link(planPath, witness); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(planPath, old); err != nil {
		t.Fatal(err)
	}
}

func TestRecoverInterruptedCommitPreservesPlanChangedBeforeRestore(t *testing.T) {
	fixture := newInterruptedCommitFixture(t)
	concurrent := append([]byte(nil), fixture.planAfter...)
	concurrent = bytes.Replace(concurrent, []byte("test recovery"), []byte("concurrent recovery edit"), 1)
	gatePlanRecoveryBeforeSwapHook = func(path string) {
		if err := os.WriteFile(path, concurrent, 0o644); err != nil {
			t.Fatalf("concurrent plan edit: %v", err)
		}
	}
	t.Cleanup(func() { gatePlanRecoveryBeforeSwapHook = nil })

	if _, err := recoverInterruptedCommit(fixture.repo, fixture.storePath); err == nil ||
		!strings.Contains(err.Error(), "atomic file changed") {
		t.Fatalf("concurrent plan recovery error = %v", err)
	}
	got, err := os.ReadFile(filepath.Join(fixture.repo, filepath.FromSlash(plan.Path)))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, concurrent) {
		t.Fatalf("refused recovery replaced concurrent plan bytes")
	}
	if _, err := os.Stat(filepath.Join(fixture.repo, filepath.FromSlash(gateCommitIntentFile))); err != nil {
		t.Fatalf("refused recovery removed its durable intent: %v", err)
	}
}

func TestPostCommitTrailerMutationIsRefusedAndRecovered(t *testing.T) {
	t.Parallel()
	fixture := newInterruptedCommitFixtureWithHook(t, []byte(
		"#!/bin/sh\n"+
			"grep -v '^Overgo-Code-Manifest:' \"$1\" > \"$1.overgo-test\" || exit 1\n"+
			"mv \"$1.overgo-test\" \"$1\"\n",
	))
	var intent gateCommitIntent
	if err := readJSON(fixture.repo, gateCommitIntentFile, &intent); err != nil {
		t.Fatal(err)
	}
	intent.Commit = fixture.commit
	if err := writeGateCommitIntent(fixture.repo, intent); err != nil {
		t.Fatal(err)
	}
	if err := validateInterruptedCommitAuthority(fixture.repo, intent, nil); err == nil ||
		!strings.Contains(err.Error(), "wrong completion authority") {
		t.Fatalf("mutated completion validation = %v", err)
	}

	if _, err := recoverInterruptedCommit(fixture.repo, fixture.storePath); err != nil {
		t.Fatalf("recover commit with refused completion authority: %v", err)
	}
	if head := recoveryGit(t, fixture.repo, "rev-parse", "HEAD"); head != fixture.parent {
		t.Fatalf("HEAD = %s, want restored parent %s", head, fixture.parent)
	}
	if got, err := os.ReadFile(filepath.Join(fixture.repo, filepath.FromSlash(plan.Path))); err != nil {
		t.Fatal(err)
	} else if string(got) != string(fixture.planBefore) {
		t.Fatalf("plan was not restored after trailer refusal\ngot:\n%s\nwant:\n%s", got, fixture.planBefore)
	}
	assertNoCommitIntent(t, fixture.repo)
	store := openRecoveryStore(t, fixture)
	defer store.Close()
	if attempts, err := runrecord.AttemptsForPreparation(t.Context(), store, fixture.preparation.ID); err != nil {
		t.Fatal(err)
	} else if len(attempts) != 0 {
		t.Fatalf("mutated completion published successful attempts: %+v", attempts)
	}
	finalization, found, err := runrecord.GateFinalizationForPreparation(
		t.Context(), store, fixture.preparation.ID,
	)
	if err != nil {
		t.Fatal(err)
	}
	if !found || finalization.Outcome != runrecord.OutcomeCancelled {
		t.Fatalf("mutated completion finalization = (%+v, %t), want one cancellation", finalization, found)
	}
}

func TestPostCommitForeignAuthorityTupleIsRefused(t *testing.T) {
	t.Parallel()
	foreign := testutil.ArtifactID(t, artifact.KindProfile, "foreign completion code manifest")
	fixture := newInterruptedCommitFixtureWithHook(t, []byte(
		"#!/bin/sh\n"+
			"sed 's|^Overgo-Code-Manifest:.*$|Overgo-Code-Manifest: "+foreign.String()+"|' \"$1\" > \"$1.overgo-test\" || exit 1\n"+
			"mv \"$1.overgo-test\" \"$1\"\n",
	))
	var intent gateCommitIntent
	if err := readJSON(fixture.repo, gateCommitIntentFile, &intent); err != nil {
		t.Fatal(err)
	}
	if err := validateInterruptedCommitAuthority(fixture.repo, intent, nil); err == nil ||
		!strings.Contains(err.Error(), "expected authority tuple") {
		t.Fatalf("foreign completion tuple validation = %v", err)
	}
}

func TestInterruptedCommitRejectsDeletedBaselinePlan(t *testing.T) {
	t.Parallel()
	fixture := newInterruptedCommitFixture(t)
	var intent gateCommitIntent
	if err := readJSON(fixture.repo, gateCommitIntentFile, &intent); err != nil {
		t.Fatal(err)
	}
	intent.Commit = fixture.commit
	document, err := plan.Parse(intent.Plan)
	if err != nil {
		t.Fatal(err)
	}
	document.Items = document.Items[:1]
	intent.Plan, err = json.MarshalIndent(document, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	intent.Plan = append(intent.Plan, '\n')
	if err := validateInterruptedCommitAuthority(fixture.repo, intent, nil); err == nil ||
		!strings.Contains(err.Error(), "deleted a baseline item or step") {
		t.Fatalf("tampered pre-advance plan validation = %v", err)
	}
}

func TestInterruptedCompletionRequiresPreparationReceipt(t *testing.T) {
	t.Parallel()
	fixture := newInterruptedCommitFixture(t)
	publishSuccessfulInterruptedAttempt(t, fixture, true)
	var intent gateCommitIntent
	if err := readJSON(fixture.repo, gateCommitIntentFile, &intent); err != nil {
		t.Fatal(err)
	}
	intent.Commit = fixture.commit
	intent.PreparationCommit[0]++
	store := openRecoveryStore(t, fixture)
	defer store.Close()
	recorded, err := interruptedCompletionRecorded(t.Context(), store, intent)
	if err != nil {
		t.Fatal(err)
	}
	if recorded {
		t.Fatal("interrupted completion accepted a preparation commit other than its store introduction")
	}
}

func TestInterruptedCommitInspectionRejectsReplacementRefs(t *testing.T) {
	t.Parallel()
	fixture := newInterruptedCommitFixture(t)
	var intent gateCommitIntent
	if err := readJSON(fixture.repo, gateCommitIntentFile, &intent); err != nil {
		t.Fatal(err)
	}
	invalid := recoveryGit(
		t,
		fixture.repo,
		"commit-tree",
		intent.Tree,
		"-p",
		intent.Parent,
		"-m",
		"commit without completion authority",
	)
	runGitFixture(t, fixture.repo, "replace", invalid, fixture.commit)
	intent.Commit = invalid
	if err := validateInterruptedCommitAuthority(fixture.repo, intent, nil); err == nil ||
		!strings.Contains(err.Error(), "replacement refs") {
		t.Fatalf("replacement-hidden invalid commit validation = %v", err)
	}
}

func TestRecoverInterruptedCommitRejectsLegacyGrafts(t *testing.T) {
	t.Parallel()
	fixture := newInterruptedCommitFixture(t)
	graftsPath := recoveryGit(
		t,
		fixture.repo,
		"rev-parse",
		"--path-format=absolute",
		"--git-path",
		"info/grafts",
	)
	if err := os.MkdirAll(filepath.Dir(graftsPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(graftsPath, []byte("legacy graft state\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := recoverInterruptedCommit(fixture.repo, fixture.storePath); err == nil ||
		!strings.Contains(err.Error(), "legacy Git grafts") {
		t.Fatalf("legacy graft recovery error = %v", err)
	}
	if head := recoveryGit(t, fixture.repo, "rev-parse", "HEAD"); head != fixture.commit {
		t.Fatalf("HEAD moved to %s, want interrupted commit %s", head, fixture.commit)
	}
	if _, err := os.Stat(filepath.Join(fixture.repo, filepath.FromSlash(gateCommitIntentFile))); err != nil {
		t.Fatalf("graft rejection removed recovery intent: %v", err)
	}
}

func TestGateMergeIntentRejectsOctopusParents(t *testing.T) {
	t.Parallel()
	parent := strings.Repeat("1", 40)
	merge := gateMergeIntent{
		IndexTree: strings.Repeat("2", 40),
		Head:      []byte(strings.Repeat("3", 40) + "\n" + strings.Repeat("4", 40) + "\n"),
		Message:   []byte("merge\n"),
	}
	if err := merge.validate(parent); err == nil {
		t.Fatal("octopus merge intent was accepted")
	}
}

type interruptedCommitFixture struct {
	repo              string
	storePath         string
	sourcePath        string
	parent            string
	commit            string
	planBefore        []byte
	planAfter         []byte
	sourceAfter       []byte
	recipe            artifact.ID
	candidate         artifact.ID
	environment       runrecord.Environment
	preparation       runrecord.GateLifecycle
	preparationCommit artifact.CommitID
	manifest          automationcheck.ManifestPlan
}

func newInterruptedCommitFixture(t *testing.T) interruptedCommitFixture {
	t.Helper()
	return newInterruptedCommitFixtureWithHook(t, nil)
}

func newInterruptedCommitFixtureWithHook(t *testing.T, commitMessageHook []byte, batches ...*plan.VerificationBatch) interruptedCommitFixture {
	t.Helper()
	repo := t.TempDir()
	storePath := "store"
	sourcePath := "candidate.go"
	if err := os.MkdirAll(filepath.Join(repo, "tmp"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(repo, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	document := plan.Plan{
		Campaign: "interrupted-commit", Doctrine: "test recovery",
		Items: []plan.Item{
			{
				ID: "ratchet", Status: plan.StatusOpen,
				Steps: []plan.Step{{
					ID: "recover", Status: plan.StatusOpen, Verify: "go test ./cmd/gate",
				}},
			},
			{
				ID: "unrelated", Status: plan.StatusOpen,
				Steps: []plan.Step{{
					ID: "preserve", Status: plan.StatusOpen, Verify: "go test ./cmd/gate",
				}},
			},
		},
	}
	if len(batches) > 1 {
		t.Fatal("interrupted fixture accepts one verification batch")
	}
	if len(batches) == 1 {
		document.Items[0].Steps[0].VerificationBatch = batches[0]
	}
	planPath := filepath.Join(repo, filepath.FromSlash(plan.Path))
	if err := plan.Save(planPath, document); err != nil {
		t.Fatal(err)
	}
	planBefore, err := os.ReadFile(planPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo, sourcePath), []byte("package candidate\n\nconst value = 1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "init", "-q")
	runGitFixture(t, repo, "config", "user.email", "recovery@example.invalid")
	runGitFixture(t, repo, "config", "user.name", "Recovery Test")
	runGitFixture(t, repo, "config", "core.autocrlf", "false")
	runGitFixture(t, repo, "add", "--", plan.Path, sourcePath)
	runGitFixture(t, repo, "commit", "-q", "-m", "baseline")
	parent := recoveryGit(t, repo, "rev-parse", "HEAD")
	indexBefore, err := captureGateIndex(repo)
	if err != nil {
		t.Fatal(err)
	}

	environment, err := runrecord.NewEnvironment(runrecord.Environment{
		Host: "test", OS: "test", Arch: "test", Device: "host", Backend: "go", Driver: "none", Runtime: "go-test",
	})
	if err != nil {
		t.Fatal(err)
	}
	preparation, err := runrecord.NewGatePreparation(strings.Repeat("a", 64), environment.ID, time.Unix(100, 1))
	if err != nil {
		t.Fatal(err)
	}
	preparationCommit := publishInterruptedPreparation(t, repo, storePath, environment, preparation)

	advanced, err := plan.Advance(document, "ratchet", "recover")
	if err != nil {
		t.Fatal(err)
	}
	if err := plan.Save(planPath, advanced); err != nil {
		t.Fatal(err)
	}
	planAfter, err := os.ReadFile(planPath)
	if err != nil {
		t.Fatal(err)
	}
	sourceAfter := []byte("package candidate\n\nconst value = 2\n")
	if err := os.WriteFile(filepath.Join(repo, sourcePath), sourceAfter, 0o644); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "add", "--", plan.Path, sourcePath)
	indexAfter, err := captureGateIndex(repo)
	if err != nil {
		t.Fatal(err)
	}
	indexRestore, err := buildGateIndexForTree(repo, indexBefore.Tree)
	if err != nil {
		t.Fatal(err)
	}

	codeManifest := testutil.ArtifactID(t, artifact.KindProfile, "completion code manifest")
	baseManifest := testutil.ArtifactID(t, artifact.KindProfile, "completion base manifest")
	var invocations []automationcheck.Invocation
	if batch := document.Items[0].Steps[0].VerificationBatch; batch != nil {
		for _, checkpoint := range batch.Checkpoints {
			invocations = append(invocations, automationcheck.Invocation{
				ID:    testutil.ArtifactID(t, artifact.KindRecipe, checkpoint.GateCheckName()),
				Check: automationcheck.Descriptor{Name: checkpoint.GateCheckName(), Phase: runrecord.PhaseTest, Always: true},
			})
		}
	}
	invocations = append(invocations, automationcheck.Invocation{
		ID:    testutil.ArtifactID(t, artifact.KindRecipe, "completion invocation"),
		Check: automationcheck.Descriptor{Name: "commit", Phase: runrecord.PhasePackage},
	})
	manifest, err := automationcheck.BindManifestPlan(
		baseManifest, codeManifest, strings.Repeat("b", 64), preparation.TreeKey,
		automationcheck.Surface{Identity: "interrupted completion fixture"}, automationcheck.Impact{},
		invocations,
	)
	if err != nil {
		t.Fatal(err)
	}
	recipe := manifest.ID
	message, err := plan.CompletionCommitMessageWithMergeAuthority(
		[]byte("Complete interrupted gate"), document, "ratchet", "recover",
		recipe, codeManifest, preparation.ID, preparationCommit, plan.MergeProjectionSemanticUnion, artifact.ID{},
	)
	if err != nil {
		t.Fatal(err)
	}
	messagePath := filepath.Join(t.TempDir(), "message.txt")
	if err := os.WriteFile(messagePath, message, 0o600); err != nil {
		t.Fatal(err)
	}
	intent := gateCommitIntent{
		Version: artifact.InitialDocumentVersion, Preparation: preparation.ID,
		PreparationCommit: preparationCommit, Recipe: recipe,
		CandidateManifest: codeManifest,
		Parent:            parent, HeadReference: mustCurrentHeadReference(t, repo),
		IndexTree: indexBefore.Tree, Tree: indexAfter.Tree,
		IndexBefore: indexBefore.Data, IndexAfter: indexAfter.Data, IndexRestore: indexRestore.Data,
		IndexMode: uint32(indexBefore.Mode.Perm()),
		PlanRef:   "ratchet/recover", Paths: []string{sourcePath, plan.Path},
		Plan: planBefore, AdvancedPlan: planAfter, PlanMode: 0o644,
	}
	if len(commitMessageHook) != 0 {
		hookPath, err := gitMetadataPath(repo, "hooks/commit-msg")
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(hookPath, commitMessageHook, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	runGitFixture(t, repo, "commit", "-q", "-F", messagePath)
	commit := recoveryGit(t, repo, "rev-parse", "HEAD")
	postCommitIndex, err := captureGateIndex(repo)
	if err != nil {
		t.Fatal(err)
	}
	intent.Commit, intent.Tree, intent.IndexAfter = commit, postCommitIndex.Tree, postCommitIndex.Data
	if err := writeGateCommitIntent(repo, intent); err != nil {
		t.Fatal(err)
	}

	return interruptedCommitFixture{
		repo: repo, storePath: storePath, sourcePath: sourcePath, parent: parent, commit: commit,
		planBefore: planBefore, planAfter: planAfter, sourceAfter: sourceAfter,
		recipe: recipe, candidate: codeManifest, environment: environment, preparation: preparation,
		preparationCommit: preparationCommit, manifest: manifest,
	}
}

func publishInterruptedPreparation(
	t *testing.T,
	repo, storePath string,
	environment runrecord.Environment,
	preparation runrecord.GateLifecycle,
) artifact.CommitID {
	return publishInterruptedPreparationWithKey(
		t, repo, storePath, environment, preparation, "fixture/interrupted/preparation",
	)
}

func publishInterruptedPreparationWithKey(
	t *testing.T,
	repo, storePath string,
	environment runrecord.Environment,
	preparation runrecord.GateLifecycle,
	key string,
) artifact.CommitID {
	t.Helper()
	environmentContent, err := environment.Content()
	if err != nil {
		t.Fatal(err)
	}
	preparationContent, err := preparation.Content()
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(filepath.Join(repo, storePath))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	binding, err := gatePreparationAlias(t.Context(), store, preparation.ID)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := artifact.NewDocumentBatch(
		key,
		[]artifact.Content{environmentContent, preparationContent}, preparation.Lineage(),
		[]artifact.AliasBinding{binding},
	)
	if err != nil {
		t.Fatal(err)
	}
	commit, err := store.Commit(t.Context(), batch)
	if err != nil {
		t.Fatal(err)
	}
	introduction, found, err := store.ArtifactIntroduction(t.Context(), preparation.ID)
	if err != nil {
		t.Fatal(err)
	}
	if !found || introduction.Commit != commit {
		t.Fatalf("preparation introduction = %+v, found %v, want commit %s", introduction, found, commit)
	}
	return introduction.Commit
}

func publishSuccessfulInterruptedAttempt(
	t *testing.T,
	fixture interruptedCommitFixture,
	includeAttempt bool,
) runrecord.AttemptRecord {
	t.Helper()
	attempt, batch := successfulInterruptedAttemptBatch(t, fixture, includeAttempt, "fixture/interrupted/success")
	store := openRecoveryStore(t, fixture)
	defer store.Close()
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	return attempt
}

func successfulInterruptedAttemptBatch(
	t *testing.T,
	fixture interruptedCommitFixture,
	includeAttempt bool,
	key string,
) (runrecord.AttemptRecord, artifact.Batch) {
	return successfulAttemptBatchForReference(
		t, fixture, "ratchet", "recover", "go test ./cmd/gate", includeAttempt, key,
	)
}

func successfulAttemptBatchForReference(
	t *testing.T,
	fixture interruptedCommitFixture,
	item, step, verify string,
	includeAttempt bool,
	key string,
	members ...runrecord.GateStep,
) (runrecord.AttemptRecord, artifact.Batch) {
	t.Helper()
	acceptance, err := runrecord.FormatCompletionAcceptanceEvidence(
		testevidence.VerifyPolicyV1, item+"/"+step, verify,
	)
	if err != nil {
		t.Fatal(err)
	}
	record, err := runrecord.NewGateRecord(
		fixture.recipe, fixture.environment.ID, fixture.commit, runrecord.OutcomeSucceeded, "", 2,
		append(slices.Clone(members), []runrecord.GateStep{
			{Name: "acceptance", Phase: runrecord.PhaseTest, Outcome: runrecord.StepSucceeded, DurationNS: 1, Evidence: acceptance},
			{Name: "commit", Phase: runrecord.PhasePackage, Outcome: runrecord.StepSucceeded, DurationNS: 1},
		}...),
	)
	if err != nil {
		t.Fatal(err)
	}
	finalization, err := runrecord.NewGateFinalization(
		fixture.preparation, fixture.commit, record.Result.ID, runrecord.OutcomeSucceeded,
	)
	if err != nil {
		t.Fatal(err)
	}
	attempt, err := runrecord.NewAttemptRecord(runrecord.AttemptRecord{
		PlanItem: item, PlanStep: step, Result: record.Result.ID, Recipe: fixture.recipe,
		CodeCommit: fixture.commit, Outcome: runrecord.OutcomeSucceeded, WallNS: 2,
		CandidateManifest: fixture.candidate,
	})
	if err != nil {
		t.Fatal(err)
	}
	batch, err := record.Batch(key)
	if err != nil {
		t.Fatal(err)
	}
	finalizationContent, err := finalization.Content()
	if err != nil {
		t.Fatal(err)
	}
	batch.Artifacts = append(batch.Artifacts,
		artifact.Descriptor{ID: fixture.recipe}, artifact.Descriptor{ID: fixture.candidate},
	)
	batch.Contents = append(batch.Contents, finalizationContent)
	batch.Lineage = append(batch.Lineage, finalization.Lineage()...)
	appendGateFinalizationAlias(&batch, fixture.preparation.ID, finalization.ID)
	if includeAttempt {
		attemptContent, err := attempt.Content()
		if err != nil {
			t.Fatal(err)
		}
		batch.Contents = append(batch.Contents, attemptContent)
		batch.Lineage = append(batch.Lineage, attempt.Lineage()...)
		analysis, err := automationcheck.NewManifestAnalysis(
			codemanifest.Delta{Base: fixture.manifest.BaseManifest, Candidate: fixture.manifest.CandidateManifest},
			codemanifest.Impact{
				Base: fixture.manifest.BaseManifest.String(), Candidate: fixture.manifest.CandidateManifest.String(),
			},
			fixture.manifest, automationcheck.SelectionMetrics{},
			automationcheck.MeasureManifest(len(fixture.manifest.Invocations), len(fixture.manifest.Invocations), 0, 0, 1, 0, 0, 0),
		)
		if err != nil {
			t.Fatal(err)
		}
		analysisContent, err := analysis.Content()
		if err != nil {
			t.Fatal(err)
		}
		batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: fixture.manifest.BaseManifest})
		batch.Contents = append(batch.Contents, analysisContent)
		batch.Lineage = append(batch.Lineage, artifact.Lineage{
			Child: record.Result.ID, Parent: analysis.ID, Relation: artifact.RelationDependsOn,
		})
	}
	return attempt, batch
}

func openRecoveryStore(t *testing.T, fixture interruptedCommitFixture) *overgodb.Store {
	t.Helper()
	store, err := overgodb.Open(filepath.Join(fixture.repo, fixture.storePath))
	if err != nil {
		t.Fatal(err)
	}
	return store
}

func recoveryGit(t *testing.T, repo string, arguments ...string) string {
	t.Helper()
	output, err := command(repo, "git", arguments...)
	if err != nil {
		t.Fatalf("git %v: %v", arguments, err)
	}
	return strings.TrimSpace(output)
}

func mustCurrentHeadReference(t *testing.T, repo string) string {
	t.Helper()
	reference, err := currentHeadReference(repo)
	if err != nil {
		t.Fatalf("current HEAD reference: %v", err)
	}
	return reference
}

func prepareRecoveryMergeRaceBranch(t *testing.T, fixture interruptedCommitFixture) string {
	t.Helper()
	baseBranch := recoveryGit(t, fixture.repo, "rev-parse", "--abbrev-ref", "HEAD")
	side := "recovery-race-side"
	runGitFixture(t, fixture.repo, "checkout", "-q", "-b", side)
	if err := os.WriteFile(filepath.Join(fixture.repo, "recovery-race.txt"), []byte("race\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, fixture.repo, "add", "--", "recovery-race.txt")
	runGitFixture(t, fixture.repo, "commit", "-q", "-m", "recovery race side")
	runGitFixture(t, fixture.repo, "checkout", "-q", baseBranch)
	if head := recoveryGit(t, fixture.repo, "rev-parse", "HEAD"); head != fixture.commit {
		t.Fatalf("race fixture restored HEAD = %s, want %s", head, fixture.commit)
	}
	return side
}

func assertNoCommitIntent(t *testing.T, repo string) {
	t.Helper()
	_, err := os.Stat(filepath.Join(repo, filepath.FromSlash(gateCommitIntentFile)))
	if !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("interrupted commit intent remains: %v", err)
	}
}

func readRecoveryGitMetadata(t *testing.T, repo, name string) []byte {
	t.Helper()
	path, err := gitMetadataPath(repo, name)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return data
}
