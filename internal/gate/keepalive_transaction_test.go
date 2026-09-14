package gate

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/plan"
)

func TestIntentKeepaliveSurvivesGCForStagedOnlyRollback(t *testing.T) {
	repo := t.TempDir()
	runGitFixture(t, repo, "init", "-q")
	runGitFixture(t, repo, "config", "user.email", "keepalive@example.invalid")
	runGitFixture(t, repo, "config", "user.name", "Keepalive Test")
	runGitFixture(t, repo, "config", "core.autocrlf", "false")
	if err := os.MkdirAll(filepath.Join(repo, "tmp"), 0o755); err != nil {
		t.Fatal(err)
	}
	candidatePath := filepath.Join(repo, "candidate.txt")
	if err := os.WriteFile(candidatePath, []byte("baseline\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "add", "--", "candidate.txt")
	runGitFixture(t, repo, "commit", "-q", "-m", "baseline")

	stagedOnly := []byte("staged only before crash\n")
	if err := os.WriteFile(candidatePath, stagedOnly, 0o644); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "add", "--", "candidate.txt")
	before, err := captureGateIndex(repo)
	if err != nil {
		t.Fatal(err)
	}
	stagedBlob := recoveryGit(t, repo, "rev-parse", ":candidate.txt")

	if err := os.WriteFile(candidatePath, []byte("accepted completion\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	tree, err := (&gateContext{repo: repo, paths: []string{"candidate.txt"}}).plannedTree()
	if err != nil {
		t.Fatal(err)
	}
	after, err := buildGateIndexForTree(repo, tree)
	if err != nil {
		t.Fatal(err)
	}
	acceptedBlob := recoveryGit(t, repo, "rev-parse", after.Tree+":candidate.txt")

	intent := newKeepaliveTransactionIntent(t, repo, before, after, nil)
	boundBeforeIntent := false
	gateKeepaliveBoundHook = func(repository string, bound gateCommitIntent) {
		boundBeforeIntent = true
		if _, err := os.Stat(filepath.Join(repository, filepath.FromSlash(gateCommitIntentFile))); !os.IsNotExist(err) {
			t.Fatalf("keepalive boundary already published intent: %v", err)
		}
		runGitFixture(t, repository, "reflog", "expire", "--expire=now", "--all")
		runGitFixture(t, repository, "gc", "--prune=now")
		if _, err := command(repository, "git", "cat-file", "-e", acceptedBlob+"^{blob}"); err != nil {
			t.Fatalf("accepted-only blob was pruned before intent publication: %v", err)
		}
		if _, err := command(repository, "git", "cat-file", "-e", bound.Commit+"^{commit}"); err != nil {
			t.Fatalf("completion commit was pruned before intent publication: %v", err)
		}
	}
	t.Cleanup(func() { gateKeepaliveBoundHook = nil })
	if err := writeGateCommitIntent(repo, intent); err != nil {
		t.Fatal(err)
	}
	gateKeepaliveBoundHook = nil
	if !boundBeforeIntent {
		t.Fatal("keepalive was not bound before durable intent publication")
	}
	if err := readJSON(repo, gateCommitIntentFile, &intent); err != nil {
		t.Fatal(err)
	}
	if _, err := command(repo, "git", "rev-list", "--all"); err != nil {
		t.Fatalf("keepalive is not transparent to Git revision traversal: %v", err)
	}
	runGitFixture(t, repo, "reflog", "expire", "--expire=now", "--all")
	runGitFixture(t, repo, "gc", "--prune=now")
	if _, err := command(repo, "git", "cat-file", "-e", acceptedBlob+"^{blob}"); err != nil {
		t.Fatalf("accepted-only completion blob was pruned before index install: %v", err)
	}
	if _, err := command(repo, "git", "cat-file", "-e", intent.Commit+"^{commit}"); err != nil {
		t.Fatalf("unreferenced completion commit was pruned before branch CAS: %v", err)
	}
	if err := installCompletionIndexForTest(repo, intent); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "reflog", "expire", "--expire=now", "--all")
	runGitFixture(t, repo, "gc", "--prune=now")
	if _, err := command(repo, "git", "cat-file", "-e", stagedBlob+"^{blob}"); err != nil {
		t.Fatalf("staged-only rollback blob was pruned: %v", err)
	}

	if err := restoreCapturedIndex(repo, intent); err != nil {
		t.Fatalf("restore staged-only index after prune: %v", err)
	}
	if got := recoveryGit(t, repo, "rev-parse", ":candidate.txt"); got != stagedBlob {
		t.Fatalf("restored staged blob = %s, want %s", got, stagedBlob)
	}
	if got, err := command(repo, "git", "show", ":candidate.txt"); err != nil {
		t.Fatal(err)
	} else if !bytes.Equal([]byte(got), stagedOnly) {
		t.Fatalf("restored staged-only bytes = %q, want %q", got, stagedOnly)
	}
	assertResolvedIntentKeepalive(t, repo, intent)
}

func TestIntentKeepaliveSurvivesGCForAutoMergeRollback(t *testing.T) {
	t.Parallel()
	repo, transaction := newRealConflictMergeTransactionFixture(t)
	if err := os.MkdirAll(filepath.Join(repo, "tmp"), 0o755); err != nil {
		t.Fatal(err)
	}
	index := gateIndexSnapshot{
		Tree: transaction.IndexTree, Data: bytes.Clone(transaction.IndexBefore),
		Mode: os.FileMode(transaction.IndexMode),
	}
	intent := newKeepaliveTransactionIntent(t, repo, index, index, transaction.Merge)
	if err := writeGateCommitIntent(repo, intent); err != nil {
		t.Fatal(err)
	}
	if err := readJSON(repo, gateCommitIntentFile, &intent); err != nil {
		t.Fatal(err)
	}
	autoMerge := strings.TrimSpace(string(intent.Merge.AutoMerge))
	if err := clearCommittedMergeState(repo, intent); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "reflog", "expire", "--expire=now", "--all")
	runGitFixture(t, repo, "gc", "--prune=now")
	if _, err := command(repo, "git", "cat-file", "-e", autoMerge+"^{tree}"); err != nil {
		t.Fatalf("AUTO_MERGE rollback tree was pruned: %v", err)
	}

	if err := restoreInterruptedMergeForTest(repo, intent); err != nil {
		t.Fatalf("restore AUTO_MERGE after prune: %v", err)
	}
	if got := recoveryGit(t, repo, "rev-parse", "--verify", "AUTO_MERGE"); got != autoMerge {
		t.Fatalf("restored AUTO_MERGE = %s, want %s", got, autoMerge)
	}
	assertResolvedIntentKeepalive(t, repo, intent)
}

func TestIntentKeepaliveRefIsLockedAcrossIndexMutation(t *testing.T) {
	repo, before, after := newIndexCASFixture(t)
	if err := os.MkdirAll(filepath.Join(repo, "tmp"), 0o755); err != nil {
		t.Fatal(err)
	}
	intent := newKeepaliveTransactionIntent(t, repo, before, after, nil)
	if err := writeGateCommitIntent(repo, intent); err != nil {
		t.Fatal(err)
	}
	if err := readJSON(repo, gateCommitIntentFile, &intent); err != nil {
		t.Fatal(err)
	}
	var competingErr error
	var competingOutput string
	gateIndexCASLockedHook = func(repository string) {
		competingOutput, competingErr = command(
			repository, "git", "update-ref", "-d", intent.KeepaliveRef,
		)
	}
	t.Cleanup(func() { gateIndexCASLockedHook = nil })

	if err := installCompletionIndexForTest(repo, intent); err != nil {
		t.Fatal(err)
	}
	if competingErr == nil || !strings.Contains(competingOutput, ".lock") {
		t.Fatalf("keepalive writer under transaction lock = (%v, %q)", competingErr, competingOutput)
	}
	if err := requireGateIntentKeepalive(repo, intent); err != nil {
		t.Fatalf("locked keepalive moved: %v", err)
	}
	if err := restoreCapturedIndex(repo, intent); err != nil {
		t.Fatal(err)
	}
	assertResolvedIntentKeepalive(t, repo, intent)
}

func TestPreparedCompletionBlocksBranchAndHeadContendersAtIndexSeam(t *testing.T) {
	repo, before, after := newIndexCASFixture(t)
	if err := os.MkdirAll(filepath.Join(repo, "tmp"), 0o755); err != nil {
		t.Fatal(err)
	}
	intent := newKeepaliveTransactionIntent(t, repo, before, after, nil)
	if err := writeGateCommitIntent(repo, intent); err != nil {
		t.Fatal(err)
	}
	if err := readJSON(repo, gateCommitIntentFile, &intent); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "branch", "escape", intent.Parent)
	competingMessage := filepath.Join(t.TempDir(), "competing.txt")
	if err := os.WriteFile(competingMessage, []byte("competing commit\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	competing, err := createGateCommit(repo, gateCommitIntent{
		Parent: intent.Parent, Tree: intent.IndexTree,
	}, competingMessage)
	if err != nil {
		t.Fatal(err)
	}
	var branchOutput, headOutput string
	var branchErr, headErr error
	gateCommitIndexInstalledHook = func(repository string, prepared gateCommitIntent) {
		branchOutput, branchErr = command(
			repository, "git", "update-ref", prepared.HeadReference, competing, prepared.Parent,
		)
		headOutput, headErr = command(
			repository, "git", "symbolic-ref", "HEAD", "refs/heads/escape",
		)
	}
	t.Cleanup(func() { gateCommitIndexInstalledHook = nil })

	if err := installCompletionIndexAndAdvanceGateReference(repo, intent); err != nil {
		t.Fatal(err)
	}
	gateCommitIndexInstalledHook = nil
	if branchErr == nil || !strings.Contains(branchOutput, ".lock") {
		t.Fatalf("branch contender under prepared transaction = (%v, %q)", branchErr, branchOutput)
	}
	if headErr == nil || !strings.Contains(headOutput, "HEAD.lock") {
		t.Fatalf("HEAD contender under prepared transaction = (%v, %q)", headErr, headOutput)
	}
	if !exactGateHead(repo, intent.HeadReference, intent.Commit) {
		t.Fatal("prepared completion did not publish the exact captured branch and HEAD")
	}
	if index, err := captureGateIndex(repo); err != nil {
		t.Fatal(err)
	} else if !bytes.Equal(index.Data, intent.IndexAfter) {
		t.Fatal("prepared completion did not retain its exact accepted index")
	}
	assertResolvedIntentKeepalive(t, repo, intent)
}

func TestPreparedRecoveryBlocksBranchAndHeadContendersUntilRollbackCommit(t *testing.T) {
	fixture := newInterruptedCommitFixture(t)
	var intent gateCommitIntent
	if err := readJSON(fixture.repo, gateCommitIntentFile, &intent); err != nil {
		t.Fatal(err)
	}
	detachPlanPublicationForTest(t, fixture)
	runGitFixture(t, fixture.repo, "branch", "escape", intent.Parent)
	competingMessage := filepath.Join(t.TempDir(), "competing.txt")
	if err := os.WriteFile(competingMessage, []byte("competing recovery commit\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	competing, err := createGateCommit(fixture.repo, gateCommitIntent{
		Parent: intent.Commit, Tree: intent.Tree,
	}, competingMessage)
	if err != nil {
		t.Fatal(err)
	}
	var branchOutput, headOutput string
	var branchErr, headErr error
	gateRecoveryAfterStepHook = func(repository, step string) {
		if step != "admission" {
			return
		}
		if _, err := os.Stat(filepath.Join(repository, filepath.FromSlash(plan.Path))); !os.IsNotExist(err) {
			t.Fatalf("plan publication was resolved before branch authority was locked: %v", err)
		}
		branchOutput, branchErr = command(
			repository, "git", "update-ref", intent.HeadReference, competing, intent.Commit,
		)
		headOutput, headErr = command(
			repository, "git", "symbolic-ref", "HEAD", "refs/heads/escape",
		)
	}
	t.Cleanup(func() { gateRecoveryAfterStepHook = nil })

	if _, err := recoverInterruptedCommit(fixture.repo, fixture.storePath); err != nil {
		t.Fatal(err)
	}
	gateRecoveryAfterStepHook = nil
	if branchErr == nil || !strings.Contains(branchOutput, ".lock") {
		t.Fatalf("recovery branch contender = (%v, %q)", branchErr, branchOutput)
	}
	if headErr == nil || !strings.Contains(headOutput, "HEAD.lock") {
		t.Fatalf("recovery HEAD contender = (%v, %q)", headErr, headOutput)
	}
	if !exactGateHead(fixture.repo, intent.HeadReference, intent.Parent) {
		t.Fatal("prepared recovery did not publish the exact restored parent and HEAD")
	}
}

func newKeepaliveTransactionIntent(
	t *testing.T,
	repo string,
	before, after gateIndexSnapshot,
	merge *gateMergeIntent,
) gateCommitIntent {
	t.Helper()
	document := plan.Plan{
		Campaign: "keepalive-transaction", Doctrine: "keep exact rollback objects reachable",
		Items: []plan.Item{
			{ID: "rollback", Status: plan.StatusOpen, Steps: []plan.Step{{
				ID: "recover", Status: plan.StatusOpen, Verify: "go test ./cmd/gate",
			}}},
			{ID: "survivor", Status: plan.StatusOpen, Steps: []plan.Step{{
				ID: "remain", Status: plan.StatusOpen, Verify: "go test ./cmd/gate",
			}}},
		},
	}
	planBeforePath := filepath.Join(t.TempDir(), "before.json")
	if err := plan.Save(planBeforePath, document); err != nil {
		t.Fatal(err)
	}
	planBefore, err := os.ReadFile(planBeforePath)
	if err != nil {
		t.Fatal(err)
	}
	advanced, err := plan.Advance(document, "rollback", "recover")
	if err != nil {
		t.Fatal(err)
	}
	planAfterPath := filepath.Join(t.TempDir(), "after.json")
	if err := plan.Save(planAfterPath, advanced); err != nil {
		t.Fatal(err)
	}
	planAfter, err := os.ReadFile(planAfterPath)
	if err != nil {
		t.Fatal(err)
	}
	restore, err := buildGateIndexForTree(repo, before.Tree)
	if err != nil {
		t.Fatal(err)
	}
	identify := func(kind artifact.Kind, seed string) artifact.ID {
		id, err := artifact.IdentifyBytes(kind, []byte(seed+"/"+t.Name()))
		if err != nil {
			t.Fatal(err)
		}
		return id
	}
	var preparationCommit artifact.CommitID
	preparationCommit[0] = 1
	intent := gateCommitIntent{
		Version:     artifact.InitialDocumentVersion,
		Preparation: identify(artifact.KindEvidence, "preparation"), PreparationCommit: preparationCommit,
		Recipe:            identify(artifact.KindRecipe, "recipe"),
		CandidateManifest: identify(artifact.KindProfile, "candidate"),
		Parent:            recoveryGit(t, repo, "rev-parse", "HEAD"), HeadReference: mustCurrentHeadReference(t, repo),
		Merge: merge, IndexTree: before.Tree, Tree: after.Tree,
		IndexBefore: bytes.Clone(before.Data), IndexAfter: bytes.Clone(after.Data),
		IndexRestore: bytes.Clone(restore.Data), IndexMode: uint32(before.Mode.Perm()),
		PlanRef: "rollback/recover", Paths: []string{plan.Path},
		Plan: planBefore, AdvancedPlan: planAfter, PlanMode: 0o644,
	}
	messagePath := filepath.Join(t.TempDir(), "completion-message.txt")
	if err := os.WriteFile(messagePath, []byte("keepalive completion fixture\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	intent.Commit, err = createGateCommit(repo, intent, messagePath)
	if err != nil {
		t.Fatal(err)
	}
	return intent
}

func assertResolvedIntentKeepalive(t *testing.T, repo string, intent gateCommitIntent) {
	t.Helper()
	if err := removeGateCommitIntent(repo); err != nil {
		t.Fatal(err)
	}
	if _, found, err := gateIntentKeepaliveRefValue(repo, intent.KeepaliveRef); err != nil {
		t.Fatal(err)
	} else if found {
		t.Fatalf("resolved keepalive ref %s remains", intent.KeepaliveRef)
	}
}

func bindGateIntentKeepaliveForTest(t *testing.T, repo string, intent gateCommitIntent) gateCommitIntent {
	t.Helper()
	if !intent.Preparation.Valid() {
		var err error
		intent.Preparation, err = artifact.IdentifyBytes(
			artifact.KindEvidence, []byte("keepalive-fixture/"+t.Name()),
		)
		if err != nil {
			t.Fatal(err)
		}
	}
	if !validGitObjectID(intent.Commit) {
		intent.Commit = recoveryGit(t, repo, "rev-parse", "HEAD")
	}
	if err := initializeGateIntentKeepalive(repo, &intent); err != nil {
		t.Fatal(err)
	}
	if err := ensureGateIntentKeepalive(repo, intent); err != nil {
		t.Fatal(err)
	}
	return intent
}
