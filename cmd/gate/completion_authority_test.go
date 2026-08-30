package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/codemanifest"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestCompletedPlanIdentityCannotBeReused pins the gate-side trailer writer:
// an operator cannot smuggle a second completion identity into the message
// that the shared authority reader will later consume.
func TestCompletedPlanIdentityCannotBeReused(t *testing.T) {
	repository := t.TempDir()
	if err := os.Mkdir(filepath.Join(repository, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	document := plan.Plan{Campaign: "gate", Doctrine: "fixture", Items: []plan.Item{{
		ID: "item", Status: plan.StatusOpen, Steps: []plan.Step{{
			ID: "step", Status: plan.StatusOpen, Verify: "go test ./...",
		}},
	}}}
	if err := plan.Save(filepath.Join(repository, filepath.FromSlash(plan.Path)), document); err != nil {
		t.Fatal(err)
	}
	messagePath := filepath.Join(repository, "message.txt")
	if err := os.WriteFile(messagePath, []byte("subject\n\nOvergo-Plan-Item: reused\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	gate := gateContext{
		repo: repository, planRef: "item/step", messageFile: messagePath,
		manifestPlan: &automationcheck.ManifestPlan{
			ID: testutil.ArtifactID(t, artifact.KindRecipe, "completion-plan"),
		},
		candidateManifest: &codemanifest.Manifest{
			ID: testutil.ArtifactID(t, artifact.KindProfile, "completion-code"),
		},
	}
	if _, err := gate.completionMessageFile(document); err == nil {
		t.Fatal("operator-supplied completion identity was accepted")
	}
}

func TestAcceptanceEvidenceRejectsPlanMutationBeforeCommit(t *testing.T) {
	repository := t.TempDir()
	if err := os.Mkdir(filepath.Join(repository, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	document := plan.Plan{Campaign: "gate", Doctrine: "fixture", Items: []plan.Item{{
		ID: "item", Status: plan.StatusOpen, Steps: []plan.Step{{
			ID: "step", Status: plan.StatusOpen, Verify: "go test ./...",
		}},
	}}}
	if err := plan.Save(filepath.Join(repository, filepath.FromSlash(plan.Path)), document); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repository, "init", "-q")
	runGitFixture(t, repository, "config", "user.email", "authority@example.invalid")
	runGitFixture(t, repository, "config", "user.name", "Authority Test")
	runGitFixture(t, repository, "add", "--", plan.Path)
	runGitFixture(t, repository, "commit", "-q", "-m", "baseline")
	store, err := overgodb.Open(filepath.Join(repository, gateStorePath))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	authority, err := plan.ResolveCompletionAuthority(context.Background(), repository, "HEAD", document, store)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := completionAcceptanceEvidenceForPlan(document, "item/step", authority); err != nil {
		t.Fatal(err)
	}
	mutated := document
	mutated.Items = append([]plan.Item(nil), document.Items...)
	mutated.Items[0].Steps = append([]plan.Step(nil), document.Items[0].Steps...)
	mutated.Items[0].Steps[0].Verify = "go test ./changed"
	if _, err := completionAcceptanceEvidenceForPlan(mutated, "item/step", authority); err == nil {
		t.Fatal("plan mutation retained acceptance authority")
	}
}

func TestCompletionAuthorityRejectsAlternateGateStore(t *testing.T) {
	if err := requireCanonicalGateStore("alternate-store", true); err == nil {
		t.Fatal("alternate completion-authority store was accepted")
	}
	if err := requireCanonicalGateStore(gateStorePath, true); err != nil {
		t.Fatalf("canonical store rejected: %v", err)
	}
	if err := requireCanonicalGateStore("alternate-store", false); err != nil {
		t.Fatalf("read-only mode rejected injected store: %v", err)
	}
}

func TestMutatingGateModeCannotHideBehindReadOnlyMode(t *testing.T) {
	if err := requireExclusiveGateMode(false, false, true, false, false, true, false); err == nil {
		t.Fatal("recover-interrupted combined with inspect-plan was accepted")
	}
	if err := requireExclusiveGateMode(true, false, false, false, true, false, false); err == nil {
		t.Fatal("reconcile combined with watchdog was accepted")
	}
	if err := requireExclusiveGateMode(false, false, false, false, false, true, false); err != nil {
		t.Fatalf("single read-only mode rejected: %v", err)
	}
}

func TestProspectiveGateCompletionRejectsUnrelatedPlanDeletion(t *testing.T) {
	repository := t.TempDir()
	if err := os.Mkdir(filepath.Join(repository, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	baseline := plan.Plan{Campaign: "gate", Doctrine: "fixture", Items: []plan.Item{
		{ID: "complete", Status: plan.StatusOpen, Steps: []plan.Step{{
			ID: "do", Status: plan.StatusOpen, Verify: "go test ./...",
		}}},
		{ID: "unrelated", Status: plan.StatusOpen, Steps: []plan.Step{{
			ID: "keep", Status: plan.StatusOpen, Verify: "go test ./...",
		}}},
	}}
	if err := plan.Save(filepath.Join(repository, filepath.FromSlash(plan.Path)), baseline); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repository, "init", "-q")
	runGitFixture(t, repository, "config", "user.email", "transition@example.invalid")
	runGitFixture(t, repository, "config", "user.name", "Transition Test")
	runGitFixture(t, repository, "add", "--", plan.Path)
	runGitFixture(t, repository, "commit", "-q", "-m", "baseline")
	parent := recoveryGit(t, repository, "rev-parse", "HEAD")
	preAdvance := baseline
	preAdvance.Items = append([]plan.Item(nil), baseline.Items[:1]...)
	child, err := plan.Advance(preAdvance, "complete", "do")
	if err != nil {
		t.Fatal(err)
	}
	recipe := testutil.ArtifactID(t, artifact.KindRecipe, "prospective-recipe")
	profile := testutil.ArtifactID(t, artifact.KindProfile, "prospective-profile")
	preparation := testutil.ArtifactID(t, artifact.KindEvidence, "prospective-preparation")
	message, err := plan.CompletionCommitMessage(
		[]byte("prospective completion"), preAdvance, "complete", "do", recipe, profile, preparation,
		artifact.CommitID{1},
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := verifyProspectiveGateCompletion(
		repository, gateCommitIntent{Parent: parent}, preAdvance, child, string(message), nil,
	); err == nil {
		t.Fatal("gate accepted completion after unrelated plan deletion")
	}
}

func TestProspectiveGateCompletionRejectsAmbiguousMergeBase(t *testing.T) {
	repository := t.TempDir()
	if err := os.Mkdir(filepath.Join(repository, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	document := plan.Plan{Campaign: "gate", Doctrine: "fixture", Items: []plan.Item{{
		ID: "complete", Status: plan.StatusOpen, Steps: []plan.Step{{
			ID: "do", Status: plan.StatusOpen, Verify: "go test ./...",
		}},
	}}}
	if err := plan.Save(filepath.Join(repository, filepath.FromSlash(plan.Path)), document); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repository, "init", "-q")
	runGitFixture(t, repository, "config", "user.email", "transition@example.invalid")
	runGitFixture(t, repository, "config", "user.name", "Transition Test")
	runGitFixture(t, repository, "add", "--", plan.Path)
	runGitFixture(t, repository, "commit", "-q", "-m", "root")
	root := recoveryGit(t, repository, "rev-parse", "HEAD")
	tree := recoveryGit(t, repository, "rev-parse", "HEAD^{tree}")
	commitTree := func(message string, parents ...string) string {
		t.Helper()
		arguments := []string{"commit-tree", tree}
		for _, parent := range parents {
			arguments = append(arguments, "-p", parent)
		}
		arguments = append(arguments, "-m", message)
		return recoveryGit(t, repository, arguments...)
	}
	left := commitTree("left", root)
	right := commitTree("right", root)
	leftMerge := commitTree("left merge", left, right)
	rightMerge := commitTree("right merge", right, left)
	if bases := strings.Fields(recoveryGit(t, repository, "merge-base", "--all", leftMerge, rightMerge)); len(bases) != 2 {
		t.Fatalf("criss-cross fixture merge bases = %v, want two", bases)
	}

	child, err := plan.Advance(document, "complete", "do")
	if err != nil {
		t.Fatal(err)
	}
	recipe := testutil.ArtifactID(t, artifact.KindRecipe, "ambiguous-recipe")
	profile := testutil.ArtifactID(t, artifact.KindProfile, "ambiguous-profile")
	preparation := testutil.ArtifactID(t, artifact.KindEvidence, "ambiguous-preparation")
	message, err := plan.CompletionCommitMessage(
		[]byte("ambiguous completion"), document, "complete", "do", recipe, profile, preparation,
		artifact.CommitID{1},
	)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(filepath.Join(repository, "store"))
	if err != nil {
		t.Fatal(err)
	}
	unprotectedIntent := gateCommitIntent{
		Parent: left,
		Merge:  &gateMergeIntent{Head: []byte(right + "\n")},
	}
	if err := verifyProspectiveGateCompletion(
		repository, unprotectedIntent, document, child, string(message), store,
	); err == nil || !strings.Contains(err.Error(), "outside the protected epoch") {
		store.Close()
		t.Fatalf("unprotected prospective merge error = %v", err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	intent := gateCommitIntent{Parent: leftMerge, Merge: &gateMergeIntent{Head: []byte(rightMerge + "\n")}}
	if err := verifyProspectiveGateCompletion(repository, intent, document, child, string(message), nil); err == nil ||
		!strings.Contains(err.Error(), "one merge base, found 2") {
		t.Fatalf("ambiguous prospective merge-base error = %v", err)
	}
}

func TestProspectiveGateCompletionAuditsProtectedSideHistory(t *testing.T) {
	fixture := newInterruptedCommitFixture(t)
	publishSuccessfulInterruptedAttempt(t, fixture, true)
	if err := removeGateCommitIntent(fixture.repo); err != nil {
		t.Fatal(err)
	}
	mainBranch := recoveryGit(t, fixture.repo, "rev-parse", "--abbrev-ref", "HEAD")
	target := plan.Plan{Campaign: "interrupted-commit", Doctrine: "test recovery", Items: []plan.Item{{
		ID: "next", Status: plan.StatusOpen, Steps: []plan.Step{{
			ID: "do", Status: plan.StatusOpen, Verify: "go test ./cmd/gate",
		}},
	}}}
	planPath := filepath.Join(fixture.repo, filepath.FromSlash(plan.Path))
	if err := plan.Save(planPath, target); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, fixture.repo, "add", "--", plan.Path)
	runGitFixture(t, fixture.repo, "commit", "-q", "-m", "add next row")
	runGitFixture(t, fixture.repo, "checkout", "-q", "-b", "raw-side-history")
	side := target
	side.Items = append(append([]plan.Item(nil), target.Items...), plan.Item{
		ID: "hidden", Status: plan.StatusOpen, Steps: []plan.Step{{
			ID: "do", Status: plan.StatusOpen, Verify: "go test ./cmd/gate",
		}},
	})
	if err := plan.Save(planPath, side); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, fixture.repo, "add", "--", plan.Path)
	runGitFixture(t, fixture.repo, "commit", "-q", "-m", "raw add hidden row")
	if err := plan.Save(planPath, target); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, fixture.repo, "add", "--", plan.Path)
	runGitFixture(t, fixture.repo, "commit", "-q", "-m", "raw delete hidden row")
	sideParent := recoveryGit(t, fixture.repo, "rev-parse", "HEAD")
	runGitFixture(t, fixture.repo, "checkout", "-q", mainBranch)
	runGitFixture(t, fixture.repo, "commit", "-q", "--allow-empty", "-m", "clean main parent")
	mainParent := recoveryGit(t, fixture.repo, "rev-parse", "HEAD")
	child, err := plan.Advance(target, "next", "do")
	if err != nil {
		t.Fatal(err)
	}
	store := openRecoveryStore(t, fixture)
	defer store.Close()
	intent := gateCommitIntent{Parent: mainParent, Merge: &gateMergeIntent{Head: []byte(sideParent + "\n")}}
	if err := verifyProspectiveGateCompletion(fixture.repo, intent, target, child, "", store); err == nil ||
		!strings.Contains(err.Error(), "deleted a plan item or step without gated completion") {
		t.Fatalf("protected side-history audit error = %v", err)
	}
}

func TestProspectiveGateCompletionRejectsCrossParentIdentityReuse(t *testing.T) {
	fixture := newInterruptedCommitFixture(t)
	publishSuccessfulInterruptedAttempt(t, fixture, true)
	if err := removeGateCommitIntent(fixture.repo); err != nil {
		t.Fatal(err)
	}
	baseBranch := recoveryGit(t, fixture.repo, "rev-parse", "--abbrev-ref", "HEAD")
	common := fixture.commit
	reused := plan.Item{
		ID: "reused", Status: plan.StatusOpen,
		Steps: []plan.Step{{ID: "do", Status: plan.StatusOpen, Verify: "go test ./cmd/gate"}},
	}

	runGitFixture(t, fixture.repo, "checkout", "-q", "-b", "completed-side", common)
	incoming, err := plan.Parse(fixture.planAfter)
	if err != nil {
		t.Fatal(err)
	}
	incoming.Items = append(incoming.Items, reused)
	if err := plan.Save(filepath.Join(fixture.repo, filepath.FromSlash(plan.Path)), incoming); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, fixture.repo, "add", "--", plan.Path)
	runGitFixture(t, fixture.repo, "commit", "-q", "-m", "add reusable completion identity")
	incomingRevision := completeGateRowForMergeAuthority(t, fixture, incoming, "reused", "do")

	runGitFixture(t, fixture.repo, "checkout", "-q", baseBranch)
	local, err := plan.Parse(fixture.planAfter)
	if err != nil {
		t.Fatal(err)
	}
	local.Items = append(local.Items, reused)
	if err := plan.Save(filepath.Join(fixture.repo, filepath.FromSlash(plan.Path)), local); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, fixture.repo, "add", "--", plan.Path)
	runGitFixture(t, fixture.repo, "commit", "-q", "-m", "retain reusable completion identity")
	localRevision := recoveryGit(t, fixture.repo, "rev-parse", "HEAD")

	mergeItem := plan.Item{
		ID: "merge", Status: plan.StatusOpen,
		Steps: []plan.Step{{ID: "do", Status: plan.StatusOpen, Verify: "go test ./cmd/gate"}},
	}
	preAdvance := local
	preAdvance.Items = append([]plan.Item{mergeItem}, local.Items...)
	child, err := plan.Advance(preAdvance, "merge", "do")
	if err != nil {
		t.Fatal(err)
	}
	message, err := plan.CompletionCommitMessage(
		[]byte("complete merge"), preAdvance, "merge", "do",
		testutil.ArtifactID(t, artifact.KindRecipe, "cross-parent merge recipe"),
		testutil.ArtifactID(t, artifact.KindProfile, "cross-parent merge profile"),
		testutil.ArtifactID(t, artifact.KindEvidence, "cross-parent merge preparation"),
		artifact.CommitID{1},
	)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(filepath.Join(fixture.repo, fixture.storePath))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	for _, test := range []struct {
		name, first, second string
	}{
		{name: "live parent first", first: localRevision, second: incomingRevision},
		{name: "completed parent first", first: incomingRevision, second: localRevision},
	} {
		t.Run(test.name, func(t *testing.T) {
			intent := gateCommitIntent{
				Parent: test.first,
				Merge:  &gateMergeIntent{Head: []byte(test.second + "\n")},
			}
			err := verifyProspectiveGateCompletion(
				fixture.repo, intent, preAdvance, child, string(message), store,
			)
			if err == nil || !strings.Contains(err.Error(), "reuses retired item reused") {
				t.Fatalf("cross-parent identity reuse error = %v", err)
			}
		})
	}
}

func completeGateRowForMergeAuthority(
	t *testing.T,
	fixture interruptedCommitFixture,
	document plan.Plan,
	item, step string,
) string {
	t.Helper()
	preparation, err := runrecord.NewGatePreparation(
		strings.Repeat("c", 64), fixture.environment.ID, time.Unix(300, 1),
	)
	if err != nil {
		t.Fatal(err)
	}
	preparationCommit := publishInterruptedPreparationWithKey(
		t, fixture.repo, fixture.storePath, fixture.environment, preparation,
		"fixture/cross-parent/preparation",
	)
	child, err := plan.Advance(document, item, step)
	if err != nil {
		t.Fatal(err)
	}
	planPath := filepath.Join(fixture.repo, filepath.FromSlash(plan.Path))
	if err := plan.Save(planPath, child); err != nil {
		t.Fatal(err)
	}
	candidate := testutil.ArtifactID(t, artifact.KindProfile, "cross-parent completion profile")
	baseManifest := testutil.ArtifactID(t, artifact.KindProfile, "cross-parent base profile")
	manifest, err := automationcheck.BindManifestPlan(
		baseManifest, candidate, strings.Repeat("d", 64), preparation.TreeKey,
		automationcheck.Surface{Identity: "cross-parent completion fixture"}, automationcheck.Impact{},
		[]automationcheck.Invocation{{
			ID:    testutil.ArtifactID(t, artifact.KindRecipe, "cross-parent completion invocation"),
			Check: automationcheck.Descriptor{Name: "commit", Phase: runrecord.PhasePackage},
		}},
	)
	if err != nil {
		t.Fatal(err)
	}
	message, err := plan.CompletionCommitMessage(
		[]byte("complete reusable identity"), document, item, step,
		manifest.ID, candidate, preparation.ID, preparationCommit,
	)
	if err != nil {
		t.Fatal(err)
	}
	messagePath := filepath.Join(t.TempDir(), "completion-message.txt")
	if err := os.WriteFile(messagePath, message, 0o600); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, fixture.repo, "add", "--", plan.Path)
	runGitFixture(t, fixture.repo, "commit", "-q", "-F", messagePath)
	commit := recoveryGit(t, fixture.repo, "rev-parse", "HEAD")
	completion := interruptedCommitFixture{
		repo: fixture.repo, storePath: fixture.storePath, commit: commit,
		recipe: manifest.ID, candidate: candidate, environment: fixture.environment,
		preparation: preparation, preparationCommit: preparationCommit, manifest: manifest,
	}
	_, batch := successfulAttemptBatchForReference(
		t, completion, item, step, "go test ./cmd/gate", true, "fixture/cross-parent/success",
	)
	store := openRecoveryStore(t, completion)
	defer store.Close()
	if _, err := store.Commit(context.Background(), batch); err != nil {
		t.Fatal(err)
	}
	return commit
}
