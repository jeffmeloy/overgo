package gate

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/codemanifest"
	"overgo/internal/gitauthority"
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
	document := completionAuthorityPlan()
	if err := plan.Save(filepath.Join(repository, filepath.FromSlash(plan.Path)), document); err != nil {
		t.Fatal(err)
	}
	messagePath := filepath.Join(repository, "message.txt")
	if err := os.WriteFile(messagePath, []byte("subject\n\nOvergo-Plan-Item: reused\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	manifest := testutil.ArtifactID(t, artifact.KindRecipe, "completion-plan")
	codeManifest := testutil.ArtifactID(t, artifact.KindProfile, "completion-code")
	preparation := testutil.ArtifactID(t, artifact.KindEvidence, "completion-preparation")
	preparationCommit := artifact.CommitID{1}
	gate := gateContext{
		repo: repository, planRef: "item/step", messageFile: messagePath,
		manifestPlan: &automationcheck.ManifestPlan{
			ID: manifest,
		},
		candidateManifest: &codemanifest.Manifest{
			ID: codeManifest,
		},
		preparation: runrecord.GateLifecycle{
			ID: preparation,
		},
		preparationCommit: preparationCommit,
	}
	if _, err := gate.completionMessageFile(document); err == nil {
		t.Fatal("operator-supplied completion identity was accepted")
	} else if !strings.Contains(err.Error(), "reserved trailer Overgo-Plan-Item") {
		t.Fatalf("completion identity refused for the wrong reason: %v", err)
	}
	if err := os.WriteFile(messagePath, []byte("subject\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	mutated := document
	mutated.Items = append([]plan.Item(nil), document.Items...)
	mutated.Items[0].Steps = append([]plan.Step(nil), document.Items[0].Steps...)
	mutated.Items[0].Steps[0].Verify = "go test ./changed"
	if err := plan.Save(filepath.Join(repository, filepath.FromSlash(plan.Path)), mutated); err != nil {
		t.Fatal(err)
	}
	completionPath, err := gate.completionMessageFile(document)
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(completionPath)
	completion, err := os.ReadFile(completionPath)
	if err != nil {
		t.Fatal(err)
	}
	expected, err := plan.CompletionCommitMessageWithMergeAuthority(
		[]byte("subject\n"), document, "item", "step", manifest, codeManifest, preparation, preparationCommit, plan.MergeProjectionSemanticUnion, artifact.ID{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if string(completion) != string(expected) {
		t.Fatal("gate completion message differs from the canonical pre-advance message")
	}
	mutatedExpected, err := plan.CompletionCommitMessageWithMergeAuthority(
		[]byte("subject\n"), mutated, "item", "step", manifest, codeManifest, preparation, preparationCommit, plan.MergeProjectionSemanticUnion, artifact.ID{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if string(completion) == string(mutatedExpected) {
		t.Fatal("mutable on-disk plan replaced the pre-advance completion authority")
	}
}

func TestAcceptanceEvidenceRejectsPlanMutationBeforeCommit(t *testing.T) {
	repository := t.TempDir()
	if err := os.Mkdir(filepath.Join(repository, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	document := completionAuthorityPlan()
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
	authority, err := plan.ResolveCompletionAuthority(t.Context(), repository, "HEAD", document, store)
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

func TestGatePlanProjectionIsExplicitAndMergeOnly(t *testing.T) {
	if projection, err := gatePlanProjection("", false); err != nil ||
		projection != plan.MergeProjectionSemanticUnion {
		t.Fatalf("default projection = %q, %v", projection, err)
	}
	if _, err := gatePlanProjection("first-parent-target", false); err == nil ||
		!strings.Contains(err.Error(), "requires -merge") {
		t.Fatalf("non-merge target projection error = %v", err)
	}
	if projection, err := gatePlanProjection("first-parent-target", true); err != nil ||
		projection != plan.MergeProjectionFirstParentTarget {
		t.Fatalf("merge target projection = %q, %v", projection, err)
	}
	for _, invalid := range []string{"semantic-union", " first-parent-target", "incoming-plan"} {
		if _, err := gatePlanProjection(invalid, true); err == nil {
			t.Fatalf("invalid projection %q accepted", invalid)
		}
	}
}

func TestGateMergeSourceStoreIsFirstParentTargetOnly(t *testing.T) {
	store := filepath.Join(t.TempDir(), gitauthority.CanonicalOvergoDBDirectory)
	resolved, err := gateMergeSourceStore(store, true, plan.MergeProjectionFirstParentTarget)
	if err != nil || !filepath.IsAbs(resolved) || filepath.Base(resolved) != gitauthority.CanonicalOvergoDBDirectory {
		t.Fatalf("target merge source store = (%q, %v)", resolved, err)
	}
	if _, err := gateMergeSourceStore("", true, plan.MergeProjectionFirstParentTarget); err == nil {
		t.Fatal("target projection accepted no source authority store")
	}
	for _, projection := range []plan.MergeProjection{
		plan.MergeProjectionSemanticUnion,
	} {
		if _, err := gateMergeSourceStore(store, true, projection); err == nil {
			t.Fatalf("projection %q accepted a foreign source store", projection)
		}
	}
	if _, err := gateMergeSourceStore(store, false, plan.MergeProjectionFirstParentTarget); err == nil {
		t.Fatal("non-merge accepted a source authority store")
	}
}

func TestGateCompletionMessageBindsFirstParentTarget(t *testing.T) {
	repository := t.TempDir()
	document := completionAuthorityPlan()
	messagePath := filepath.Join(repository, "message.txt")
	if err := os.WriteFile(messagePath, []byte("target merge\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	manifest := testutil.ArtifactID(t, artifact.KindRecipe, "target-gate-plan")
	codeManifest := testutil.ArtifactID(t, artifact.KindProfile, "target-gate-code")
	preparation := testutil.ArtifactID(t, artifact.KindEvidence, "target-gate-preparation")
	preparationCommit := artifact.CommitID{1}
	mergeAuthority := testutil.ArtifactID(t, artifact.KindEvidence, "target-gate-merge-authority")
	gate := gateContext{
		repo: repository, planRef: "item/step", messageFile: messagePath,
		manifestPlan:      &automationcheck.ManifestPlan{ID: manifest},
		candidateManifest: &codemanifest.Manifest{ID: codeManifest},
		preparation:       runrecord.GateLifecycle{ID: preparation},
		preparationCommit: preparationCommit,
		planProjection:    plan.MergeProjectionFirstParentTarget,
		mergeAuthority:    &plan.FirstParentTargetMergeAuthority{ID: mergeAuthority},
	}
	completionPath, err := gate.completionMessageFile(document)
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(completionPath)
	message, err := os.ReadFile(completionPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := plan.VerifyCompletionCommitMessageWithMergeAuthority(
		string(message), document, "item", "step", manifest, codeManifest,
		preparation, preparationCommit, plan.MergeProjectionFirstParentTarget, mergeAuthority,
	); err != nil {
		t.Fatalf("target projection was not bound to the completion message: %v", err)
	}
	if err := plan.VerifyCompletionCommitMessageWithMergeAuthority(
		string(message), document, "item", "step", manifest, codeManifest,
		preparation, preparationCommit, plan.MergeProjectionSemanticUnion, artifact.ID{},
	); err == nil {
		t.Fatal("target projection completion replayed as default semantic union")
	}
}

func TestProspectiveGateCompletionRejectsAuditedSourceDependency(t *testing.T) {
	fixture := newInterruptedCommitFixture(t)
	publishSuccessfulInterruptedAttempt(t, fixture, true)
	if err := removeGateCommitIntent(fixture.repo); err != nil {
		t.Fatal(err)
	}
	local, err := plan.Parse(fixture.planAfter)
	if err != nil {
		t.Fatal(err)
	}
	primary := recoveryGit(t, fixture.repo, "rev-parse", "--abbrev-ref", "HEAD")
	localRevision := recoveryGit(t, fixture.repo, "rev-parse", "HEAD")
	runGitFixture(t, fixture.repo, "checkout", "-q", "-b", "source-audit")
	incoming := local
	incoming.Items = append(append([]plan.Item(nil), local.Items...), plan.Item{
		ID: "source-only", Status: plan.StatusOpen,
		Steps: []plan.Step{{ID: "do", Status: plan.StatusOpen, Verify: "go test ./cmd/gate"}},
	})
	if err := plan.Save(filepath.Join(fixture.repo, filepath.FromSlash(plan.Path)), incoming); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, fixture.repo, "add", "--", plan.Path)
	runGitFixture(t, fixture.repo, "commit", "-q", "-m", "add source-only completion")
	incomingRevision := completeGateRowForMergeAuthority(t, fixture, incoming, "source-only", "do")
	incoming, err = plan.Advance(incoming, "source-only", "do")
	if err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, fixture.repo, "checkout", "-q", primary)

	targetStore, err := overgodb.Open(filepath.Join(fixture.repo, fixture.storePath))
	if err != nil {
		t.Fatal(err)
	}
	defer targetStore.Close()
	sourcePath := filepath.Join(t.TempDir(), "source-store")
	if _, _, err := targetStore.Backup(sourcePath); err != nil {
		t.Fatal(err)
	}
	sourceStore, err := overgodb.Open(sourcePath)
	if err != nil {
		t.Fatal(err)
	}
	defer sourceStore.Close()
	localAuthority, err := plan.ResolveCompletionAuthority(
		t.Context(), fixture.repo, localRevision, local, targetStore,
	)
	if err != nil {
		t.Fatal(err)
	}
	incomingAuthority, err := plan.ResolveCompletionAuthority(
		t.Context(), fixture.repo, incomingRevision, incoming, sourceStore,
	)
	if err != nil {
		t.Fatal(err)
	}
	mergeRow := plan.Item{
		ID: "merge-boundary", Status: plan.StatusOpen,
		Steps: []plan.Step{{ID: "do", Status: plan.StatusOpen, Verify: "go test ./cmd/gate"}},
	}
	preAdvance := local
	preAdvance.Items = append(append([]plan.Item(nil), local.Items...), mergeRow)
	child, err := plan.Advance(preAdvance, "merge-boundary", "do")
	if err != nil {
		t.Fatal(err)
	}
	receipt, err := plan.NewFirstParentTargetMergeAuthority(
		t.Context(), fixture.repo, localRevision, incomingRevision, localRevision,
		local, incoming, local, preAdvance, child, "merge-boundary", "do",
		fixture.preparation.ID, fixture.preparationCommit,
		localAuthority, incomingAuthority, targetStore, sourceStore,
	)
	if err != nil {
		t.Fatal(err)
	}
	malicious := preAdvance
	malicious.Items = append([]plan.Item(nil), preAdvance.Items...)
	mergeIndex := len(malicious.Items) - 1
	malicious.Items[mergeIndex].Steps = append([]plan.Step(nil), malicious.Items[mergeIndex].Steps...)
	malicious.Items[mergeIndex].Steps[0].DependsOn = []string{"source-only/do"}
	maliciousChild, err := plan.Advance(malicious, "merge-boundary", "do")
	if err != nil {
		t.Fatal(err)
	}
	encodedPlan, err := json.Marshal(malicious)
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(encodedPlan)
	receipt.PreAdvancePlanDigest = hex.EncodeToString(digest[:])
	receipt.ID = artifact.ID{}
	encodedReceipt, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	receipt, err = plan.ParseFirstParentTargetMergeAuthority(encodedReceipt)
	if err != nil {
		t.Fatal(err)
	}
	receiptContent, err := receipt.Content()
	if err != nil {
		t.Fatal(err)
	}
	message, err := plan.CompletionCommitMessageWithMergeAuthority(
		[]byte("complete audited merge boundary"), malicious, "merge-boundary", "do",
		fixture.recipe, fixture.candidate, fixture.preparation.ID, fixture.preparationCommit,
		plan.MergeProjectionFirstParentTarget, receipt.ID,
	)
	if err != nil {
		t.Fatal(err)
	}
	intent := gateCommitIntent{
		Parent: localRevision, Merge: &gateMergeIntent{Head: []byte(incomingRevision + "\n")},
		PlanProjection: plan.MergeProjectionFirstParentTarget, MergeAuthority: receiptContent.Data,
		PlanRef: "merge-boundary/do", Preparation: fixture.preparation.ID,
		PreparationCommit: fixture.preparationCommit,
	}
	if err := verifyProspectiveGateCompletion(
		fixture.repo, intent, malicious, maliciousChild, string(message), targetStore,
	); err == nil || !strings.Contains(err.Error(), "pruned dependency source-only/do") {
		t.Fatalf("gate recovery admitted audited source dependency: %v", err)
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
	message, err := plan.CompletionCommitMessageWithMergeAuthority(
		[]byte("prospective completion"), preAdvance, "complete", "do", recipe, profile, preparation,
		artifact.CommitID{1}, plan.MergeProjectionSemanticUnion, artifact.ID{},
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
	message, err := plan.CompletionCommitMessageWithMergeAuthority(
		[]byte("ambiguous completion"), document, "complete", "do", recipe, profile, preparation,
		artifact.CommitID{1}, plan.MergeProjectionSemanticUnion, artifact.ID{},
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
	// A criss-cross history (each side merged the other) has two bases;
	// the one on the target's first-parent chain is selected, so the
	// completion proceeds to the next refusal, the absent authority store.
	intent := gateCommitIntent{Parent: leftMerge, Merge: &gateMergeIntent{Head: []byte(rightMerge + "\n")}}
	if err := verifyProspectiveGateCompletion(repository, intent, document, child, string(message), nil); err == nil ||
		!strings.Contains(err.Error(), "locked authority store") {
		t.Fatalf("criss-cross prospective merge error = %v", err)
	}
	if selected, err := plan.CompletionMergeBase(t.Context(), repository, leftMerge, rightMerge); err != nil || selected != left {
		t.Fatalf("criss-cross merge base = %q, %v; want the target's first-parent base %q", selected, err, left)
	}
	// A target whose first-parent chain holds none of the bases refuses.
	fork := commitTree("fork", root)
	forkMerge := commitTree("fork merge", fork, leftMerge)
	if bases := strings.Fields(recoveryGit(t, repository, "merge-base", "--all", forkMerge, rightMerge)); len(bases) != 2 {
		t.Fatalf("forked fixture merge bases = %v, want two", bases)
	}
	forked := gateCommitIntent{Parent: forkMerge, Merge: &gateMergeIntent{Head: []byte(rightMerge + "\n")}}
	if err := verifyProspectiveGateCompletion(repository, forked, document, child, string(message), nil); err == nil ||
		!strings.Contains(err.Error(), "first-parent chain, found 0 of 2") {
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
	message, err := plan.CompletionCommitMessageWithMergeAuthority(
		[]byte("complete merge"), preAdvance, "merge", "do",
		testutil.ArtifactID(t, artifact.KindRecipe, "cross-parent merge recipe"),
		testutil.ArtifactID(t, artifact.KindProfile, "cross-parent merge profile"),
		testutil.ArtifactID(t, artifact.KindEvidence, "cross-parent merge preparation"),
		artifact.CommitID{1}, plan.MergeProjectionSemanticUnion, artifact.ID{},
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
	message, err := plan.CompletionCommitMessageWithMergeAuthority(
		[]byte("complete reusable identity"), document, item, step,
		manifest.ID, candidate, preparation.ID, preparationCommit, plan.MergeProjectionSemanticUnion, artifact.ID{},
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
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	return commit
}

func completionAuthorityPlan() plan.Plan {
	return plan.Plan{Campaign: "gate", Doctrine: "fixture", Items: []plan.Item{{
		ID: "item", Status: plan.StatusOpen, Steps: []plan.Step{{
			ID: "step", Status: plan.StatusOpen, Verify: "go test ./...",
		}},
	}}}
}
