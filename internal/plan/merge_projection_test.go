package plan

import (
	"context"
	"errors"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func targetProjectionMessage(t *testing.T, document Plan, item, step string) []byte {
	t.Helper()
	message, err := CompletionCommitMessageWithMergeAuthority(
		[]byte("target plan completion"), document, item, step,
		testutil.ArtifactID(t, artifact.KindRecipe, "target-plan-manifest"),
		testutil.ArtifactID(t, artifact.KindProfile, "target-plan-code-manifest"),
		testutil.ArtifactID(t, artifact.KindEvidence, "target-plan-preparation"),
		artifact.CommitID{1}, MergeProjectionFirstParentTarget,
		testutil.ArtifactID(t, artifact.KindEvidence, "target-plan-merge-authority"),
	)
	if err != nil {
		t.Fatal(err)
	}
	return message
}

func TestFirstParentTargetProspectiveTransitionLeavesIncomingWorkSourceOwned(t *testing.T) {
	base := standardCompletionPlan()
	local := base
	incoming := base
	incoming.Items = append(slices.Clone(base.Items), Item{
		ID: "incoming-only", Status: StatusOpen,
		Steps: []Step{{ID: "do", Status: StatusOpen, Verify: "go test ./..."}},
	})
	child, err := Advance(local, "root", "do")
	if err != nil {
		t.Fatal(err)
	}
	message := targetProjectionMessage(t, local, "root", "do")
	if err := VerifyProspectiveCompletionTransitionWithProjection(
		[]Plan{local, incoming}, &base, local, child, string(message),
		MergeProjectionFirstParentTarget,
	); err != nil {
		t.Fatalf("target projection refused incoming-only source work: %v", err)
	}
	if slices.ContainsFunc(child.Items, func(item Item) bool { return item.ID == "incoming-only" }) {
		t.Fatal("incoming-only work leaked into the target plan")
	}
	if err := VerifyProspectiveCompletionTransition(
		[]Plan{local, incoming}, &base, local, child, string(message),
	); err == nil {
		t.Fatal("explicit target trailer was silently interpreted as semantic union")
	}
}

func TestFirstParentTargetRejectsLocalIdentityDeletion(t *testing.T) {
	base := standardCompletionPlan()
	preAdvance := base
	preAdvance.Items = slices.Delete(slices.Clone(base.Items), 1, 2)
	child, err := Advance(preAdvance, "root", "do")
	if err != nil {
		t.Fatal(err)
	}
	message := targetProjectionMessage(t, preAdvance, "root", "do")
	if err := VerifyProspectiveCompletionTransitionWithProjection(
		[]Plan{base, base}, &base, preAdvance, child, string(message),
		MergeProjectionFirstParentTarget,
	); err == nil || !strings.Contains(err.Error(), "deleted a baseline") {
		t.Fatalf("first-parent deletion error = %v", err)
	}
}

func TestFirstParentTargetTrailerIsCanonicalAndMergeOnly(t *testing.T) {
	document := standardCompletionPlan()
	child, err := Advance(document, "root", "do")
	if err != nil {
		t.Fatal(err)
	}
	message := string(targetProjectionMessage(t, document, "root", "do"))
	if err := VerifyProspectiveCompletionTransitionWithProjection(
		[]Plan{document}, nil, document, child, message, MergeProjectionFirstParentTarget,
	); err == nil || !strings.Contains(err.Error(), "requires a two-parent merge") {
		t.Fatalf("non-merge target projection error = %v", err)
	}
	duplicate := strings.Replace(
		message,
		completionMergeProjectionTrailer+": "+string(MergeProjectionFirstParentTarget),
		completionMergeProjectionTrailer+": "+string(MergeProjectionFirstParentTarget)+"\n"+
			completionMergeProjectionTrailer+": "+string(MergeProjectionFirstParentTarget),
		1,
	)
	if _, _, err := parseCompletionTrailers(duplicate); err == nil || !strings.Contains(err.Error(), "at most once") {
		t.Fatalf("duplicate target projection error = %v", err)
	}
	withoutAuthority := ""
	for line := range strings.SplitSeq(message, "\n") {
		if strings.HasPrefix(line, completionMergeAuthorityTrailer+":") {
			continue
		}
		withoutAuthority += line + "\n"
	}
	if _, _, err := parseCompletionTrailers(withoutAuthority); err == nil || !strings.Contains(err.Error(), "lacks merge authority") {
		t.Fatalf("missing target merge authority error = %v", err)
	}
	if _, err := CompletionCommitMessageWithMergeAuthority(
		[]byte("target without receipt"), document, "root", "do",
		testutil.ArtifactID(t, artifact.KindRecipe, "missing-receipt-manifest"),
		testutil.ArtifactID(t, artifact.KindProfile, "missing-receipt-code-manifest"),
		testutil.ArtifactID(t, artifact.KindEvidence, "missing-receipt-preparation"),
		artifact.CommitID{1}, MergeProjectionFirstParentTarget, artifact.ID{},
	); err == nil || !strings.Contains(err.Error(), "requires merge authority") {
		t.Fatalf("target writer without receipt error = %v", err)
	}
	invalid := strings.Replace(
		message, string(MergeProjectionFirstParentTarget), "incoming-plan", 1,
	)
	if _, _, err := parseCompletionTrailers(invalid); err == nil || !strings.Contains(err.Error(), "invalid value") {
		t.Fatalf("invalid target projection error = %v", err)
	}
	if _, err := CompletionCommitMessage(
		[]byte("operator\n\n"+completionMergeProjectionTrailer+": first-parent-target"),
		document, "root", "do",
		testutil.ArtifactID(t, artifact.KindRecipe, "reserved-manifest"),
		testutil.ArtifactID(t, artifact.KindProfile, "reserved-code-manifest"),
		testutil.ArtifactID(t, artifact.KindEvidence, "reserved-preparation"), artifact.CommitID{1},
	); err == nil || !strings.Contains(err.Error(), "reserved trailer") {
		t.Fatalf("operator target trailer error = %v", err)
	}
}

func TestFirstParentTargetProspectiveAuthorityKeepsIncomingRowsOffTarget(t *testing.T) {
	const repository = "test-repository"
	const localRevision = "1111111111111111111111111111111111111111"
	const incomingRevision = "2222222222222222222222222222222222222222"
	local := Plan{Items: []Item{{
		ID: "local", Status: StatusOpen,
		Steps: []Step{{ID: "do", Status: StatusOpen, Verify: "go test ./..."}},
	}}}
	incoming := Plan{Items: []Item{{
		ID: "incoming", Status: StatusOpen,
		Steps: []Step{{ID: "do", Status: StatusOpen, Verify: "go test ./..."}},
	}}}
	localAuthority := prospectiveMergeAuthority(t, local, repository, localRevision, "common")
	incomingAuthority := prospectiveMergeAuthority(t, incoming, repository, incomingRevision, "common")
	if err := VerifyProspectiveMergeAuthorityWithProjection(
		repository, localRevision, incomingRevision, local, incoming, local,
		localAuthority, incomingAuthority, MergeProjectionFirstParentTarget,
	); err != nil {
		t.Fatalf("incoming-only source row was required on target: %v", err)
	}
	if err := VerifyProspectiveMergeAuthority(
		repository, localRevision, incomingRevision, local, incoming, local,
		localAuthority, incomingAuthority,
	); err == nil {
		t.Fatal("semantic union stopped requiring the incoming parent identity")
	}
	waiting := local
	waiting.Items = slices.Clone(local.Items)
	waiting.Items[0].Steps = slices.Clone(local.Items[0].Steps)
	waiting.Items[0].Steps[0].DependsOn = []string{"retired/do"}
	waitingAuthority := prospectiveMergeAuthority(t, waiting, repository, localRevision, "common")
	if err := VerifyProspectiveMergeAuthorityWithProjection(
		repository, localRevision, incomingRevision, waiting, incoming, waiting,
		waitingAuthority, incomingAuthority, MergeProjectionFirstParentTarget,
	); err == nil || !strings.Contains(err.Error(), "pruned dependency retired/do") {
		t.Fatalf("target missing dependency evidence error = %v", err)
	}
	incomingAuthority.completedReferences["retired/do"] = completionEvidence{commit: incomingRevision}
	if err := VerifyProspectiveMergeAuthorityWithProjection(
		repository, localRevision, incomingRevision, waiting, incoming, waiting,
		waitingAuthority, incomingAuthority, MergeProjectionFirstParentTarget,
	); err == nil || !strings.Contains(err.Error(), "pruned dependency retired/do") {
		t.Fatalf("incoming completion evidence entered target authority: %v", err)
	}
	waitingAuthority.completedReferences["retired/do"] = completionEvidence{commit: localRevision}
	if err := VerifyProspectiveMergeAuthorityWithProjection(
		repository, localRevision, incomingRevision, waiting, incoming, waiting,
		waitingAuthority, incomingAuthority, MergeProjectionFirstParentTarget,
	); err != nil {
		t.Fatalf("local completion evidence did not authorize target dependency: %v", err)
	}
	delete(incomingAuthority.completedReferences, "retired/do")
	incomingAuthority.completedReferences["local/do"] = completionEvidence{commit: incomingRevision}
	if err := VerifyProspectiveMergeAuthorityWithProjection(
		repository, localRevision, incomingRevision, local, incoming, local,
		localAuthority, incomingAuthority, MergeProjectionFirstParentTarget,
	); err != nil {
		t.Fatalf("incoming completion identity constrained target: %v", err)
	}
	incomingRetirement := completionEvidence{commit: incomingRevision, retiredItem: true}
	incomingAuthority.completedReferences["local/retired"] = incomingRetirement
	incomingAuthority.retiredItems["local"] = incomingRetirement
	if err := VerifyProspectiveMergeAuthorityWithProjection(
		repository, localRevision, incomingRevision, local, incoming, local,
		localAuthority, incomingAuthority, MergeProjectionFirstParentTarget,
	); err != nil {
		t.Fatalf("incoming retirement identity constrained target: %v", err)
	}
	deleted := Plan{}
	if err := VerifyProspectiveMergeAuthorityWithProjection(
		repository, localRevision, incomingRevision, local, incoming, deleted,
		localAuthority, incomingAuthority, MergeProjectionFirstParentTarget,
	); err == nil || !strings.Contains(err.Error(), "deleted a parent") {
		t.Fatalf("target local deletion error = %v", err)
	}
}

func TestFirstParentTargetSourcePreflightRequiresSharedStoreAncestry(t *testing.T) {
	repository := t.TempDir()
	localRevision := strings.Repeat("1", 40)
	incomingRevision := strings.Repeat("2", 40)
	local := Plan{Items: []Item{{
		ID: "local", Status: StatusOpen,
		Steps: []Step{{ID: "do", Status: StatusOpen, Verify: "go test ./..."}},
	}}}
	incoming := Plan{Items: []Item{{
		ID: "incoming", Status: StatusOpen,
		Steps: []Step{{ID: "do", Status: StatusOpen, Verify: "go test ./..."}},
	}}}
	targetStore, err := overgodb.Open(filepath.Join(t.TempDir(), "target-store"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = targetStore.Close() })
	common := testutil.ArtifactID(t, artifact.KindEvidence, "shared-store-prefix")
	if _, err := targetStore.Commit(t.Context(), artifact.Batch{
		Key: "common", Artifacts: []artifact.Descriptor{{ID: common}},
	}); err != nil {
		t.Fatal(err)
	}
	sourcePath := filepath.Join(t.TempDir(), "source-store")
	if _, _, err := targetStore.Backup(sourcePath); err != nil {
		t.Fatal(err)
	}
	sourceStore, err := overgodb.Open(sourcePath)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = sourceStore.Close() })
	for key, store := range map[string]*overgodb.Store{"local": targetStore, "incoming": sourceStore} {
		marker := testutil.ArtifactID(t, artifact.KindEvidence, key+"-divergence")
		if _, err := store.Commit(t.Context(), artifact.Batch{
			Key: key, Artifacts: []artifact.Descriptor{{ID: marker}},
		}); err != nil {
			t.Fatal(err)
		}
	}
	localAuthority := prospectiveMergeAuthority(t, local, repository, localRevision, strings.Repeat("a", 40))
	incomingAuthority := prospectiveMergeAuthority(t, incoming, repository, incomingRevision, strings.Repeat("b", 40))
	localAuthority.storeHead, localAuthority.storeSequence = targetStore.Head()
	incomingAuthority.storeHead, incomingAuthority.storeSequence = sourceStore.Head()
	if err := VerifyProspectiveMergeAuthorityWithProjection(
		repository, localRevision, incomingRevision, local, incoming, local,
		localAuthority, incomingAuthority, MergeProjectionFirstParentTarget,
	); err == nil || !strings.Contains(err.Error(), "share a protected completion epoch") {
		t.Fatalf("strict same-epoch verifier error = %v", err)
	}
	prefix, err := VerifyFirstParentTargetMergeSources(
		t.Context(), repository, localRevision, incomingRevision, local, incoming, local,
		localAuthority, incomingAuthority, targetStore, sourceStore,
	)
	if err != nil || !prefix.Commit.Valid() || prefix.Sequence == 0 {
		t.Fatalf("receipt-backed epoch join = (%+v, %v)", prefix, err)
	}
	retirementContract, err := completionContractDigest(Item{
		ID: "retired", Status: StatusOpen,
		Steps: []Step{{ID: "done", Status: StatusOpen, Verify: "go test ./..."}},
	})
	if err != nil {
		t.Fatal(err)
	}
	retirementLocal := completionEvidence{
		commit: localRevision, verify: "go test ./...", contract: retirementContract, retiredItem: true,
	}
	retirementIncoming := completionEvidence{
		commit: incomingRevision, verify: "go test ./...", contract: retirementContract, retiredItem: true,
	}
	retirementLocalAuthority := localAuthority
	retirementLocalAuthority.completedReferences = maps.Clone(localAuthority.completedReferences)
	retirementLocalAuthority.retiredItems = maps.Clone(localAuthority.retiredItems)
	retirementLocalAuthority.completedReferences["retired/local"] = retirementLocal
	retirementLocalAuthority.retiredItems["retired"] = retirementLocal
	retirementIncomingAuthority := incomingAuthority
	retirementIncomingAuthority.completedReferences = maps.Clone(incomingAuthority.completedReferences)
	retirementIncomingAuthority.retiredItems = maps.Clone(incomingAuthority.retiredItems)
	retirementIncomingAuthority.completedReferences["retired/incoming"] = retirementIncoming
	retirementIncomingAuthority.retiredItems["retired"] = retirementIncoming
	if _, err := VerifyFirstParentTargetMergeSources(
		t.Context(), repository, localRevision, incomingRevision, local, incoming, local,
		retirementLocalAuthority, retirementIncomingAuthority, targetStore, sourceStore,
	); err != nil {
		t.Fatalf("source retirement conflict constrained target preflight: %v", err)
	}
	orphanRetirementAuthority := incomingAuthority
	orphanRetirementAuthority.retiredItems = maps.Clone(incomingAuthority.retiredItems)
	orphanRetirementAuthority.retiredItems["orphan"] = retirementIncoming
	if _, err := VerifyFirstParentTargetMergeSources(
		t.Context(), repository, localRevision, incomingRevision, local, incoming, local,
		localAuthority, orphanRetirementAuthority, targetStore, sourceStore,
	); err == nil || !strings.Contains(err.Error(), "0 exact completion references") {
		t.Fatalf("orphan retirement preflight error = %v", err)
	}

	foreignStore, err := overgodb.Open(filepath.Join(t.TempDir(), "foreign-store"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = foreignStore.Close() })
	foreign := testutil.ArtifactID(t, artifact.KindEvidence, "foreign-store")
	if _, err := foreignStore.Commit(t.Context(), artifact.Batch{
		Key: "foreign", Artifacts: []artifact.Descriptor{{ID: foreign}},
	}); err != nil {
		t.Fatal(err)
	}
	foreignAuthority := incomingAuthority
	foreignAuthority.storeHead, foreignAuthority.storeSequence = foreignStore.Head()
	if _, err := VerifyFirstParentTargetMergeSources(
		t.Context(), repository, localRevision, incomingRevision, local, incoming, local,
		localAuthority, foreignAuthority, targetStore, foreignStore,
	); err == nil || !strings.Contains(err.Error(), "no shared hash-chain prefix") {
		t.Fatalf("foreign store prefix error = %v", err)
	}
}

func TestFirstParentTargetReconcilesEquivalentIndependentCompletion(t *testing.T) {
	initial := Plan{Campaign: "completion reconciliation", Doctrine: "local evidence remains authoritative", Items: []Item{
		{ID: "duplicate", Status: StatusOpen, Steps: []Step{
			{ID: "done", Status: StatusOpen, Verify: "go test ./..."},
			{ID: "remain", Status: StatusOpen, Verify: "go test ./...", DependsOn: []string{"duplicate/done"}},
		}},
		{ID: "target", Status: StatusOpen, Steps: []Step{{ID: "do", Status: StatusOpen, Verify: "go test ./..."}}},
	}}
	fixture := newCompletionFixture(t, initial, "duplicate", "done")
	primary := strings.TrimSpace(string(runGit(
		t, fixture.repository, nil, "rev-parse", "--abbrev-ref", "HEAD",
	)))
	runGit(t, fixture.repository, nil, "branch", "independent-source")

	sourceStorePath := filepath.Join(t.TempDir(), "source-store")
	// Fork the store at the common preparation, before either final attempt.
	if _, _, err := fixture.store.Backup(sourceStorePath); err != nil {
		t.Fatal(err)
	}
	localCompletion := fixture.commit(fixture.canonicalMessage(), true)
	localPlan := fixture.child
	sourceStore, err := overgodb.Open(sourceStorePath)
	if err != nil {
		t.Fatal(err)
	}
	sourceClosed := false
	t.Cleanup(func() {
		if !sourceClosed {
			_ = sourceStore.Close()
		}
	})

	runGit(t, fixture.repository, nil, "checkout", "-q", "independent-source")
	sourcePreparation, err := runrecord.NewGatePreparation(
		strings.Repeat("d", 64), fixture.preparation.Environment, time.Unix(2, 0),
	)
	if err != nil {
		t.Fatal(err)
	}
	sourcePreparationContent, err := sourcePreparation.Content()
	if err != nil {
		t.Fatal(err)
	}
	sourcePreparationCommit, err := sourceStore.Commit(t.Context(), artifact.Batch{
		Key: "source/prepared/duplicate", Contents: []artifact.Content{sourcePreparationContent},
		Lineage: sourcePreparation.Lineage(),
	})
	if err != nil {
		t.Fatal(err)
	}
	sourceCodeManifest := testutil.ArtifactID(t, artifact.KindProfile, "duplicate-source-code")
	sourceManifest := completionManifestPlan(t, sourceCodeManifest, sourcePreparation.TreeKey).ID
	sourceMessage, err := CompletionCommitMessage(
		[]byte("independently complete duplicate"), initial, "duplicate", "done",
		sourceManifest, sourceCodeManifest, sourcePreparation.ID, sourcePreparationCommit,
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := Save(filepath.Join(fixture.repository, filepath.FromSlash(Path)), localPlan); err != nil {
		t.Fatal(err)
	}
	runGit(t, fixture.repository, nil, "add", "--", Path)
	runGit(t, fixture.repository, sourceMessage, "commit", "-q", "-F", "-")
	incomingCompletion := strings.TrimSpace(string(runGit(t, fixture.repository, nil, "rev-parse", "HEAD")))
	publishCompletionAttempt(
		t, sourceStore, incomingCompletion, "duplicate", "done", "go test ./...", "go test ./...",
		sourceManifest, sourceCodeManifest, sourcePreparation,
	)

	runGit(t, fixture.repository, nil, "checkout", "-q", primary)
	runGit(t, fixture.repository, nil, "merge", "--no-ff", "--no-commit", "independent-source")
	mergeBaseRevision := strings.TrimSpace(string(runGit(
		t, fixture.repository, nil, "merge-base", localCompletion, incomingCompletion,
	)))
	child, err := Advance(localPlan, "target", "do")
	if err != nil {
		t.Fatal(err)
	}
	fixture.preAdvance, fixture.child = localPlan, child
	fixture.item, fixture.step = "target", "do"
	fixture.verify, fixture.acceptance = "go test ./...", "go test ./..."
	fixture.codeManifest = testutil.ArtifactID(t, artifact.KindProfile, "duplicate-target-code")
	fixture.rotatePreparation(strings.Repeat("e", 64), time.Unix(3, 0))
	fixture.manifest = completionManifestPlan(t, fixture.codeManifest, fixture.preparation.TreeKey).ID
	localAuthority, err := ResolveCompletionAuthority(
		t.Context(), fixture.repository, localCompletion, localPlan, fixture.store,
	)
	if err != nil {
		t.Fatal(err)
	}
	incomingAuthority, err := ResolveCompletionAuthority(
		t.Context(), fixture.repository, incomingCompletion, localPlan, sourceStore,
	)
	if err != nil {
		t.Fatal(err)
	}
	if localAuthority.completedReferences["duplicate/done"] == incomingAuthority.completedReferences["duplicate/done"] {
		t.Fatal("independent completions unexpectedly share evidence")
	}

	for name, mutate := range map[string]func(completionEvidence) completionEvidence{
		"verifier": func(evidence completionEvidence) completionEvidence {
			evidence.verify = "go test ./different"
			return evidence
		},
		"completion contract": func(evidence completionEvidence) completionEvidence {
			evidence.contract[0] ^= 0xff
			return evidence
		},
	} {
		t.Run("audit "+name+" source conflict", func(t *testing.T) {
			mismatched := incomingAuthority
			mismatched.completedReferences = maps.Clone(incomingAuthority.completedReferences)
			changed := mutate(mismatched.completedReferences["duplicate/done"])
			mismatched.completedReferences["duplicate/done"] = changed
			audit, err := NewFirstParentTargetMergeAuthority(
				t.Context(), fixture.repository, localCompletion, incomingCompletion, mergeBaseRevision,
				localPlan, localPlan, initial, localPlan, child, "target", "do",
				fixture.preparation.ID, fixture.preparationCommit,
				localAuthority, mismatched, fixture.store, sourceStore,
			)
			if err != nil {
				t.Fatalf("%s source conflict blocked target: %v", name, err)
			}
			if audit.SourceAuthorityPolicy != FirstParentTargetSourceAuditOnlyPolicy ||
				len(audit.AuditedIncomingCompletions) != 1 ||
				audit.AuditedIncomingCompletions[0] != exportProjectedCompletion("duplicate/done", changed) {
				t.Fatalf("%s incoming audit = %+v", name, audit.AuditedIncomingCompletions)
			}
		})
	}
	retiredMismatch := incomingAuthority
	retiredMismatch.completedReferences = maps.Clone(incomingAuthority.completedReferences)
	retiredMismatch.retiredItems = maps.Clone(incomingAuthority.retiredItems)
	retired := retiredMismatch.completedReferences["duplicate/done"]
	retired.retiredItem = true
	retiredMismatch.completedReferences["duplicate/done"] = retired
	retiredMismatch.retiredItems["duplicate"] = retired
	receipt, err := NewFirstParentTargetMergeAuthority(
		t.Context(), fixture.repository, localCompletion, incomingCompletion, mergeBaseRevision,
		localPlan, localPlan, initial, localPlan, child, "target", "do",
		fixture.preparation.ID, fixture.preparationCommit,
		localAuthority, retiredMismatch, fixture.store, sourceStore,
	)
	if err != nil {
		t.Fatalf("coherent source retirement conflict blocked target: %v", err)
	}
	if receipt.SourceAuthorityPolicy != FirstParentTargetSourceAuditOnlyPolicy ||
		len(receipt.AuditedIncomingCompletions) != 1 ||
		len(receipt.AuditedIncomingRetirements) != 1 ||
		receipt.AuditedIncomingCompletions[0].Reference != "duplicate/done" ||
		receipt.AuditedIncomingCompletions[0].Commit != incomingCompletion {
		t.Fatalf("audited incoming authority = (%+v, %+v)",
			receipt.AuditedIncomingCompletions, receipt.AuditedIncomingRetirements)
	}
	tamperedContract := receipt
	tamperedContract.AuditedIncomingCompletions = slices.Clone(receipt.AuditedIncomingCompletions)
	tamperedContract.AuditedIncomingRetirements = slices.Clone(receipt.AuditedIncomingRetirements)
	tamperedContract.AuditedIncomingCompletions[0].ContractDigest = strings.Repeat("a", 64)
	tamperedContract.AuditedIncomingRetirements[0].Evidence.ContractDigest = strings.Repeat("a", 64)
	tamperedContract.ID = artifact.ID{}
	if _, err := firstParentTargetMergeAuthorityCodec.New(tamperedContract); err == nil ||
		!strings.Contains(err.Error(), "audited incoming authority digest differs") {
		t.Fatalf("tampered incoming audit error = %v", err)
	}
	assertInvalidAudit := func(
		name, want string,
		mutate func(*FirstParentTargetMergeAuthority),
	) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			invalid := receipt
			invalid.AuditedIncomingCompletions = slices.Clone(receipt.AuditedIncomingCompletions)
			invalid.AuditedIncomingRetirements = slices.Clone(receipt.AuditedIncomingRetirements)
			mutate(&invalid)
			invalid.ID = artifact.ID{}
			if _, invalidErr := firstParentTargetMergeAuthorityCodec.New(invalid); invalidErr == nil ||
				!strings.Contains(invalidErr.Error(), want) {
				t.Fatalf("invalid source audit error = %v, want %q", invalidErr, want)
			}
		})
	}
	assertInvalidAudit("missing policy", "lacks source-audit-only policy", func(value *FirstParentTargetMergeAuthority) {
		value.SourceAuthorityPolicy = ""
	})
	assertInvalidAudit("missing completion", "retirement evidence differs from its completion", func(value *FirstParentTargetMergeAuthority) {
		value.AuditedIncomingCompletions = []ProjectedCompletionEvidence{}
	})
	assertInvalidAudit("duplicate completion", "completion evidence is not uniquely sorted", func(value *FirstParentTargetMergeAuthority) {
		value.AuditedIncomingCompletions = append(
			value.AuditedIncomingCompletions, value.AuditedIncomingCompletions[0],
		)
	})
	assertInvalidAudit("unsorted completion", "completion evidence is not uniquely sorted", func(value *FirstParentTargetMergeAuthority) {
		extra := value.AuditedIncomingCompletions[0]
		extra.Reference = "alpha/do"
		extra.RetiredItem = false
		value.AuditedIncomingCompletions = append(value.AuditedIncomingCompletions, extra)
	})
	assertInvalidAudit("missing retirement", "retired completion lacks its exact item authority", func(value *FirstParentTargetMergeAuthority) {
		value.AuditedIncomingRetirements = []ProjectedRetirementEvidence{}
	})
	assertInvalidAudit("orphan retirement", "invalid retirement evidence", func(value *FirstParentTargetMergeAuthority) {
		value.AuditedIncomingRetirements[0].Item = "orphan"
	})
	assertInvalidAudit("duplicate retirement", "retirement evidence is not uniquely sorted", func(value *FirstParentTargetMergeAuthority) {
		value.AuditedIncomingRetirements = append(
			value.AuditedIncomingRetirements, value.AuditedIncomingRetirements[0],
		)
	})
	message, err := CompletionCommitMessageWithMergeAuthority(
		[]byte("reconcile independent completion"), localPlan, "target", "do",
		fixture.manifest, fixture.codeManifest, fixture.preparation.ID, fixture.preparationCommit,
		MergeProjectionFirstParentTarget, receipt.ID,
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := Save(filepath.Join(fixture.repository, filepath.FromSlash(Path)), child); err != nil {
		t.Fatal(err)
	}
	runGit(t, fixture.repository, nil, "add", "--", Path)
	runGit(t, fixture.repository, message, "commit", "-q", "-F", "-")
	fixture.completionHash = strings.TrimSpace(string(runGit(t, fixture.repository, nil, "rev-parse", "HEAD")))
	publishCompletionAttempt(
		t, fixture.store, fixture.completionHash, "target", "do", "go test ./...", "go test ./...",
		fixture.manifest, fixture.codeManifest, fixture.preparation, receipt,
	)
	if err := sourceStore.Close(); err != nil {
		t.Fatal(err)
	}
	sourceClosed = true
	if err := os.Rename(sourceStorePath, sourceStorePath+".unavailable"); err != nil {
		t.Fatal(err)
	}
	resolved, err := resolveFixture(fixture, child, "HEAD")
	if err != nil {
		t.Fatalf("source-independent reconciliation replay: %v", err)
	}
	retained := resolved.completedReferences["duplicate/done"]
	if retained.commit != localCompletion || retained.commit == incomingCompletion {
		t.Fatalf("reconciled authority retained commit %s, want local %s", retained.commit, localCompletion)
	}
	if !planHasItem(child, "duplicate") {
		t.Fatal("local duplicate item did not remain open")
	}
	if _, retired := resolved.retiredItems["duplicate"]; retired {
		t.Fatal("source tombstone retired the target's open duplicate item")
	}
	// The same independently completed evidence remains ambiguous outside the
	// explicit receipt-backed first-parent path.
	strictLocal, strictIncoming := localAuthority, incomingAuthority
	strictIncoming.protectionSeeds = maps.Clone(strictLocal.protectionSeeds)
	if err := VerifyProspectiveMergeAuthority(
		fixture.repository, localCompletion, incomingCompletion, localPlan, localPlan, localPlan,
		strictLocal, strictIncoming,
	); err == nil || !strings.Contains(err.Error(), "ambiguous completion authority") {
		t.Fatalf("semantic-union duplicate error = %v", err)
	}
}

func TestCompletionContractDigestBindsCanonicalItemSnapshot(t *testing.T) {
	base := Item{
		ID: "contract", Title: "item title", Owner: "owner", Status: StatusOpen,
		Steps: []Step{{
			ID: "done", Title: "step title", Status: StatusOpen, Verify: "go test ./...",
			Rationale: "reason", DependsOn: []string{"prior/done"},
			Capabilities: []string{"capability"}, Outcome: []byte(`{"metric":1}`),
		}},
	}
	baseline, err := completionContractDigest(base)
	if err != nil {
		t.Fatal(err)
	}
	mutations := map[string]func(*Item){
		"item title": func(item *Item) { item.Title = "different item title" },
		"step title": func(item *Item) { item.Steps[0].Title = "different step title" },
		"rationale":  func(item *Item) { item.Steps[0].Rationale = "different reason" },
		"dependency": func(item *Item) { item.Steps[0].DependsOn = []string{"other/done"} },
		"capability": func(item *Item) { item.Steps[0].Capabilities = []string{"other"} },
		"outcome":    func(item *Item) { item.Steps[0].Outcome = []byte(`{"metric":2}`) },
	}
	for name, mutate := range mutations {
		t.Run(name, func(t *testing.T) {
			changed := base
			changed.Steps = slices.Clone(base.Steps)
			changed.Steps[0].DependsOn = slices.Clone(base.Steps[0].DependsOn)
			changed.Steps[0].Capabilities = slices.Clone(base.Steps[0].Capabilities)
			changed.Steps[0].Outcome = slices.Clone(base.Steps[0].Outcome)
			mutate(&changed)
			digest, digestErr := completionContractDigest(changed)
			if digestErr != nil {
				t.Fatal(digestErr)
			}
			if digest == baseline {
				t.Fatalf("%s change preserved completion contract digest", name)
			}
		})
	}
}

func TestAudioGateReplayAcceptance(t *testing.T) {
	initial := Plan{Campaign: "target merges", Doctrine: "first parent owns target", Items: []Item{
		{ID: "seed", Status: StatusOpen, Steps: []Step{{ID: "do", Status: StatusOpen, Verify: "go test ./..."}}},
		{ID: "first", Status: StatusOpen, Steps: []Step{{ID: "do", Status: StatusOpen, Verify: "go test ./..."}}},
		{ID: "second", Status: StatusOpen, Steps: []Step{{ID: "do", Status: StatusOpen, Verify: "go test ./..."}}},
	}}
	fixture := newCompletionFixture(t, initial, "seed", "do")
	fixture.commit(fixture.canonicalMessage(), true)
	target := fixture.child
	primary := strings.TrimSpace(string(runGit(
		t, fixture.repository, nil, "rev-parse", "--abbrev-ref", "HEAD",
	)))

	mergeTarget := func(source, incomingID, targetID, treeKey string, started time.Time) {
		t.Helper()
		runGit(t, fixture.repository, nil, "checkout", "-q", "-b", source)
		completedSourceID := incomingID + "-done"
		incoming := target
		incoming.Items = append(slices.Clone(target.Items),
			Item{
				ID: incomingID, Status: StatusOpen,
				Steps: []Step{{ID: "do", Status: StatusOpen, Verify: "go test ./..."}},
			},
			Item{
				ID: completedSourceID, Status: StatusOpen,
				Steps: []Step{{ID: "do", Status: StatusOpen, Verify: "go test ./..."}},
			},
		)
		if err := Save(filepath.Join(fixture.repository, filepath.FromSlash(Path)), incoming); err != nil {
			t.Fatal(err)
		}
		runGit(t, fixture.repository, nil, "add", "--", Path)
		runGit(t, fixture.repository, nil, "commit", "-q", "-m", "source-only plan work")
		sourceStorePath := filepath.Join(t.TempDir(), "source-store")
		if _, _, err := fixture.store.Backup(sourceStorePath); err != nil {
			t.Fatal(err)
		}
		sourceStore, err := overgodb.Open(sourceStorePath)
		if err != nil {
			t.Fatal(err)
		}
		sourceClosed := false
		t.Cleanup(func() {
			if !sourceClosed {
				_ = sourceStore.Close()
			}
		})
		sourceMarker := testutil.ArtifactID(t, artifact.KindEvidence, "source-store-"+targetID)
		if _, err := sourceStore.Commit(t.Context(), artifact.Batch{
			Key: "source/divergence/" + targetID, Artifacts: []artifact.Descriptor{{ID: sourceMarker}},
		}); err != nil {
			t.Fatal(err)
		}
		sourceTreeDigit := "d"
		if targetID == "second" {
			sourceTreeDigit = "e"
		}
		sourcePreparation, err := runrecord.NewGatePreparation(
			strings.Repeat(sourceTreeDigit, 64), fixture.preparation.Environment, started.Add(time.Nanosecond),
		)
		if err != nil {
			t.Fatal(err)
		}
		sourcePreparationContent, err := sourcePreparation.Content()
		if err != nil {
			t.Fatal(err)
		}
		sourcePreparationCommit, err := sourceStore.Commit(t.Context(), artifact.Batch{
			Key:      "source/prepared/" + completedSourceID,
			Contents: []artifact.Content{sourcePreparationContent}, Lineage: sourcePreparation.Lineage(),
		})
		if err != nil {
			t.Fatal(err)
		}
		sourceChild, err := Advance(incoming, completedSourceID, "do")
		if err != nil {
			t.Fatal(err)
		}
		sourceCodeManifest := testutil.ArtifactID(t, artifact.KindProfile, "source-code-"+targetID)
		sourceManifest := completionManifestPlan(t, sourceCodeManifest, sourcePreparation.TreeKey).ID
		sourceMessage, err := CompletionCommitMessage(
			[]byte("complete source-owned row"), incoming, completedSourceID, "do",
			sourceManifest, sourceCodeManifest, sourcePreparation.ID, sourcePreparationCommit,
		)
		if err != nil {
			t.Fatal(err)
		}
		if err := Save(filepath.Join(fixture.repository, filepath.FromSlash(Path)), sourceChild); err != nil {
			t.Fatal(err)
		}
		runGit(t, fixture.repository, nil, "add", "--", Path)
		runGit(t, fixture.repository, sourceMessage, "commit", "-q", "-F", "-")
		incomingRevision := strings.TrimSpace(string(runGit(
			t, fixture.repository, nil, "rev-parse", "HEAD",
		)))
		publishCompletionAttempt(
			t, sourceStore, incomingRevision, completedSourceID, "do", "go test ./...", "go test ./...",
			sourceManifest, sourceCodeManifest, sourcePreparation,
		)
		incoming = sourceChild
		runGit(t, fixture.repository, nil, "checkout", "-q", primary)
		localRevision := strings.TrimSpace(string(runGit(
			t, fixture.repository, nil, "rev-parse", "HEAD",
		)))
		runGit(t, fixture.repository, nil, "merge", "--no-ff", "--no-commit", source)
		mergeBaseRevision := strings.TrimSpace(string(runGit(
			t, fixture.repository, nil, "merge-base", localRevision, incomingRevision,
		)))

		child, err := Advance(target, targetID, "do")
		if err != nil {
			t.Fatal(err)
		}
		fixture.preAdvance, fixture.child = target, child
		fixture.item, fixture.step = targetID, "do"
		fixture.verify, fixture.acceptance = "go test ./...", "go test ./..."
		fixture.codeManifest = testutil.ArtifactID(t, artifact.KindProfile, "target-code-"+targetID)
		fixture.rotatePreparation(treeKey, started)
		if err := Save(filepath.Join(fixture.repository, filepath.FromSlash(Path)), child); err != nil {
			t.Fatal(err)
		}
		runGit(t, fixture.repository, nil, "add", "--", Path)
		manifestPlan := completionManifestPlan(t, fixture.codeManifest, fixture.preparation.TreeKey)
		fixture.manifest = manifestPlan.ID
		localAuthority, err := ResolveCompletionAuthority(
			t.Context(), fixture.repository, localRevision, target, fixture.store,
		)
		if err != nil {
			t.Fatal(err)
		}
		incomingAuthority, err := ResolveCompletionAuthority(
			t.Context(), fixture.repository, incomingRevision, incoming, sourceStore,
		)
		if err != nil {
			t.Fatal(err)
		}
		mergeAuthority, err := NewFirstParentTargetMergeAuthority(
			t.Context(), fixture.repository, localRevision, incomingRevision, mergeBaseRevision,
			target, incoming, target, target, child, targetID, "do",
			fixture.preparation.ID, fixture.preparationCommit,
			localAuthority, incomingAuthority, fixture.store, sourceStore,
		)
		if err != nil {
			t.Fatal(err)
		}
		expectedCompletions, expectedRetirements, err := completionAuthoritySnapshot(incomingAuthority)
		if err != nil {
			t.Fatal(err)
		}
		if mergeAuthority.SourceAuthorityPolicy != FirstParentTargetSourceAuditOnlyPolicy ||
			!slices.Equal(mergeAuthority.AuditedIncomingCompletions, expectedCompletions) ||
			!slices.Equal(mergeAuthority.AuditedIncomingRetirements, expectedRetirements) {
			t.Fatalf("incoming authority audit = %+v", mergeAuthority)
		}
		if targetID == "first" {
			tampered := mergeAuthority
			tampered.ChildPlanDigest = strings.Repeat("0", 64)
			if err := VerifyFirstParentTargetMergeAuthorityTransition(
				localRevision, incomingRevision, mergeBaseRevision,
				target, incoming, target, target, child, targetID, "do",
				fixture.preparation.ID, fixture.preparationCommit, tampered,
			); err == nil {
				t.Fatal("tampered projected merge receipt was accepted")
			}
			sourceIndex := slices.IndexFunc(incoming.Items, func(item Item) bool { return item.ID == incomingID })
			if sourceIndex < 0 {
				t.Fatal("source-only row is absent from incoming plan")
			}
			adoptedPreAdvance := target
			adoptedPreAdvance.Items = append(slices.Clone(target.Items), incoming.Items[sourceIndex])
			adoptedChild, advanceErr := Advance(adoptedPreAdvance, targetID, "do")
			if advanceErr != nil {
				t.Fatal(advanceErr)
			}
			adoptedReceipt := mergeAuthority
			adoptedReceipt.PreAdvancePlanDigest = planDigestText(adoptedPreAdvance)
			adoptedReceipt.ChildPlanDigest = planDigestText(adoptedChild)
			adoptedReceipt.ID = artifact.ID{}
			adoptedReceipt, advanceErr = firstParentTargetMergeAuthorityCodec.New(adoptedReceipt)
			if advanceErr != nil {
				t.Fatal(advanceErr)
			}
			if err := VerifyFirstParentTargetMergeAuthorityTransition(
				localRevision, incomingRevision, mergeBaseRevision,
				target, incoming, target, adoptedPreAdvance, adoptedChild, targetID, "do",
				fixture.preparation.ID, fixture.preparationCommit, adoptedReceipt,
			); err == nil || !strings.Contains(err.Error(), "adopts rows outside") {
				t.Fatalf("source row survived target completion: %v", err)
			}
			ephemeral := target
			ephemeral.Items = append(slices.Clone(target.Items), Item{
				ID: "merge-boundary", Status: StatusOpen,
				Steps: []Step{{
					ID: "do", Status: StatusOpen, Verify: "go test ./...",
					DependsOn: []string{completedSourceID + "/do"},
				}},
			})
			if err := VerifyFirstParentTargetLocalAuthority(
				fixture.repository, localRevision, target, ephemeral, target, localAuthority,
			); err == nil || !strings.Contains(err.Error(), "pruned dependency "+completedSourceID+"/do") {
				t.Fatalf("audited source completion authorized merge-boundary row: %v", err)
			}
			incoherent := mergeAuthority
			incoherent.AuditedIncomingRetirements = slices.Clone(mergeAuthority.AuditedIncomingRetirements)
			incoherent.AuditedIncomingRetirements[0].Evidence.ContractDigest = strings.Repeat("a", 64)
			incoherent.ID = artifact.ID{}
			if _, err := firstParentTargetMergeAuthorityCodec.New(incoherent); err == nil ||
				!strings.Contains(err.Error(), "differs from its completion") {
				t.Fatalf("incoherent incoming retirement audit error = %v", err)
			}
		}
		message, err := CompletionCommitMessageWithMergeAuthority(
			[]byte("first-parent target merge"), target, targetID, "do",
			fixture.manifest, fixture.codeManifest, fixture.preparation.ID,
			fixture.preparationCommit, MergeProjectionFirstParentTarget,
			mergeAuthority.ID,
		)
		if err != nil {
			t.Fatal(err)
		}
		runGit(t, fixture.repository, message, "commit", "-q", "-F", "-")
		fixture.completionHash = strings.TrimSpace(string(runGit(
			t, fixture.repository, nil, "rev-parse", "HEAD",
		)))
		targetAttempt := publishCompletionAttempt(
			t, fixture.store, fixture.completionHash, fixture.item, fixture.step,
			fixture.verify, fixture.acceptance, fixture.manifest, fixture.codeManifest, fixture.preparation,
			mergeAuthority,
		)
		if err := sourceStore.Close(); err != nil {
			t.Fatal(err)
		}
		sourceClosed = true
		if err := os.Rename(sourceStorePath, sourceStorePath+".unavailable"); err != nil {
			t.Fatal(err)
		}
		target = child
		if slices.ContainsFunc(target.Items, func(item Item) bool { return item.ID == incomingID }) {
			t.Fatalf("source row %s entered target plan", incomingID)
		}
		resolver := completionAuthorityResolver{
			cache: make(map[completionAuthorityCacheKey]CompletionAuthority),
		}
		resolved, err := resolver.resolve(
			t.Context(), fixture.repository, "HEAD", target, fixture.store,
		)
		if err != nil {
			t.Fatalf("resolve target merge %s: %v", targetID, err)
		}
		expectedBoundaries := 2
		if targetID == "second" {
			expectedBoundaries = 3
		}
		if resolver.uncached != expectedBoundaries {
			t.Fatalf(
				"resolve target merge %s derived %d uncached boundaries, want %d",
				targetID, resolver.uncached, expectedBoundaries,
			)
		}
		if resolver.revisionLoads != 1 {
			t.Fatalf("nested immutable parents repeated external revision resolution %d times", resolver.revisionLoads)
		}
		known := resolver.transitions[resolved.repository]
		if len(known) == 0 || resolver.transitionLoads != len(known) {
			t.Fatalf("reconstructed transitions=%d, distinct immutable transitions=%d", resolver.transitionLoads, len(known))
		}
		proofs := resolver.evidenceLoads
		if proofs != len(resolved.completedReferences) || proofs != len(resolver.evidence) {
			t.Fatalf("verified completion proofs=%d, distinct completions=%d, retained proofs=%d", proofs, len(resolved.completedReferences), len(resolver.evidence))
		}
		if _, err := resolver.resolve(t.Context(), fixture.repository, "HEAD", target, fixture.store); err != nil || resolver.transitionLoads != len(known) {
			t.Fatalf("repeat replay reconstructed immutable history: loads=%d err=%v", resolver.transitionLoads, err)
		}
		if resolver.evidenceLoads != proofs {
			t.Fatal("repeat replay reverified unchanged completion proofs")
		}
		cancelled, cancel := context.WithCancelCause(t.Context())
		cancel(context.Canceled)
		if _, err := resolver.resolve(cancelled, fixture.repository, "HEAD", target, fixture.store); !errors.Is(err, context.Canceled) {
			t.Fatalf("cached resolution ignored cancellation: %v", err)
		}
		t.Logf("target=%s verified boundaries=%d reconstructed transitions=%d (one per distinct commit)", targetID, resolver.uncached, resolver.transitionLoads)
		if !resolved.completed("seed/do") {
			t.Fatal("first-parent ancestor completion authority was lost")
		}
		if resolved.completed(completedSourceID + "/do") {
			t.Fatalf("source completion %s entered target authority", completedSourceID)
		}
		if _, retired := resolved.retiredItems[completedSourceID]; retired {
			t.Fatalf("source retirement %s entered target authority", completedSourceID)
		}
		if targetID == "first" {
			missingRepository := t.TempDir()
			runGit(t, missingRepository, nil, "init", "-q")
			if _, err := resolver.transitionPlans(t.Context(), missingRepository, []gitCompletionMessage{{
				hash: fixture.completionHash, parents: []string{localRevision, incomingRevision}, message: string(message),
			}}); err == nil {
				t.Fatal("transition reuse crossed repository identity")
			}
			t.Run("cached-history-rewrite-refused", func(t *testing.T) {
				t.Setenv("GIT_REPLACE_REF_BASE", "refs/alternate-replacements/")
				if _, err := resolver.resolve(t.Context(), fixture.repository, "HEAD", target, fixture.store); err == nil ||
					!strings.Contains(err.Error(), "custom replacement-ref namespace") {
					t.Fatalf("cached authority bypassed history integrity: %v", err)
				}
			})
			dependent := target
			dependent.Items = slices.Clone(target.Items)
			secondIndex := slices.IndexFunc(dependent.Items, func(item Item) bool { return item.ID == "second" })
			if secondIndex < 0 {
				t.Fatal("second target row is absent")
			}
			dependent.Items[secondIndex].Steps = slices.Clone(dependent.Items[secondIndex].Steps)
			dependent.Items[secondIndex].Steps[0].DependsOn = []string{completedSourceID + "/do"}
			if _, err := resolver.resolve(t.Context(), fixture.repository, "HEAD", dependent, fixture.store); err == nil ||
				!strings.Contains(err.Error(), "pruned dependency "+completedSourceID+"/do") {
				t.Fatalf("source completion satisfied target dependency: %v", err)
			}
		}
		if targetID == "second" {
			late := mergeAuthority
			late.ID = artifact.ID{}
			late.SourceStore = late.TargetStore
			late, err = firstParentTargetMergeAuthorityCodec.New(late)
			if err != nil {
				t.Fatal(err)
			}
			lateContent, err := late.Content()
			if err != nil {
				t.Fatal(err)
			}
			if _, err := fixture.store.Commit(t.Context(), artifact.Batch{
				Key: "target/late-duplicate-receipt", Contents: []artifact.Content{lateContent},
				Lineage: append(late.Lineage(), artifact.Lineage{
					Child: targetAttempt.Result, Parent: late.ID, Relation: artifact.RelationDependsOn,
				}),
			}); err != nil {
				t.Fatal(err)
			}
			if _, err := resolver.resolve(t.Context(), fixture.repository, "HEAD", target, fixture.store); err == nil ||
				!strings.Contains(err.Error(), "one exact projected merge receipt lineage") {
				t.Fatalf("late duplicate receipt error = %v", err)
			}
			if resolver.evidenceLoads <= proofs {
				t.Fatal("changed store head reused old completion verification")
			}
		}
	}

	mergeTarget("source-one", "incoming-one", "first", strings.Repeat("b", 64), time.Unix(2, 0))
	mergeTarget("source-two", "incoming-two", "second", strings.Repeat("c", 64), time.Unix(3, 0))
}
