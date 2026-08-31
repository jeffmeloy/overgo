package main

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/closureledger"
	"overgo/internal/overgodb"
)

func TestReviewedAliasRestoreAuthorityRequiresTypedBatchProof(t *testing.T) {
	t.Run("typed restore", func(t *testing.T) {
		fixture := newClosureAliasRestoreFixture(t, restoreAliasesValid)
		selector := fixture.selector()
		dryRun, err := restoreClosureAliasesAtReviewedHead(
			t.Context(), fixture.storePath, fixture.snapshot, selector,
		)
		if err != nil {
			t.Fatal(err)
		}
		selector.Confirm = true
		selector.ExpectedAliases = dryRun.DesiredAliases
		selector.ExpectedChanges = dryRun.Changes
		selector.ExpectedStale = dryRun.PredictedStale
		selector.ExpectedAuthorityDigest = dryRun.AuthorityDigest
		if _, err := restoreClosureAliasesAtReviewedHead(
			t.Context(), fixture.storePath, fixture.snapshot, selector,
		); err != nil {
			t.Fatal(err)
		}

		analysis := analyzeClosureStore(t, fixture.storePath)
		removed := closureledger.ActiveAliasPrefix + "c"
		disposition, found := analysis.Dispositions[removed]
		if !found || disposition.Blocker != "reviewed-head-restore" {
			t.Fatalf("typed restore disposition = (%+v, found=%t)", disposition, found)
		}
		commit := analysis.History.latest[removed].Commit
		if !analysis.History.reviewed.authenticates(commit, closureRestoreReviewedHeadOperation) {
			t.Fatalf("typed restore commit is not authenticated: %+v", commit)
		}
		if len(analysis.Reactivations) != 0 {
			t.Fatalf("typed restore activation was not exempted: %+v", analysis.Reactivations)
		}
	})

	t.Run("shaped key", func(t *testing.T) {
		fixture := setupClosureChainFixture(t)
		store, err := overgodb.Open(fixture.storePath)
		if err != nil {
			t.Fatal(err)
		}
		_, commitErr := store.Commit(t.Context(), artifact.Batch{
			Key: string(closureRestoreReviewedHeadOperation) + strings.Repeat("a", 64),
			Aliases: []artifact.AliasBinding{{
				Name: fixture.nextAlias, Target: fixture.successor.ID,
				Previous: artifact.IDPointer(fixture.successor.ID), Remove: true,
			}},
		})
		closeErr := store.Close()
		if commitErr != nil || closeErr != nil {
			t.Fatalf("append shaped restore = (%v, close=%v)", commitErr, closeErr)
		}

		analysis := analyzeClosureStore(t, fixture.storePath)
		disposition := analysis.Dispositions[fixture.nextAlias]
		if disposition.Blocker != "missing-successor" || disposition.Blocker == "reviewed-head-restore" {
			t.Fatalf("shaped restore disposition = %+v", disposition)
		}
		commit := analysis.History.latest[fixture.nextAlias].Commit
		if analysis.History.reviewed.authenticates(commit, closureRestoreReviewedHeadOperation) {
			t.Fatalf("shaped restore key gained authority: %+v", commit)
		}
	})

	t.Run("forged typed target", func(t *testing.T) {
		fixture := newClosureAliasRestoreFixture(t, restoreAliasTargetsNonClosure)
		store, err := overgodb.Open(fixture.storePath)
		if err != nil {
			t.Fatal(err)
		}
		desired, err := closureAliasMapAt(t.Context(), store, fixture.sourceSequence)
		if err != nil {
			t.Fatal(err)
		}
		current, err := currentClosureAliasMap(t.Context(), store)
		if err != nil {
			t.Fatal(err)
		}
		changes := closureAliasDelta(current, desired)
		record := closureAliasRestoreRecord{
			Version:    artifact.InitialDocumentVersion,
			SourceHead: fixture.sourceHead.String(), SourceSequence: fixture.sourceSequence,
			ReplacedHead: fixture.currentHead.String(), ReplacedSequence: fixture.currentSequence,
			DesiredAliases: len(desired), Changes: len(changes), PredictedStale: 0,
			AuthorityDigest: closureAuthorityDigest("forged non-closure target"),
		}
		content, err := artifact.JSONContent(closureAliasRestoreContract, record)
		if err != nil {
			t.Fatal(err)
		}
		expected := fixture.currentHead
		batch := artifact.Batch{ExpectedHead: &expected, Contents: []artifact.Content{content}, Aliases: changes}
		if err := bindClosureOperationKey(closureRestoreReviewedHeadOperation, &batch); err != nil {
			t.Fatal(err)
		}
		commit, commitErr := store.Commit(t.Context(), batch)
		closeErr := store.Close()
		if commitErr != nil || closeErr != nil {
			t.Fatalf("append forged typed restore = (%v, close=%v)", commitErr, closeErr)
		}

		analysis := analyzeClosureStore(t, fixture.storePath)
		if analysis.History.reviewed.authenticates(
			(overgodb.CommitView{Key: batch.Key, ID: commit, Sequence: fixture.currentSequence + 1}),
			closureRestoreReviewedHeadOperation,
		) {
			t.Fatal("typed restore with a non-closure source target gained authority")
		}
	})
}

func TestReviewedOperationProofInteractionsIgnoreUnrelatedCatalogSize(t *testing.T) {
	type interactions struct {
		documents, introductions, deltas, commits int
	}
	run := func(population int) interactions {
		t.Helper()
		fixture := newClosureAliasRestoreFixture(t, restoreAliasesValid)
		selector := fixture.selector()
		dryRun, err := restoreClosureAliasesAtReviewedHead(
			t.Context(), fixture.storePath, fixture.snapshot, selector,
		)
		if err != nil {
			t.Fatal(err)
		}
		selector.Confirm = true
		selector.ExpectedAliases = dryRun.DesiredAliases
		selector.ExpectedChanges = dryRun.Changes
		selector.ExpectedStale = dryRun.PredictedStale
		selector.ExpectedAuthorityDigest = dryRun.AuthorityDigest
		result, err := restoreClosureAliasesAtReviewedHead(
			t.Context(), fixture.storePath, fixture.snapshot, selector,
		)
		if err != nil {
			t.Fatal(err)
		}
		store, err := overgodb.Open(fixture.storePath)
		if err != nil {
			t.Fatal(err)
		}
		for ordinal := range population {
			content, err := artifact.JSONContent(
				artifact.JSONContract(artifact.KindEvidence, "fixture/unrelated-reviewed-proof/v1"),
				map[string]int{"ordinal": ordinal},
			)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := store.Commit(t.Context(), artifact.Batch{
				Key:      fmt.Sprintf("fixture/unrelated-reviewed-proof/%d", ordinal),
				Contents: []artifact.Content{content},
			}); err != nil {
				t.Fatal(err)
			}
		}
		_, headSequence := store.Head()
		documents, err := allClosureDocuments(t.Context(), store)
		if err != nil {
			t.Fatal(err)
		}
		history, err := readClosureAliasEventHistory(t.Context(), store, headSequence)
		if err != nil {
			t.Fatal(err)
		}
		counted := &countedClosureReviewedStore{Store: store}
		operations, err := authenticateClosureReviewedOperations(
			t.Context(), counted, history, documents,
		)
		if err != nil {
			t.Fatal(err)
		}
		commit, err := parseClosureCommitID("fixture reviewed operation", result.Commit)
		if err != nil || operations[commit] != closureRestoreReviewedHeadOperation {
			t.Fatalf("reviewed operation = (%v, %v)", operations, err)
		}
		if err := store.Close(); err != nil {
			t.Fatal(err)
		}
		return interactions{
			documents: counted.documentReads, introductions: counted.introductionReads,
			deltas: counted.deltaReads, commits: counted.commitReads,
		}
	}

	empty := run(0)
	populated := run(64)
	if empty != populated || empty != (interactions{documents: 1, introductions: 1, deltas: 1, commits: 2}) {
		t.Fatalf("proof interactions empty=%+v populated=%+v", empty, populated)
	}
}

func TestReviewedRemediationAuthorityRequiresTypedBatchProof(t *testing.T) {
	t.Run("typed remediation", func(t *testing.T) {
		fixture := setupClosureChainFixture(t)
		commit, sequence := reactivateClosureChain(t, fixture, false)
		retireClosureChainSuccessor(t, fixture)
		selector := remediationSelectorAtHead(t, fixture.storePath, commit, sequence, true)
		if _, err := remediateClosureReactivations(t.Context(), fixture.storePath, selector); err != nil {
			t.Fatal(err)
		}

		analysis := analyzeClosureStore(t, fixture.storePath)
		disposition := analysis.Dispositions[fixture.firstAlias]
		if disposition.Blocker != "reviewed-reactivation-remediation" {
			t.Fatalf("typed remediation disposition = %+v", disposition)
		}
		operation := analysis.History.latest[fixture.firstAlias].Commit
		if !analysis.History.reviewed.authenticates(operation, closureRemediateRecoveryOperation) {
			t.Fatalf("typed remediation commit is not authenticated: %+v", operation)
		}
	})

	t.Run("shaped key", func(t *testing.T) {
		fixture := setupClosureChainFixture(t)
		_, _ = reactivateClosureChain(t, fixture, false)
		retireClosureChainSuccessor(t, fixture)
		store, err := overgodb.Open(fixture.storePath)
		if err != nil {
			t.Fatal(err)
		}
		_, commitErr := store.Commit(t.Context(), artifact.Batch{
			Key: string(closureRemediateRecoveryOperation) + strings.Repeat("b", 64),
			Aliases: []artifact.AliasBinding{{
				Name: fixture.firstAlias, Target: fixture.first.ID,
				Previous: artifact.IDPointer(fixture.first.ID), Remove: true,
			}},
		})
		closeErr := store.Close()
		if commitErr != nil || closeErr != nil {
			t.Fatalf("append shaped remediation = (%v, close=%v)", commitErr, closeErr)
		}

		analysis := analyzeClosureStore(t, fixture.storePath)
		disposition := analysis.Dispositions[fixture.firstAlias]
		if disposition.Blocker != "missing-successor" ||
			disposition.Blocker == "reviewed-reactivation-remediation" {
			t.Fatalf("shaped remediation disposition = %+v", disposition)
		}
		operation := analysis.History.latest[fixture.firstAlias].Commit
		if analysis.History.reviewed.authenticates(operation, closureRemediateRecoveryOperation) {
			t.Fatalf("shaped remediation key gained authority: %+v", operation)
		}
	})

	t.Run("forged typed tuple", func(t *testing.T) {
		fixture := setupClosureChainFixture(t)
		activation, activationSequence := reactivateClosureChain(t, fixture, false)
		retireClosureChainSuccessor(t, fixture)
		before := analyzeClosureStore(t, fixture.storePath)
		selected := make([]closureRecoveryReactivation, 0)
		for _, candidate := range before.Reactivations {
			if candidate.Commit.ID == activation && candidate.Commit.Sequence == activationSequence {
				selected = append(selected, candidate)
			}
		}
		reviewed := closureRemediationReviewedAliases(selected)
		if len(reviewed) != 1 {
			t.Fatalf("reviewed tuple fixture = %+v", reviewed)
		}
		reviewed[0].RetiredDocument = fixture.successor.ID
		head, sequence := storeCoordinates(t, fixture.storePath)
		record := closureRemediationEvidence{
			Version:            artifact.InitialDocumentVersion,
			ReactivationCommit: activation.String(), ReactivationSequence: activationSequence,
			ReplacedHead: head.String(), ReplacedSequence: sequence,
			ReviewedCount: 1, ReviewedAliases: reviewed, ConfirmUnverifiedRecovery: true,
		}
		content, err := artifact.JSONContent(closureReactivationRemediationContract, record)
		if err != nil {
			t.Fatal(err)
		}
		target := fixture.first.ID
		removals := []artifact.AliasBinding{{
			Name: fixture.firstAlias, Target: target, Previous: artifact.IDPointer(target), Remove: true,
		}}
		expected := head
		batch := artifact.Batch{ExpectedHead: &expected, Contents: []artifact.Content{content}, Aliases: removals}
		if err := bindClosureOperationKey(closureRemediateRecoveryOperation, &batch); err != nil {
			t.Fatal(err)
		}
		store, err := overgodb.Open(fixture.storePath)
		if err != nil {
			t.Fatal(err)
		}
		commit, commitErr := store.Commit(t.Context(), batch)
		closeErr := store.Close()
		if commitErr != nil || closeErr != nil {
			t.Fatalf("append forged typed remediation = (%v, close=%v)", commitErr, closeErr)
		}

		after := analyzeClosureStore(t, fixture.storePath)
		if after.History.reviewed.authenticates(
			(overgodb.CommitView{Key: batch.Key, ID: commit, Sequence: sequence + 1}),
			closureRemediateRecoveryOperation,
		) {
			t.Fatal("typed remediation with a forged reviewed tuple gained authority")
		}
	})
}

func analyzeClosureStore(t *testing.T, storePath string) closureRecoveryAnalysis {
	t.Helper()
	store, err := overgodb.OpenReadOnly(storePath)
	if err != nil {
		t.Fatal(err)
	}
	documents, documentErr := allClosureDocuments(t.Context(), store)
	analysis, analysisErr := analyzeClosureRecovery(t.Context(), store, documents)
	closeErr := store.Close()
	if documentErr != nil || analysisErr != nil || closeErr != nil {
		t.Fatalf("analyze closure store = (documents=%v analysis=%v close=%v)", documentErr, analysisErr, closeErr)
	}
	return analysis
}

type countedClosureReviewedStore struct {
	*overgodb.Store
	documentReads, introductionReads, deltaReads, commitReads int
}

func (store *countedClosureReviewedStore) VisitDocuments(
	ctx context.Context,
	query overgodb.DocumentQuery,
	visit func(overgodb.DocumentView) error,
) (overgodb.DocumentPage, error) {
	store.documentReads++
	return store.Store.VisitDocuments(ctx, query, visit)
}

func (store *countedClosureReviewedStore) ArtifactIntroduction(
	ctx context.Context,
	id artifact.ID,
) (overgodb.ArtifactIntroduction, bool, error) {
	store.introductionReads++
	return store.Store.ArtifactIntroduction(ctx, id)
}

func (store *countedClosureReviewedStore) CommitDeltaAt(
	ctx context.Context,
	sequence uint64,
) (overgodb.CommitDelta, bool, error) {
	store.deltaReads++
	return store.Store.CommitDeltaAt(ctx, sequence)
}

func (store *countedClosureReviewedStore) CommitAt(
	ctx context.Context,
	sequence uint64,
) (overgodb.CommitView, bool, error) {
	store.commitReads++
	return store.Store.CommitAt(ctx, sequence)
}
