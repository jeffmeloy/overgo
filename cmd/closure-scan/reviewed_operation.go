package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/closureledger"
	"overgo/internal/overgodb"
	"overgo/internal/strictjson"
)

// closureReviewedOperations contains only operation commits whose typed review
// record and exact mutation batch reproduce the durable operation key.
type closureReviewedOperations map[artifact.CommitID]closureCommitOperation

type closureReviewedOperationStore interface {
	overgodb.DocumentReader
	ArtifactIntroduction(context.Context, artifact.ID) (overgodb.ArtifactIntroduction, bool, error)
	CommitAt(context.Context, uint64) (overgodb.CommitView, bool, error)
	CommitDeltaAt(context.Context, uint64) (overgodb.CommitDelta, bool, error)
}

func (operations closureReviewedOperations) authenticates(
	commit overgodb.CommitView,
	operation closureCommitOperation,
) bool {
	return operations[commit.ID] == operation &&
		closureOperationKey(commit.Key, string(operation))
}

func authenticateClosureReviewedOperations(
	ctx context.Context,
	store closureReviewedOperationStore,
	history closureAliasEventHistory,
	documents []closureledger.Document,
) (closureReviewedOperations, error) {
	operations := closureReviewedOperations{}
	rejected := map[artifact.CommitID]bool{}
	_, err := store.VisitDocuments(ctx, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{
			closureAliasRestoreContract,
			closureReactivationRemediationContract,
		},
		Order: overgodb.DocumentOldestFirst,
	}, func(view overgodb.DocumentView) error {
		introduction, found, err := store.ArtifactIntroduction(ctx, view.Content.Descriptor.ID)
		if err != nil {
			return err
		}
		if !found || introduction.Sequence != view.Sequence {
			return nil
		}
		committed, found, err := store.CommitDeltaAt(ctx, introduction.Sequence)
		if err != nil {
			return err
		}
		if !found || committed.Commit.ID != introduction.Commit {
			return nil
		}
		history.reviewed = operations
		operation, valid, err := authenticateClosureReviewedOperation(
			ctx, store, committed, view.Content, history, documents,
		)
		if err != nil {
			return err
		}
		commit := committed.Commit
		if !valid || rejected[commit.ID] {
			return nil
		}
		if _, exists := operations[commit.ID]; exists {
			delete(operations, commit.ID)
			rejected[commit.ID] = true
			return nil
		}
		operations[commit.ID] = operation
		return nil
	})
	return operations, err
}

func authenticateClosureReviewedOperation(
	ctx context.Context,
	store closureReviewedOperationStore,
	committed overgodb.CommitDelta,
	content artifact.Content,
	history closureAliasEventHistory,
	documents []closureledger.Document,
) (closureCommitOperation, bool, error) {
	commit := committed.Commit
	switch content.Descriptor.Schema {
	case closureAliasRestoreSchema:
		if !closureOperationKey(commit.Key, string(closureRestoreReviewedHeadOperation)) {
			return "", false, nil
		}
		valid, err := authenticateClosureAliasRestore(
			ctx, store, committed, content, history, documents,
		)
		return closureRestoreReviewedHeadOperation, valid, err
	case closureReactivationRemediationSchema:
		if !closureOperationKey(commit.Key, string(closureRemediateRecoveryOperation)) {
			return "", false, nil
		}
		valid, err := authenticateClosureReactivationRemediation(
			ctx, store, committed, content, history, documents,
		)
		return closureRemediateRecoveryOperation, valid, err
	default:
		return "", false, nil
	}
}

func authenticateClosureAliasRestore(
	ctx context.Context,
	store closureReviewedOperationStore,
	committed overgodb.CommitDelta,
	content artifact.Content,
	history closureAliasEventHistory,
	documents []closureledger.Document,
) (bool, error) {
	commit := committed.Commit
	aliases := committed.Delta.Aliases
	var record closureAliasRestoreRecord
	if strictjson.DecodeBytes(content.Data, &record) != nil ||
		!canonicalClosureReviewedContent(content, closureAliasRestoreContract, record) ||
		record.Version != artifact.InitialDocumentVersion || record.SourceSequence == 0 ||
		record.ReplacedSequence == 0 || record.SourceSequence > record.ReplacedSequence ||
		record.ReplacedSequence != commit.Sequence-1 || record.DesiredAliases < 0 ||
		record.Changes <= 0 || record.Changes != len(aliases) || record.PredictedStale != 0 ||
		!validClosureAuthorityDigest(record.AuthorityDigest) {
		return false, nil
	}
	source, sourceValid := reviewedClosureCommitID(record.SourceHead)
	replaced, replacedValid := reviewedClosureCommitID(record.ReplacedHead)
	if !sourceValid || !replacedValid {
		return false, nil
	}
	sourceExists, err := closureCommitCoordinateExists(ctx, store, source, record.SourceSequence)
	if err != nil || !sourceExists {
		return false, err
	}
	replacedExists, err := closureCommitCoordinateExists(ctx, store, replaced, record.ReplacedSequence)
	if err != nil || !replacedExists {
		return false, err
	}
	desired := closureAliasMapFromHistory(history, record.SourceSequence)
	current := closureAliasMapFromHistory(history, record.ReplacedSequence)
	if err := requireCanonicalClosureAliasTargets(desired, documents); err != nil {
		return false, nil
	}
	expected := closureAliasDelta(current, desired)
	if len(desired) != record.DesiredAliases || !equalClosureAliasBindings(expected, aliases) {
		return false, nil
	}
	return closureReviewedOperationMatches(
		closureRestoreReviewedHeadOperation, committed, replaced, content, aliases,
	), nil
}

func authenticateClosureReactivationRemediation(
	ctx context.Context,
	store closureReviewedOperationStore,
	committed overgodb.CommitDelta,
	content artifact.Content,
	history closureAliasEventHistory,
	documents []closureledger.Document,
) (bool, error) {
	commit := committed.Commit
	var record closureRemediationEvidence
	if strictjson.DecodeBytes(content.Data, &record) != nil ||
		!canonicalClosureReviewedContent(content, closureReactivationRemediationContract, record) ||
		record.Version != artifact.InitialDocumentVersion || record.ReactivationSequence == 0 ||
		record.ReplacedSequence == 0 || record.ReactivationSequence > record.ReplacedSequence ||
		record.ReplacedSequence != commit.Sequence-1 ||
		!validClosureRemediationReviewedAliases(record.ReviewedAliases, record.ReviewedCount) {
		return false, nil
	}
	reactivation, reactivationValid := reviewedClosureCommitID(record.ReactivationCommit)
	replaced, replacedValid := reviewedClosureCommitID(record.ReplacedHead)
	if !reactivationValid || !replacedValid {
		return false, nil
	}
	reactivationExists, err := closureCommitCoordinateExists(
		ctx, store, reactivation, record.ReactivationSequence,
	)
	if err != nil || !reactivationExists {
		return false, err
	}
	replacedExists, err := closureCommitCoordinateExists(ctx, store, replaced, record.ReplacedSequence)
	if err != nil || !replacedExists {
		return false, err
	}
	before := closureHistoryAtSequence(history, record.ReplacedSequence)
	analysis, err := resolveClosureRecovery(documents, before)
	if err != nil {
		return false, err
	}
	analysis.Reactivations, err = auditClosureRecoveryReactivations(documents, before)
	if err != nil {
		return false, err
	}
	selected := make([]closureRecoveryReactivation, 0, record.ReviewedCount)
	coordinateConflict := false
	for _, candidate := range analysis.Reactivations {
		sameCommit := candidate.Commit.ID == reactivation
		sameSequence := candidate.Commit.Sequence == record.ReactivationSequence
		if sameCommit != sameSequence {
			coordinateConflict = true
		}
		if sameCommit && sameSequence {
			selected = append(selected, candidate)
		}
	}
	if coordinateConflict || len(selected) != record.ReviewedCount {
		return false, nil
	}
	slices.SortFunc(selected, func(left, right closureRecoveryReactivation) int {
		return strings.Compare(left.Alias, right.Alias)
	})
	requiresConfirmation := false
	removals := make([]artifact.AliasBinding, 0, len(selected))
	for _, candidate := range selected {
		if candidate.Eligible || candidate.Blocker == "" {
			return false, nil
		}
		switch candidate.Provenance {
		case "rebind-operation-unverified", "legacy-operation-unverified":
			requiresConfirmation = true
		default:
			return false, nil
		}
		target := candidate.CurrentDocument
		removals = append(removals, artifact.AliasBinding{
			Name: candidate.Alias, Target: target, Previous: artifact.IDPointer(target), Remove: true,
		})
	}
	if requiresConfirmation && !record.ConfirmUnverifiedRecovery ||
		!slices.Equal(record.ReviewedAliases, closureRemediationReviewedAliases(selected)) ||
		!equalClosureAliasBindings(removals, committed.Delta.Aliases) {
		return false, nil
	}
	return closureReviewedOperationMatches(
		closureRemediateRecoveryOperation, committed, replaced, content, removals,
	), nil
}

func closureCommitCoordinateExists(
	ctx context.Context,
	store closureReviewedOperationStore,
	want artifact.CommitID,
	sequence uint64,
) (bool, error) {
	commit, found, err := store.CommitAt(ctx, sequence)
	return found && commit.ID == want, err
}

func reviewedClosureCommitID(value string) (artifact.CommitID, bool) {
	id, err := parseClosureCommitID("reviewed operation commit", value)
	return id, err == nil
}

func canonicalClosureReviewedContent(
	content artifact.Content,
	contract artifact.DocumentContract,
	record any,
) bool {
	canonical, err := artifact.JSONContent(contract, record)
	return err == nil && canonical.Descriptor == content.Descriptor &&
		bytes.Equal(canonical.Data, content.Data)
}

func closureReviewedOperationMatches(
	operation closureCommitOperation,
	committed overgodb.CommitDelta,
	expectedHead artifact.CommitID,
	content artifact.Content,
	aliases []artifact.AliasBinding,
) bool {
	expected := expectedHead
	logical := artifact.Batch{
		ExpectedHead: &expected,
		Contents:     []artifact.Content{content.Clone()},
		Aliases:      artifact.CloneAliasBindings(aliases),
	}
	if bindClosureOperationKey(operation, &logical) != nil || logical.Key != committed.Commit.Key {
		return false
	}
	request := logical
	request.Artifacts = []artifact.Descriptor{content.Descriptor}
	encoded, err := json.Marshal(request)
	if err != nil || sha256.Sum256(encoded) != committed.Request {
		return false
	}
	delta := committed.Delta
	return committed.Previous == expectedHead && delta.Key == committed.Commit.Key &&
		delta.ExpectedHead != nil && *delta.ExpectedHead == expectedHead &&
		len(delta.Artifacts) == 1 && delta.Artifacts[0] == content.Descriptor &&
		len(delta.Contents) == 1 && delta.Contents[0].Descriptor == content.Descriptor &&
		bytes.Equal(delta.Contents[0].Data, content.Data) &&
		equalClosureAliasBindings(delta.Aliases, aliases) &&
		len(delta.Manifests)+len(delta.Lineage)+len(delta.Causality)+len(delta.Locations) == 0
}

func equalClosureAliasBindings(left, right []artifact.AliasBinding) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index].Name != right[index].Name || left[index].Target != right[index].Target ||
			left[index].Remove != right[index].Remove ||
			!equalClosureAliasPrevious(left[index].Previous, right[index].Previous) {
			return false
		}
	}
	return true
}

func equalClosureAliasPrevious(left, right *artifact.ID) bool {
	return left == nil && right == nil || left != nil && right != nil && *left == *right
}

func validClosureRemediationReviewedAliases(
	reviewed []closureRemediationReviewedAlias,
	count int,
) bool {
	if count <= 0 || len(reviewed) != count {
		return false
	}
	for index, row := range reviewed {
		if !strings.HasPrefix(row.Alias, closureledger.ActiveAliasPrefix) ||
			strings.TrimSpace(row.Alias) != row.Alias ||
			!row.RetiredDocument.Valid() || !row.ReactivatedDocument.Valid() || !row.CurrentDocument.Valid() ||
			row.ActivationSequence == 0 || row.RetirementSequence == 0 || row.CurrentSequence == 0 ||
			row.Blocker == "" || strings.TrimSpace(row.Blocker) != row.Blocker ||
			row.Provenance == "" || strings.TrimSpace(row.Provenance) != row.Provenance ||
			index > 0 && reviewed[index-1].Alias >= row.Alias {
			return false
		}
	}
	return true
}

func closureHistoryAtSequence(
	history closureAliasEventHistory,
	sequence uint64,
) closureAliasEventHistory {
	result := closureAliasEventHistory{
		latest:   map[string]overgodb.AliasEvent{},
		byCommit: map[artifact.CommitID][]artifact.AliasBinding{},
		byAlias:  map[string][]overgodb.AliasEvent{},
		reviewed: history.reviewed,
	}
	for alias, events := range history.byAlias {
		for _, event := range events {
			if event.Commit.Sequence > sequence {
				break
			}
			result.latest[alias] = event
			result.byAlias[alias] = append(result.byAlias[alias], event)
			result.byCommit[event.Commit.ID] = append(result.byCommit[event.Commit.ID], event.Binding)
		}
	}
	return result
}

func closureAliasMapFromHistory(
	history closureAliasEventHistory,
	sequence uint64,
) map[string]artifact.ID {
	result := map[string]artifact.ID{}
	for alias, event := range closureHistoryAtSequence(history, sequence).latest {
		if !event.Binding.Remove {
			result[alias] = event.Binding.Target
		}
	}
	return result
}
