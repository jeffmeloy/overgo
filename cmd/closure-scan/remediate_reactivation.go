package main

import (
	"cmp"
	"context"
	"encoding/hex"
	"errors"
	"fmt"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

const closureReactivationRemediationSchema = "overgo/closure-reactivation-remediation/v1"

var closureReactivationRemediationContract = artifact.JSONContract(
	artifact.KindEvidence,
	closureReactivationRemediationSchema,
)

type closureRemediationSelector struct {
	ReactivationCommit        artifact.CommitID
	ReactivationSequence      uint64
	ExpectedHead              artifact.CommitID
	ExpectedSequence          uint64
	ExpectedCount             int
	ConfirmUnverifiedRecovery bool
}

type closureRemediationResult struct {
	Reviewed int         `json:"reviewed"`
	Retired  int         `json:"retired"`
	Record   artifact.ID `json:"record"`
	Commit   string      `json:"commit"`
}

type closureRemediationEvidence struct {
	Version                   uint16                            `json:"version"`
	ReactivationCommit        string                            `json:"reactivation_commit"`
	ReactivationSequence      uint64                            `json:"reactivation_sequence"`
	ReplacedHead              string                            `json:"replaced_head"`
	ReplacedSequence          uint64                            `json:"replaced_sequence"`
	ReviewedCount             int                               `json:"reviewed_count"`
	ReviewedAliases           []closureRemediationReviewedAlias `json:"reviewed_aliases"`
	ConfirmUnverifiedRecovery bool                              `json:"confirm_unverified_recovery"`
}

type closureRemediationReviewedAlias struct {
	Alias               string      `json:"alias"`
	RetiredDocument     artifact.ID `json:"retired_document"`
	ReactivatedDocument artifact.ID `json:"reactivated_document"`
	CurrentDocument     artifact.ID `json:"current_document"`
	ActivationSequence  uint64      `json:"activation_sequence"`
	RetirementSequence  uint64      `json:"retirement_sequence"`
	CurrentSequence     uint64      `json:"current_sequence"`
	Blocker             string      `json:"blocker"`
	Provenance          string      `json:"provenance"`
}

func parseClosureRemediationSelector(
	reactivationCommit string,
	reactivationSequence uint64,
	expectedHead string,
	expectedSequence uint64,
	expectedCount int,
	confirmUnverifiedRecovery bool,
) (closureRemediationSelector, error) {
	selected, err := parseClosureCommitID("reactivation commit", reactivationCommit)
	if err != nil {
		return closureRemediationSelector{}, err
	}
	head, err := parseClosureCommitID("expected head", expectedHead)
	if err != nil {
		return closureRemediationSelector{}, err
	}
	selector := closureRemediationSelector{
		ReactivationCommit: selected, ReactivationSequence: reactivationSequence,
		ExpectedHead: head, ExpectedSequence: expectedSequence, ExpectedCount: expectedCount,
		ConfirmUnverifiedRecovery: confirmUnverifiedRecovery,
	}
	if err := selector.validate(); err != nil {
		return closureRemediationSelector{}, err
	}
	return selector, nil
}

func parseClosureCommitID(label, text string) (artifact.CommitID, error) {
	var id artifact.CommitID
	if len(text) != hex.EncodedLen(len(id)) {
		return artifact.CommitID{}, fmt.Errorf("closure-scan: invalid %s", label)
	}
	if _, err := hex.Decode(id[:], []byte(text)); err != nil || !id.Valid() || id.String() != text {
		return artifact.CommitID{}, fmt.Errorf("closure-scan: invalid %s", label)
	}
	return id, nil
}

func (selector closureRemediationSelector) validate() error {
	if !selector.ReactivationCommit.Valid() || selector.ReactivationSequence == 0 ||
		!selector.ExpectedHead.Valid() || selector.ExpectedSequence == 0 || selector.ExpectedCount <= 0 {
		return errors.New("closure-scan: invalid reactivation remediation selector")
	}
	if selector.ReactivationSequence > selector.ExpectedSequence {
		return errors.New("closure-scan: reactivation sequence exceeds expected store head")
	}
	return nil
}

// remediateClosureReactivations performs one bounded reviewed repair. Open
// holds the store's writer lock from head validation through the CAS append;
// ExpectedHead is repeated on the batch so in-process movement also fails.
func remediateClosureReactivations(
	ctx context.Context,
	storePath string,
	selector closureRemediationSelector,
) (result closureRemediationResult, finalErr error) {
	if ctx == nil {
		return result, errors.New("closure-scan: nil remediation context")
	}
	if err := selector.validate(); err != nil {
		return result, err
	}
	store, err := overgodb.Open(storePath)
	if err != nil {
		return result, err
	}
	defer func() {
		finalErr = closureRemediationFinalError(result.Commit != "", finalErr, store.Close())
	}()
	head, sequence := store.Head()
	if head != selector.ExpectedHead || sequence != selector.ExpectedSequence {
		return result, fmt.Errorf(
			"closure-scan: remediation store head is stale: have %s@%d, reviewed %s@%d",
			head, sequence, selector.ExpectedHead, selector.ExpectedSequence,
		)
	}
	documents, err := allClosureDocuments(ctx, store)
	if err != nil {
		return result, err
	}
	analysis, err := analyzeClosureRecovery(ctx, store, documents)
	if err != nil {
		return result, err
	}
	if currentHead, currentSequence := store.Head(); currentHead != head || currentSequence != sequence {
		return result, errors.New("closure-scan: remediation head moved during audit")
	}
	selected := make([]closureRecoveryReactivation, 0)
	coordinateConflict := false
	for _, reactivation := range analysis.Reactivations {
		sameCommit := reactivation.Commit.ID == selector.ReactivationCommit
		sameSequence := reactivation.Commit.Sequence == selector.ReactivationSequence
		if sameCommit != sameSequence {
			coordinateConflict = true
		}
		if sameCommit && sameSequence {
			selected = append(selected, reactivation)
		}
	}
	if coordinateConflict {
		return result, errors.New("closure-scan: reactivation selector commit and sequence disagree")
	}
	if len(selected) == 0 {
		return result, errors.New("closure-scan: selected reactivation has no active audit candidates")
	}
	if len(selected) != selector.ExpectedCount {
		return result, fmt.Errorf(
			"closure-scan: selected reactivation count changed: have %d, reviewed %d",
			len(selected), selector.ExpectedCount,
		)
	}
	slices.SortFunc(selected, func(left, right closureRecoveryReactivation) int {
		return cmp.Compare(left.Alias, right.Alias)
	})
	removals := make([]artifact.AliasBinding, 0, len(selected))
	seen := make(map[string]bool, len(selected))
	for _, reactivation := range selected {
		if seen[reactivation.Alias] {
			return result, fmt.Errorf("closure-scan: duplicate remediation alias %s", reactivation.Alias)
		}
		seen[reactivation.Alias] = true
		if reactivation.Eligible {
			return result, fmt.Errorf("closure-scan: selected reactivation %s is proven eligible", reactivation.Alias)
		}
		if reactivation.Blocker == "" {
			return result, fmt.Errorf("closure-scan: selected reactivation %s has ambiguous eligibility", reactivation.Alias)
		}
		switch reactivation.Provenance {
		case "rebind-operation-unverified", "legacy-operation-unverified":
			if !selector.ConfirmUnverifiedRecovery {
				return result, fmt.Errorf(
					"closure-scan: selected reactivation %s has unverified operation provenance; require explicit confirmation",
					reactivation.Alias,
				)
			}
		default:
			return result, fmt.Errorf("closure-scan: selected reactivation %s has unknown provenance", reactivation.Alias)
		}
		latest, found := analysis.History.latest[reactivation.Alias]
		if !found || latest.Binding.Remove || latest.Binding.Target != reactivation.CurrentDocument ||
			latest.Commit.Sequence != reactivation.CurrentSequence {
			return result, fmt.Errorf("closure-scan: selected reactivation %s active tuple changed", reactivation.Alias)
		}
		target := latest.Binding.Target
		removals = append(removals, artifact.AliasBinding{
			Name: reactivation.Alias, Target: target, Previous: artifact.IDPointer(target), Remove: true,
		})
	}
	result.Reviewed = len(removals)
	evidence := closureRemediationEvidence{
		Version:                   artifact.InitialDocumentVersion,
		ReactivationCommit:        selector.ReactivationCommit.String(),
		ReactivationSequence:      selector.ReactivationSequence,
		ReplacedHead:              selector.ExpectedHead.String(),
		ReplacedSequence:          selector.ExpectedSequence,
		ReviewedCount:             len(removals),
		ReviewedAliases:           closureRemediationReviewedAliases(selected),
		ConfirmUnverifiedRecovery: selector.ConfirmUnverifiedRecovery,
	}
	content, err := artifact.JSONContent(closureReactivationRemediationContract, evidence)
	if err != nil {
		return result, err
	}
	expectedHead := selector.ExpectedHead
	batch := artifact.Batch{
		ExpectedHead: &expectedHead,
		Contents:     []artifact.Content{content},
		Aliases:      removals,
	}
	if err := bindClosureOperationKey(closureRemediateRecoveryOperation, &batch); err != nil {
		return result, err
	}
	commit, err := store.Commit(ctx, batch)
	if err != nil {
		return result, err
	}
	result.Record = content.Descriptor.ID
	result.Commit = commit.String()
	result.Retired = len(removals)
	return result, nil
}

func closureRemediationReviewedAliases(
	selected []closureRecoveryReactivation,
) []closureRemediationReviewedAlias {
	reviewed := make([]closureRemediationReviewedAlias, 0, len(selected))
	for _, reactivation := range selected {
		reviewed = append(reviewed, closureRemediationReviewedAlias{
			Alias:               reactivation.Alias,
			RetiredDocument:     reactivation.RetiredDocument,
			ReactivatedDocument: reactivation.ReactivatedDocument,
			CurrentDocument:     reactivation.CurrentDocument,
			ActivationSequence:  reactivation.Commit.Sequence,
			RetirementSequence:  reactivation.RetirementSequence,
			CurrentSequence:     reactivation.CurrentSequence,
			Blocker:             reactivation.Blocker,
			Provenance:          reactivation.Provenance,
		})
	}
	slices.SortFunc(reviewed, func(left, right closureRemediationReviewedAlias) int {
		return cmp.Compare(left.Alias, right.Alias)
	})
	return reviewed
}

func closureRemediationFinalError(committed bool, operationErr, closeErr error) error {
	if committed {
		return nil
	}
	return errors.Join(operationErr, closeErr)
}
