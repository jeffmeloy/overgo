package plan

import (
	"context"
	"errors"
	"strings"

	"overgo/internal/worklease"
)

// AcceptedProgressSince reports newly gated local prerequisites or sibling
// substeps for an open row. It derives progress from the existing completion
// authority, never from a second ledger or the mere movement of HEAD.
func (authority CompletionAuthority) AcceptedProgressSince(ctx context.Context, document Plan, itemID, stepID, checkpoint string) (bool, error) {
	if ctx == nil || !authority.resolves(document) || !worklease.ValidCommit(checkpoint) {
		return false, errors.New("plan: progress requires resolved completion authority and an exact checkpoint")
	}
	item, found := exactPlanItem(document, itemID)
	if !found {
		return false, errors.New("plan: progress item is absent")
	}
	step, found := exactPlanStep(document, itemID, stepID)
	if !found || step.Status != StatusOpen {
		return false, errors.New("plan: progress step is not open")
	}
	// A merge may import accepted work, but it is not local implementation
	// progress. Require the checkpoint on this exact first-parent chain and
	// consider only subsequent non-merge completion commits on that chain.
	raw, err := gitCompletionCommand(ctx, authority.repository, "rev-list", "--first-parent", authority.revision)
	if err != nil {
		return false, err
	}
	newCommits := map[string]bool{}
	reached := false
	for revision := range strings.FieldsSeq(string(raw)) {
		if revision == checkpoint {
			reached = true
			break
		}
		newCommits[revision] = true
	}
	if !reached {
		return false, errors.New("plan: progress checkpoint is outside the first-parent history")
	}
	dependencies := map[string][]string{}
	for reference, evidence := range authority.completedReferences {
		dependencies[reference] = strings.Fields(evidence.dependencies)
	}
	for _, retained := range document.Items {
		for _, candidate := range retained.Steps {
			dependencies[retained.ID+"/"+candidate.ID] = candidate.DependsOn
		}
	}
	relevant := map[string]bool{}
	queue := append([]string(nil), step.DependsOn...)
	for len(queue) > 0 {
		reference := queue[0]
		queue = queue[1:]
		if relevant[reference] {
			continue
		}
		relevant[reference] = true
		queue = append(queue, dependencies[reference]...)
	}
	for reference, evidence := range authority.completedReferences {
		sibling := strings.HasPrefix(reference, itemID+"/")
		if newCommits[evidence.commit] && !evidence.merge && evidence.owner == item.Owner && (sibling || relevant[reference]) {
			return true, nil
		}
	}
	return false, nil
}
