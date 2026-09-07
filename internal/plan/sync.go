package plan

import (
	"bytes"
	"cmp"
	"fmt"
	"reflect"
	"slices"

	"overgo/internal/artifact"
)

// MergeDocuments merges immutable plan snapshots by retained row identity.
// Historical inputs may contain lifecycle states or dependency syntax that a
// current live plan no longer accepts; callers publishing the result as live
// state still pass through Save and its current Validate rules.
func MergeDocuments(base, local, upstream Plan) (Plan, error) {
	// Absence is never proof without authority: no pruning is accepted here.
	return mergeDocuments(base, local, upstream, validatePlanGraph, func(string) bool { return false })
}

// MergeDocumentsWithCompletion merges like MergeDocuments but accepts a row
// pruned by one or both sides when the union of the parents' completion
// authorities proves that exact identity was completed through the gate and
// any retaining side left the row unchanged from base. Pruning without
// ancestor gate evidence still refuses: absence is never proof.
func MergeDocumentsWithCompletion(
	base, local, upstream Plan,
	localAuthority, upstreamAuthority CompletionAuthority,
) (Plan, error) {
	completed := func(reference string) bool {
		return localAuthority.completed(reference) || upstreamAuthority.completed(reference)
	}
	return mergeDocuments(base, local, upstream, validatePlanGraph, completed)
}

func mergeDocuments(base, local, upstream Plan, validate func(Plan) error, completed func(string) bool) (Plan, error) {
	campaign, doctrine, err := mergeProjectionHeaders(base, local, upstream)
	if err != nil {
		return Plan{}, err
	}
	census, err := mergeCensusAuthority(base.Census, local.Census, upstream.Census)
	if err != nil {
		return Plan{}, err
	}
	itemID := func(item Item) string { return item.ID }
	baseItems, localItems, upstreamItems := indexByID(base.Items, itemID), indexByID(local.Items, itemID), indexByID(upstream.Items, itemID)
	// The lane is the local plan's: a lane plan never takes another lane's identity.
	merged := Plan{Campaign: campaign, Doctrine: doctrine, Lane: cmp.Or(local.Lane, upstream.Lane), Census: census}
	for _, id := range unionOrder(itemID, base.Items, local.Items, upstream.Items) {
		baseItem, inBase := baseItems[id]
		localItem, inLocal := localItems[id]
		upstreamItem, inUpstream := upstreamItems[id]
		if inBase && (!inLocal || !inUpstream) {
			if err := acceptItemPruning(baseItem, localItem, inLocal, upstreamItem, inUpstream, completed); err != nil {
				return Plan{}, err
			}
			continue
		}
		if !inBase {
			switch {
			case inLocal && inUpstream && !sameItem(localItem, upstreamItem):
				return Plan{}, fmt.Errorf("plan projection: independently added item %q conflicts", id)
			case inLocal:
				merged.Items = append(merged.Items, localItem)
			case inUpstream:
				merged.Items = append(merged.Items, upstreamItem)
			}
			continue
		}
		item, err := mergeItem(baseItem, localItem, upstreamItem, completed)
		if err != nil {
			return Plan{}, err
		}
		if len(item.Steps) == 0 && len(baseItem.Steps) != 0 {
			// Every base step was individually pruned with completion
			// evidence; an emptied item leaves the document like the gate
			// prunes it.
			continue
		}
		merged.Items = append(merged.Items, item)
	}
	return merged, validate(merged)
}

// acceptItemPruning admits the removal of one whole base item: the identity
// is gone from at least one side, every base step must carry gated completion
// evidence, and any retaining side must have left the item unchanged from
// base so no concurrent edit is silently discarded.
func acceptItemPruning(
	baseItem Item,
	localItem Item, inLocal bool,
	upstreamItem Item, inUpstream bool,
	completed func(string) bool,
) error {
	if inLocal && !sameItem(localItem, baseItem) || inUpstream && !sameItem(upstreamItem, baseItem) {
		return fmt.Errorf("plan document: retained item %q was deleted beside concurrent edits", baseItem.ID)
	}
	for _, step := range baseItem.Steps {
		if !completed(baseItem.ID + "/" + step.ID) {
			return fmt.Errorf(
				"plan document: retained item %q was deleted; %s/%s lacks gated completion evidence",
				baseItem.ID, baseItem.ID, step.ID,
			)
		}
	}
	return nil
}

func mergeProjectionHeaders(base, local, upstream Plan) (string, string, error) {
	campaign, campaignErr := mergeText("campaign", base.Campaign, local.Campaign, upstream.Campaign)
	doctrine, doctrineErr := mergeText("doctrine", base.Doctrine, local.Doctrine, upstream.Doctrine)
	if campaignErr == nil && doctrineErr == nil {
		return campaign, doctrine, nil
	}
	switch {
	case len(local.Items) != 0 && len(upstream.Items) == 0:
		return local.Campaign, local.Doctrine, nil
	case len(upstream.Items) != 0 && len(local.Items) == 0, len(local.Items) == 0 && len(upstream.Items) == 0:
		return upstream.Campaign, upstream.Doctrine, nil
	case campaignErr != nil:
		return "", "", campaignErr
	default:
		return "", "", doctrineErr
	}
}

func mergeCensusAuthority(base, local, upstream *artifact.ID) (*artifact.ID, error) {
	equal := func(left, right *artifact.ID) bool {
		return left == nil && right == nil || left != nil && right != nil && *left == *right
	}
	var selected *artifact.ID
	switch {
	case equal(local, upstream), equal(upstream, base):
		selected = local
	case equal(local, base):
		selected = upstream
	default:
		return nil, fmt.Errorf("plan projection: concurrent census authority edits conflict")
	}
	if selected == nil {
		return nil, nil
	}
	cloned := *selected
	return &cloned, nil
}

func mergeItem(base, local, upstream Item, completed func(string) bool) (Item, error) {
	title, err := mergeText("item "+base.ID+" title", base.Title, local.Title, upstream.Title)
	if err != nil {
		return Item{}, err
	}
	owner, err := mergeText("item "+base.ID+" owner", base.Owner, local.Owner, upstream.Owner)
	if err != nil {
		return Item{}, err
	}
	status, err := mergeText("item "+base.ID+" status", base.Status, local.Status, upstream.Status)
	if err != nil {
		return Item{}, err
	}
	stepID := func(step Step) string { return step.ID }
	baseSteps, localSteps, upstreamSteps := indexByID(base.Steps, stepID), indexByID(local.Steps, stepID), indexByID(upstream.Steps, stepID)
	merged := Item{ID: base.ID, Title: title, Owner: owner, Status: status}
	for _, id := range unionOrder(stepID, base.Steps, local.Steps, upstream.Steps) {
		baseStep, inBase := baseSteps[id]
		localStep, inLocal := localSteps[id]
		upstreamStep, inUpstream := upstreamSteps[id]
		if inBase && (!inLocal || !inUpstream) {
			if inLocal && !sameStep(localStep, baseStep) || inUpstream && !sameStep(upstreamStep, baseStep) {
				return Item{}, fmt.Errorf("plan document: retained step %q/%q was deleted beside concurrent edits", base.ID, id)
			}
			if !completed(base.ID + "/" + id) {
				return Item{}, fmt.Errorf(
					"plan document: retained step %q/%q was deleted without gated completion evidence", base.ID, id,
				)
			}
			continue
		}
		if !inBase {
			if inLocal && inUpstream && !sameStep(localStep, upstreamStep) {
				return Item{}, fmt.Errorf("plan projection: independently added step %q/%q conflicts", base.ID, id)
			}
			if inLocal {
				merged.Steps = append(merged.Steps, localStep)
			} else if inUpstream {
				merged.Steps = append(merged.Steps, upstreamStep)
			}
			continue
		}
		step := Step{ID: id}
		if step.Title, err = mergeText("step "+base.ID+"/"+id+" title", baseStep.Title, localStep.Title, upstreamStep.Title); err != nil {
			return Item{}, err
		}
		if step.Status, err = mergeText("step "+base.ID+"/"+id+" status", baseStep.Status, localStep.Status, upstreamStep.Status); err != nil {
			return Item{}, err
		}
		if step.Verify, err = mergeText("step "+base.ID+"/"+id+" verify", baseStep.Verify, localStep.Verify, upstreamStep.Verify); err != nil {
			return Item{}, err
		}
		if step.Rationale, err = mergeText("step "+base.ID+"/"+id+" rationale", baseStep.Rationale, localStep.Rationale, upstreamStep.Rationale); err != nil {
			return Item{}, err
		}
		if step.DependsOn, err = mergeSlice("step "+base.ID+"/"+id+" dependencies", baseStep.DependsOn, localStep.DependsOn, upstreamStep.DependsOn); err != nil {
			return Item{}, err
		}
		if step.Capabilities, err = mergeSlice("step "+base.ID+"/"+id+" capabilities", baseStep.Capabilities, localStep.Capabilities, upstreamStep.Capabilities); err != nil {
			return Item{}, err
		}
		if step.Outcome, err = mergeSlice("step "+base.ID+"/"+id+" outcome", baseStep.Outcome, localStep.Outcome, upstreamStep.Outcome); err != nil {
			return Item{}, err
		}
		switch {
		case reflect.DeepEqual(localStep.VerificationBatch, upstreamStep.VerificationBatch),
			reflect.DeepEqual(upstreamStep.VerificationBatch, baseStep.VerificationBatch):
			step.VerificationBatch = localStep.VerificationBatch
		case reflect.DeepEqual(localStep.VerificationBatch, baseStep.VerificationBatch):
			step.VerificationBatch = upstreamStep.VerificationBatch
		default:
			return Item{}, fmt.Errorf("plan document: concurrent step %s/%s verification batch edits conflict", base.ID, id)
		}
		merged.Steps = append(merged.Steps, step)
	}
	if slices.ContainsFunc(merged.Steps, func(step Step) bool { return step.Status == StatusOpen }) {
		merged.Status = StatusOpen
	} else if !slices.ContainsFunc(merged.Steps, func(step Step) bool { return step.Status != StatusDone }) {
		merged.Status = StatusDone
	}
	return merged, nil
}

func mergeText(name, base, local, upstream string) (string, error) {
	switch {
	case local == upstream || upstream == base:
		return local, nil
	case local == base:
		return upstream, nil
	default:
		return "", fmt.Errorf("plan projection: concurrent %s edits conflict", name)
	}
}

func indexByID[T any](values []T, idOf func(T) string) map[string]T {
	result := make(map[string]T, len(values))
	for _, value := range values {
		result[idOf(value)] = value
	}
	return result
}

func mergeSlice[T comparable](name string, base, local, upstream []T) ([]T, error) {
	switch {
	case slices.Equal(local, upstream) || slices.Equal(upstream, base):
		return slices.Clone(local), nil
	case slices.Equal(local, base):
		return slices.Clone(upstream), nil
	default:
		return nil, fmt.Errorf("plan document: concurrent %s edits conflict", name)
	}
}

func unionOrder[T any](idOf func(T) string, groups ...[]T) []string {
	var capacity int
	for _, values := range groups {
		capacity += len(values)
	}
	ids := make([]string, 0, capacity)
	for _, values := range groups {
		for _, value := range values {
			id := idOf(value)
			if !slices.Contains(ids, id) {
				ids = append(ids, id)
			}
		}
	}
	return ids
}

func sameItem(left, right Item) bool {
	return left.ID == right.ID && left.Title == right.Title && left.Owner == right.Owner && left.Status == right.Status &&
		slices.EqualFunc(left.Steps, right.Steps, sameStep)
}

func sameStep(left, right Step) bool {
	return left.ID == right.ID && left.Title == right.Title && left.Status == right.Status &&
		left.Verify == right.Verify && left.Rationale == right.Rationale &&
		slices.Equal(left.DependsOn, right.DependsOn) && slices.Equal(left.Capabilities, right.Capabilities) &&
		bytes.Equal(left.Outcome, right.Outcome) && reflect.DeepEqual(left.VerificationBatch, right.VerificationBatch)
}
