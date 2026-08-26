package plan

import (
	"bytes"
	"fmt"
	"slices"

	"overgo/internal/artifact"
)

// MergeDocuments merges retained plan rows by identity.
func MergeDocuments(base, local, upstream Plan) (Plan, error) {
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
	merged := Plan{Campaign: campaign, Doctrine: doctrine, Census: census}
	for _, id := range unionOrder(itemID, base.Items, local.Items, upstream.Items) {
		baseItem, inBase := baseItems[id]
		localItem, inLocal := localItems[id]
		upstreamItem, inUpstream := upstreamItems[id]
		if inBase && (!inLocal || !inUpstream) {
			return Plan{}, fmt.Errorf("plan document: retained item %q was deleted", id)
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
		item, err := mergeItem(baseItem, localItem, upstreamItem)
		if err != nil {
			return Plan{}, err
		}
		merged.Items = append(merged.Items, item)
	}
	return merged, Validate(merged)
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

func mergeItem(base, local, upstream Item) (Item, error) {
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
			return Item{}, fmt.Errorf("plan document: retained step %q/%q was deleted", base.ID, id)
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
		bytes.Equal(left.Outcome, right.Outcome)
}
