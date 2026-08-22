package plan

import (
	"bytes"
	"fmt"
	"slices"
)

// MergeOpenProjections performs a three-way merge of generated open-work
// projections. Removal from either descendant means the base work completed;
// independently added work survives. Conflicting edits to the same live row
// are refused instead of guessed.
func MergeOpenProjections(base, local, upstream Plan) (Plan, error) {
	if err := validateCompletionRemovals(base, local); err != nil {
		return Plan{}, err
	}
	if err := validateCompletionRemovals(base, upstream); err != nil {
		return Plan{}, err
	}
	campaign, err := mergeText("campaign", base.Campaign, local.Campaign, upstream.Campaign)
	if err != nil {
		return Plan{}, err
	}
	doctrine, err := mergeText("doctrine", base.Doctrine, local.Doctrine, upstream.Doctrine)
	if err != nil {
		return Plan{}, err
	}
	itemID := func(item Item) string { return item.ID }
	baseItems, localItems, upstreamItems := indexByID(base.Items, itemID), indexByID(local.Items, itemID), indexByID(upstream.Items, itemID)
	completed, err := mergeCompletionRefs(base.Completed, local.Completed, upstream.Completed)
	if err != nil {
		return Plan{}, err
	}
	merged := Plan{Campaign: campaign, Doctrine: doctrine, Completed: completed}
	for _, id := range unionOrder(local.Items, upstream.Items, itemID) {
		baseItem, inBase := baseItems[id]
		localItem, inLocal := localItems[id]
		upstreamItem, inUpstream := upstreamItems[id]
		if inBase && (!inLocal || !inUpstream) {
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
		item, err := mergeItem(baseItem, localItem, upstreamItem)
		if err != nil {
			return Plan{}, err
		}
		merged.Items = append(merged.Items, item)
	}
	return merged, ValidateOpenWork(merged)
}

func mergeItem(base, local, upstream Item) (Item, error) {
	title, err := mergeText("item "+base.ID+" title", base.Title, local.Title, upstream.Title)
	if err != nil {
		return Item{}, err
	}
	status, err := mergeText("item "+base.ID+" status", base.Status, local.Status, upstream.Status)
	if err != nil {
		return Item{}, err
	}
	stepID := func(step Step) string { return step.ID }
	baseSteps, localSteps, upstreamSteps := indexByID(base.Steps, stepID), indexByID(local.Steps, stepID), indexByID(upstream.Steps, stepID)
	owner, err := mergeText("item "+base.ID+" owner", base.Owner, local.Owner, upstream.Owner)
	if err != nil {
		return Item{}, err
	}
	merged := Item{ID: base.ID, Title: title, Owner: owner, Status: status}
	for _, id := range unionOrder(local.Steps, upstream.Steps, stepID) {
		baseStep, inBase := baseSteps[id]
		localStep, inLocal := localSteps[id]
		upstreamStep, inUpstream := upstreamSteps[id]
		if inBase && (!inLocal || !inUpstream) {
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
		merged.Steps = append(merged.Steps, step)
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

func unionOrder[T any](local, upstream []T, idOf func(T) string) []string {
	ids := make([]string, 0, len(local)+len(upstream))
	for _, values := range [][]T{local, upstream} {
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
	return left.ID == right.ID && left.Title == right.Title && left.Owner == right.Owner && left.Status == right.Status && slices.EqualFunc(left.Steps, right.Steps, sameStep)
}

func sameStep(left, right Step) bool {
	return left.ID == right.ID && left.Title == right.Title && left.Status == right.Status &&
		left.Verify == right.Verify && left.Rationale == right.Rationale &&
		slices.Equal(left.DependsOn, right.DependsOn) && slices.Equal(left.Capabilities, right.Capabilities) &&
		bytes.Equal(left.Outcome, right.Outcome)
}

func validateCompletionRemovals(base, descendant Plan) error {
	receipts := indexCompletionRefs(descendant.Completed)
	descendantItems := indexByID(descendant.Items, func(item Item) string { return item.ID })
	for _, baseItem := range base.Items {
		descendantItem, itemPresent := descendantItems[baseItem.ID]
		descendantSteps := indexByID(descendantItem.Steps, func(step Step) string { return step.ID })
		for _, baseStep := range baseItem.Steps {
			if itemPresent {
				if _, stepPresent := descendantSteps[baseStep.ID]; stepPresent {
					continue
				}
			}
			key := baseItem.ID + "/" + baseStep.ID
			if _, completed := receipts[key]; !completed {
				return fmt.Errorf("plan projection: removed step %q lacks completion receipt", key)
			}
		}
	}
	return nil
}

func mergeCompletionRefs(base, local, upstream []CompletionRef) ([]CompletionRef, error) {
	all := [][]CompletionRef{base, local, upstream}
	merged := map[string]CompletionRef{}
	var order []string
	for _, refs := range all {
		for _, receipt := range refs {
			key := receipt.Item + "/" + receipt.Step
			if existing, found := merged[key]; found {
				if existing != receipt {
					return nil, fmt.Errorf("plan projection: completion %q has conflicting authorities", key)
				}
				continue
			}
			merged[key] = receipt
			order = append(order, key)
		}
	}
	result := make([]CompletionRef, 0, len(order))
	for _, key := range order {
		result = append(result, merged[key])
	}
	return result, nil
}

func indexCompletionRefs(refs []CompletionRef) map[string]CompletionRef {
	indexed := make(map[string]CompletionRef, len(refs))
	for _, receipt := range refs {
		indexed[receipt.Item+"/"+receipt.Step] = receipt
	}
	return indexed
}
