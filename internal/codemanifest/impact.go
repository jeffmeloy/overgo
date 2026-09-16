package codemanifest

import (
	"cmp"
	"errors"
	"fmt"
	"maps"
	"slices"
)

// Impact is the deterministic transitive surface capable of observing a
// manifest delta.
type Impact struct {
	Base           string                `json:"base"`
	Candidate      string                `json:"candidate"`
	Seeds          []SymbolID            `json:"seeds,omitempty"`
	Reachable      []SymbolID            `json:"reachable,omitempty"`
	Packages       []string              `json:"packages,omitempty"`
	ExternalInputs []ExternalInputChange `json:"external_inputs,omitempty"`
	Uncertainty    []Uncertainty         `json:"uncertainty,omitempty"`
}

// Close computes reverse reachability over the union of the base and
// candidate graphs. This preserves observers of removed as well as added code.
func Close(base, candidate Manifest, delta Delta) (Impact, error) {
	if err := base.Validate(); err != nil {
		return Impact{}, errors.New("code manifest impact: invalid base")
	}
	if err := candidate.Validate(); err != nil {
		return Impact{}, errors.New("code manifest impact: invalid candidate")
	}
	if delta.Base != base.ID || delta.Candidate != candidate.ID {
		return Impact{}, errors.New("code manifest impact: delta identity mismatch")
	}
	seeds := map[SymbolID]bool{}
	for _, change := range delta.Symbols {
		seeds[change.ID] = true
	}
	reverse := map[SymbolID]map[SymbolID]bool{}
	addReverseReferences(reverse, base.References)
	addReverseReferences(reverse, candidate.References)
	reachable := map[SymbolID]bool{}
	queue := make([]SymbolID, 0, len(seeds))
	for seed := range seeds {
		reachable[seed] = true
		queue = append(queue, seed)
	}
	for len(queue) != 0 {
		current := queue[0]
		queue = queue[1:]
		for caller := range reverse[current] {
			if reachable[caller] {
				continue
			}
			reachable[caller] = true
			queue = append(queue, caller)
		}
	}
	impact := Impact{
		Base: base.ID.String(), Candidate: candidate.ID.String(),
		Seeds: slices.SortedFunc(maps.Keys(seeds), compareSymbolID), Reachable: slices.SortedFunc(maps.Keys(reachable), compareSymbolID),
		ExternalInputs: slices.Clone(delta.ExternalInputs),
		Uncertainty:    slices.Clone(delta.Uncertainty),
	}
	for _, change := range delta.ExternalInputs {
		impact.Uncertainty = append(impact.Uncertainty, Uncertainty{
			Kind: UncertaintyNonGo, Path: change.Path,
			Reason: "changed non-Go input has no structural dependency adapter",
		})
	}
	packages := map[string]bool{}
	for _, symbol := range impact.Reachable {
		packages[symbol.Package] = true
	}
	for _, change := range delta.Files {
		if change.Base != nil {
			packages[change.Base.Package] = true
		}
		if change.Candidate != nil {
			packages[change.Candidate.Package] = true
		}
	}
	for _, change := range delta.ExternalInputs {
		if change.Base != nil {
			packages[change.Base.Owner] = true
		}
		if change.Candidate != nil {
			packages[change.Candidate.Owner] = true
		}
	}
	impact.Packages = slices.Sorted(maps.Keys(packages))
	impact.Uncertainty = appendReachableUncertainty(impact.Uncertainty, reachable, delta.Files, base.Uncertainty, candidate.Uncertainty)
	return impact, nil
}

// ExclusionAuthority refuses impact with unresolved structural boundaries.
func (i Impact) ExclusionAuthority() error {
	if i.Base == "" || i.Candidate == "" {
		return errors.New("code manifest impact: missing manifest identities")
	}
	if len(i.Uncertainty) != 0 {
		return fmt.Errorf("code manifest impact: %d unresolved boundaries", len(i.Uncertainty))
	}
	return nil
}

func addReverseReferences(reverse map[SymbolID]map[SymbolID]bool, references []Reference) {
	for _, reference := range references {
		target := reference.To
		if reverse[target] == nil {
			reverse[target] = map[SymbolID]bool{}
		}
		reverse[target][reference.From] = true
	}
}

// Valid symbol fields exclude NUL, preserving the legacy separator-key order.
func compareSymbolID(left, right SymbolID) int {
	return cmp.Or(cmp.Compare(left.Package, right.Package), cmp.Compare(left.Context, right.Context), cmp.Compare(left.Receiver, right.Receiver), cmp.Compare(left.Name, right.Name), cmp.Compare(left.Kind, right.Kind))
}

func appendReachableUncertainty(result []Uncertainty, reachable map[SymbolID]bool, changes []FileChange, groups ...[]Uncertainty) []Uncertainty {
	paths := make(map[string]bool, len(changes))
	for _, change := range changes {
		paths[change.Path] = true
	}
	for _, group := range groups {
		for _, item := range group {
			if item.Path == "" || paths[item.Path] || item.Symbol != nil && reachable[*item.Symbol] {
				result = append(result, item)
			}
		}
	}
	slices.SortFunc(result, compareUncertainty)
	return slices.CompactFunc(result, func(left, right Uncertainty) bool { return compareUncertainty(left, right) == 0 })
}
