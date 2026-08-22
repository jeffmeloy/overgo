package closurescan

import (
	"bytes"

	"overgo/internal/closureledger"
)

type RebindIndex map[string][]Candidate

func CompileRebindIndex(candidates []Candidate) RebindIndex {
	current := make(RebindIndex, len(candidates))
	for _, candidate := range candidates {
		current[rebindCandidateKey(candidate)] = append(current[rebindCandidateKey(candidate)], candidate)
	}
	return current
}

// Rebind matches policy and callsites, moving source identity when needed.
func (current RebindIndex) Rebind(document closureledger.Document) (closureledger.Document, bool, error) {
	bindings := make([]closureledger.SourceBinding, len(document.Bindings))
	fixture := document.Fixture
	changed := false
	for index, previous := range document.Bindings {
		var match Candidate
		found := false
		for _, candidate := range current[rebindBindingKey(previous)] {
			if found {
				return closureledger.Document{}, false, nil
			}
			match, found = candidate, true
		}
		if !found || !bytes.Equal(match.ValueJSON(), document.Value) {
			return closureledger.Document{}, false, nil
		}
		binding, err := match.Binding()
		if err != nil {
			return closureledger.Document{}, false, err
		}
		bindings[index] = binding
		if fixture == previous.Owner {
			fixture = binding.Owner
		}
		if binding != previous {
			changed = true
		}
	}
	if !changed {
		return document, true, nil
	}
	rebound, err := closureledger.New(
		document.Name, document.Value, document.Tier, document.Status, document.Understanding,
		bindings, document.ClosurePath, document.RerankTrigger, fixture,
	)
	return rebound, err == nil, err
}

func rebindCandidateKey(candidate Candidate) string {
	return string(candidate.Kind) + "\x00" + candidate.Package + "\x00" + candidate.File + "\x00" +
		candidate.Scope + "\x00" + candidate.Name + "\x00" + candidate.Expression + "\x00" + candidate.CallsiteID
}

func rebindBindingKey(binding closureledger.SourceBinding) string {
	return string(binding.Kind) + "\x00" + binding.Package + "\x00" + binding.File + "\x00" +
		binding.Scope + "\x00" + binding.Name + "\x00" + binding.Expression + "\x00" + binding.CallsiteID
}
