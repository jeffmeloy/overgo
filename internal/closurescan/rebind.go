package closurescan

import (
	"bytes"

	"overgo/internal/closureledger"
)

type RebindIndex map[string][]Candidate

func CompileRebindIndex(candidates []Candidate) RebindIndex {
	current := make(RebindIndex, len(candidates))
	for _, candidate := range candidates {
		key := rebindKey(candidate.Kind, candidate.Package, candidate.File, candidate.Scope, candidate.Name, candidate.Expression)
		current[key] = append(current[key], candidate)
	}
	return current
}

// Rebind matches policy and callsites, moving source identity when needed.
func (current RebindIndex) Rebind(document closureledger.Document) (closureledger.Document, bool, string, error) {
	bindings := make([]closureledger.SourceBinding, len(document.Bindings))
	fixture := document.Fixture
	changed := false
	for index, previous := range document.Bindings {
		candidates := current[rebindKey(previous.Kind, previous.Package, previous.File, previous.Scope, previous.Name, previous.Expression)]
		var match Candidate
		matches, callsites := 0, false
		for _, candidate := range candidates {
			if candidate.CallsiteID != previous.CallsiteID {
				continue
			}
			callsites = true
			if bytes.Equal(candidate.ValueJSON(), document.Value) {
				match, matches = candidate, matches+1
			}
		}
		if matches != 1 {
			reason := "value"
			switch {
			case len(candidates) == 0:
				reason = "declaration"
			case !callsites:
				reason = "callsite"
			case matches > 1:
				reason = "ambiguous"
			}
			return closureledger.Document{}, false, reason, nil
		}
		binding, err := match.Binding()
		if err != nil {
			return closureledger.Document{}, false, "binding", err
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
		return document, true, "exact", nil
	}
	rebound, err := closureledger.New(
		document.Name, document.Value, document.Tier, document.Status, document.Understanding,
		bindings, document.ClosurePath, document.RerankTrigger, fixture,
	)
	return rebound, err == nil, "source", err
}

func rebindKey(kind closureledger.BindingKind, pkg, file, scope, name, expression string) string {
	return string(kind) + "\x00" + pkg + "\x00" + file + "\x00" + scope + "\x00" + name + "\x00" + expression
}
