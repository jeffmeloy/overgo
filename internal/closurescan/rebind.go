package closurescan

import (
	"bytes"

	"overgo/internal/closureledger"
)

type RebindIndex map[string][]Candidate

func CompileRebindIndex(candidates []Candidate) RebindIndex {
	current := make(RebindIndex, len(candidates))
	for _, candidate := range candidates {
		current["s\x00"+candidate.StructuralID] = append(current["s\x00"+candidate.StructuralID], candidate)
		current["l\x00"+candidate.LegacyDeclarationKey()] = append(current["l\x00"+candidate.LegacyDeclarationKey()], candidate)
		key := "m\x00" + rebindKey(candidate.Kind, candidate.Package, candidate.File, candidate.Scope, candidate.Name, candidate.Expression)
		current[key] = append(current[key], candidate)
	}
	return current
}

// Rebind matches policy and callsites, moving source identity when needed.
func (current RebindIndex) Rebind(document closureledger.Document) (closureledger.Document, bool, string, error) {
	return current.rebind(document, false)
}

// RebindReviewed accepts reviewed callsite drift for unchanged declarations.
func (current RebindIndex) RebindReviewed(document closureledger.Document) (closureledger.Document, bool, string, error) {
	return current.rebind(document, true)
}

func (current RebindIndex) rebind(document closureledger.Document, reviewed bool) (closureledger.Document, bool, string, error) {
	bindings := make([]closureledger.SourceBinding, len(document.Bindings))
	fixture := document.Fixture
	changed := false
	for index, previous := range document.Bindings {
		key := "s\x00" + previous.StructuralID
		migration := previous.StructuralID == ""
		var candidates []Candidate
		if migration {
			legacyKey := "l\x00" + (Candidate{
				Kind: previous.Kind, File: previous.File, Scope: previous.Scope, Line: previous.Line, Name: previous.Name,
			}).LegacyDeclarationKey()
			for _, candidate := range current[legacyKey] {
				binding, err := candidate.Binding()
				if err != nil {
					return closureledger.Document{}, false, "binding", err
				}
				if binding.Owner == previous.Owner {
					candidates = append(candidates, candidate)
				}
			}
			if len(candidates) == 0 {
				key = "m\x00" + rebindKey(previous.Kind, previous.Package, previous.File, previous.Scope, previous.Name, previous.Expression)
				candidates = current[key]
			}
		}
		if !migration {
			candidates = current[key]
		}
		var match Candidate
		matches, callsites := 0, false
		for _, candidate := range candidates {
			if !migration && !reviewed && candidate.CallsiteID != previous.CallsiteID {
				continue
			}
			callsites = true
			if bytes.Equal(candidate.ValueJSON(), document.Value) {
				match, matches = candidate, matches+1
			}
		}
		if !migration && matches == 0 && (reviewed || previous.Kind == closureledger.BindingLiteral) {
			key = "m\x00" + rebindKey(previous.Kind, previous.Package, previous.File, previous.Scope, previous.Name, previous.Expression)
			candidates = current[key]
			migration, callsites = true, false
			for _, candidate := range candidates {
				callsites = true
				if bytes.Equal(candidate.ValueJSON(), document.Value) {
					match, matches = candidate, matches+1
				}
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

// ContentMatchedRebind rebinds one single-binding document whose
// structural and expression matches failed onto the one candidate that
// still carries the same kind, package, file, scope, constant name, and
// exact value, and whose own alias no active decision holds — the
// offset-shifted successor of the binding about to be retired. The
// closure text travels with the document unchanged, so the reviewed
// understanding is preserved instead of being retyped. Zero free matches
// or more than one refuse with the exact reason; only an unambiguous
// successor rebinds.
func ContentMatchedRebind(
	document closureledger.Document,
	candidates []Candidate,
	claimed func(alias string) bool,
) (closureledger.Document, bool, string, error) {
	if len(document.Bindings) != 1 || claimed == nil {
		return closureledger.Document{}, false, "bindings", nil
	}
	previous := document.Bindings[0]
	var match Candidate
	matches := 0
	for _, candidate := range candidates {
		if candidate.Kind != previous.Kind || candidate.Package != previous.Package ||
			candidate.File != previous.File || candidate.Scope != previous.Scope {
			continue
		}
		if candidate.Kind == closureledger.BindingConstant && candidate.Name != previous.Name {
			continue
		}
		if !bytes.Equal(candidate.ValueJSON(), document.Value) {
			continue
		}
		binding, err := candidate.Binding()
		if err != nil {
			return closureledger.Document{}, false, "binding", err
		}
		alias, err := closureledger.ActiveAlias(binding)
		if err != nil {
			return closureledger.Document{}, false, "alias", err
		}
		if claimed(alias) {
			continue
		}
		match, matches = candidate, matches+1
	}
	if matches != 1 {
		reason := "declaration"
		if matches > 1 {
			reason = "ambiguous"
		}
		return closureledger.Document{}, false, reason, nil
	}
	binding, err := match.Binding()
	if err != nil {
		return closureledger.Document{}, false, "binding", err
	}
	fixture := document.Fixture
	if fixture == previous.Owner {
		fixture = binding.Owner
	}
	rebound, err := closureledger.New(
		document.Name, document.Value, document.Tier, document.Status, document.Understanding,
		[]closureledger.SourceBinding{binding}, document.ClosurePath, document.RerankTrigger, fixture,
	)
	return rebound, err == nil, "content", err
}

func rebindKey(kind closureledger.BindingKind, pkg, file, scope, name, expression string) string {
	if kind != closureledger.BindingConstant {
		name = ""
	}
	return string(kind) + "\x00" + pkg + "\x00" + file + "\x00" + scope + "\x00" + name + "\x00" + expression
}
