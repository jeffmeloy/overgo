package gostyle

import (
	"sort"
	"strconv"
	"strings"
)

func diagnosticChanges(base, candidate []Diagnostic) []DiagnosticDelta {
	baseByKey := diagnosticsByKey(base)
	candidateByKey := diagnosticsByKey(candidate)
	var changes []DiagnosticDelta
	for key, diagnostic := range candidateByKey {
		if _, found := baseByKey[key]; !found {
			changes = append(changes, DiagnosticDelta{State: Introduced, Diagnostic: diagnostic})
		}
	}
	for key, diagnostic := range baseByKey {
		if _, found := candidateByKey[key]; !found {
			changes = append(changes, DiagnosticDelta{State: Resolved, Diagnostic: diagnostic})
		}
	}
	sort.Slice(changes, func(i, j int) bool {
		left, right := changes[i], changes[j]
		if left.Diagnostic.RuleID != right.Diagnostic.RuleID {
			return left.Diagnostic.RuleID < right.Diagnostic.RuleID
		}
		if left.Diagnostic.File != right.Diagnostic.File {
			return left.Diagnostic.File < right.Diagnostic.File
		}
		if left.Diagnostic.Line != right.Diagnostic.Line {
			return left.Diagnostic.Line < right.Diagnostic.Line
		}
		return left.State < right.State
	})
	return changes
}

func diagnosticsByKey(diagnostics []Diagnostic) map[string]Diagnostic {
	byKey := make(map[string]Diagnostic, len(diagnostics))
	seen := make(map[string]int, len(diagnostics))
	for _, diagnostic := range diagnostics {
		identity := strings.Join([]string{
			diagnostic.RuleID, diagnostic.File, diagnostic.Symbol, diagnostic.Rationale,
		}, "\x00")
		seen[identity]++
		key := identity + "\x00" + strconv.Itoa(seen[identity])
		byKey[key] = diagnostic
	}
	return byKey
}
