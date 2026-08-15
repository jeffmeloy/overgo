package gostyle

import (
	"sort"
	"strconv"
	"strings"

	"overgo/internal/repoanalysis"
)

// Compare binds rule-local deltas to complete base and candidate snapshots.
// It intentionally has no aggregate score: improvement in one rule cannot hide
// regression in another.
func Compare(base, candidate repoanalysis.SourceSnapshot, options ...Options) (BaselineReport, error) {
	option := firstOption(options)
	cache := factCache{}
	baseCensus, err := census(base, option, cache)
	if err != nil {
		return BaselineReport{}, err
	}
	candidateCensus, err := census(candidate, option, cache)
	if err != nil {
		return BaselineReport{}, err
	}
	report := BaselineReport{
		Base: snapshotEvidence(baseCensus), Candidate: snapshotEvidence(candidateCensus),
		Rules: make([]RuleDelta, len(baseCensus.Rules)),
		Analysis: AnalysisStats{
			FilesScanned: baseCensus.Analysis.FilesScanned + candidateCensus.Analysis.FilesScanned,
			FilesReused:  candidateCensus.Analysis.FilesReused,
			SyntaxPasses: baseCensus.Analysis.SyntaxPasses + candidateCensus.Analysis.SyntaxPasses,
			Diagnostics:  candidateCensus.Analysis.Diagnostics,
			Duration:     baseCensus.Analysis.Duration + candidateCensus.Analysis.Duration,
		},
	}
	for index := range baseCensus.Rules {
		baseRule, candidateRule := baseCensus.Rules[index], candidateCensus.Rules[index]
		delta := RuleDelta{
			Rule: baseRule.Rule, Selected: baseRule.Selected, SkipReason: baseRule.SkipReason,
			BaseCount: len(baseRule.Diagnostics), CandidateCount: len(candidateRule.Diagnostics),
		}
		delta.Changes = diagnosticChanges(baseRule.Diagnostics, candidateRule.Diagnostics)
		report.Rules[index] = delta
	}
	return report, nil
}

func snapshotEvidence(census CensusReport) SnapshotEvidence {
	return SnapshotEvidence{
		Identity: census.Identity, BuildContext: census.BuildContext,
		Exclusions: census.Exclusions, BuildConstraints: census.BuildConstraints,
	}
}

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
			diagnostic.File, diagnostic.Symbol, diagnostic.Message,
		}, "\x00")
		seen[identity]++
		key := identity + "\x00" + strconv.Itoa(seen[identity])
		byKey[key] = diagnostic
	}
	return byKey
}
