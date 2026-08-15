package gostyle

import (
	"strconv"
	"strings"

	"overgo/internal/repoanalysis"
)

// Compare binds rule-local deltas to complete base and candidate snapshots.
// It intentionally has no aggregate score: improvement in one rule cannot hide
// regression in another.
func Compare(base, candidate repoanalysis.SourceSnapshot, selections ...repoanalysis.BuildSelection) (BaselineReport, error) {
	var selection repoanalysis.BuildSelection
	if len(selections) > 0 {
		selection = selections[0]
	}
	cache := factCache{}
	baseCensus, err := census(base, selection, cache)
	if err != nil {
		return BaselineReport{}, err
	}
	candidateCensus, err := census(candidate, selection, cache)
	if err != nil {
		return BaselineReport{}, err
	}
	report := BaselineReport{
		Base: snapshotEvidence(baseCensus), Candidate: snapshotEvidence(candidateCensus),
		Rules: make([]ruleDelta, len(baseCensus.Rules)),
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
		delta := ruleDelta{
			Rule: baseRule.Rule, Selected: baseRule.Selected, SkipReason: baseRule.SkipReason,
			BaseCount: len(baseRule.Diagnostics), CandidateCount: len(candidateRule.Diagnostics),
			Blocking: baseRule.Rule.Maturity == enforceable,
		}
		delta.Introduced, delta.Resolved = diagnosticChanges(baseRule.Diagnostics, candidateRule.Diagnostics)
		report.Rules[index] = delta
	}
	return report, nil
}

func snapshotEvidence(census censusReport) snapshotFacts {
	return snapshotFacts{
		Identity: census.Identity, BuildContext: census.BuildContext,
		Exclusions: census.Exclusions, BuildConstraints: census.BuildConstraints,
	}
}

func diagnosticChanges(base, candidate []diagnostic) (introduced, resolved []diagnostic) {
	baseByKey := diagnosticsByKey(base)
	candidateByKey := diagnosticsByKey(candidate)
	for key, diagnostic := range candidateByKey {
		if _, found := baseByKey[key]; !found {
			introduced = append(introduced, diagnostic)
		}
	}
	for key, diagnostic := range baseByKey {
		if _, found := candidateByKey[key]; !found {
			resolved = append(resolved, diagnostic)
		}
	}
	sortDiagnostics(introduced)
	sortDiagnostics(resolved)
	return introduced, resolved
}

func diagnosticsByKey(diagnostics []diagnostic) map[string]diagnostic {
	byKey := make(map[string]diagnostic, len(diagnostics))
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
