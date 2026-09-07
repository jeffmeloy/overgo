package main

import (
	"fmt"
	"io"

	"overgo/internal/closurescan"
	"overgo/internal/repoanalysis"
)

// auditIdentityLiterals prints one line per literal the scanner classified
// by an identity rule and returns the count; the rule is the closure.
func auditIdentityLiterals(output io.Writer, candidates []closurescan.Candidate) int {
	audited := 0
	for _, candidate := range candidates {
		if candidate.Identity == "" {
			continue
		}
		audited++
		fmt.Fprintf(output, "closure-scan: identity %s %s:%d %s\n", candidate.Identity, candidate.File, candidate.Line, candidate.Expression)
	}
	return audited
}

// refuseIdentityTriage: a triage row or proposal naming a rule-classified
// literal is refused; the rule already closes it as a mathematical fact.
func refuseIdentityTriage(candidate closurescan.Candidate) error {
	if candidate.Identity == "" {
		return nil
	}
	return fmt.Errorf("%s at %s:%d is classified by rule %s as a mathematical fact; no triage row",
		candidate.Name, candidate.File, candidate.Line, candidate.Identity)
}

// identityCandidate: the rule-classified literal candidate with this name,
// found by scanning the snapshot's literals; false when no such literal.
func identityCandidate(snapshot repoanalysis.SourceSnapshot, name string) (closurescan.Candidate, bool, error) {
	candidates, err := closurescan.ScanSnapshot(snapshot, nil, closurescan.CandidateLiterals)
	if err != nil {
		return closurescan.Candidate{}, false, err
	}
	for _, candidate := range candidates {
		if candidate.Name == name && candidate.Identity != "" {
			return candidate, true, nil
		}
	}
	return closurescan.Candidate{}, false, nil
}
