// Package agentworkflow projects measured repository authority into bounded
// agent planning context without granting agents verification authority.
package agentworkflow

import (
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/codemanifest"
)

// RequiredCheck identifies one immutable verification definition selected for the candidate.
type RequiredCheck struct {
	Name       string                 `json:"name"`
	Definition artifact.ID            `json:"definition"`
	Matched    []automationcheck.Fact `json:"matched,omitempty"`
}

// ManifestContext is a read-only projection of selector authority for agent
// change planning. ProvenExclusions always comes from ManifestPlan.
type ManifestContext struct {
	Plan              artifact.ID                 `json:"plan"`
	CandidateManifest artifact.ID                 `json:"candidate_manifest"`
	AffectedSymbols   []codemanifest.SymbolID     `json:"affected_symbols,omitempty"`
	RiskBoundaries    []codemanifest.Uncertainty  `json:"risk_boundaries,omitempty"`
	OwnedPackages     []string                    `json:"owned_packages,omitempty"`
	RequiredChecks    []RequiredCheck             `json:"required_checks"`
	ProvenExclusions  []automationcheck.Exclusion `json:"proven_exclusions,omitempty"`
	PriorEvidence     []artifact.ID               `json:"prior_evidence,omitempty"`
}

// NewManifestContext derives agent context only from immutable selector
// output and caller-supplied prior evidence identities.
func NewManifestContext(impact codemanifest.Impact, plan automationcheck.ManifestPlan, prior []artifact.ID) (ManifestContext, error) {
	if err := plan.Validate(); err != nil || impact.Candidate != plan.CandidateManifest.String() {
		return ManifestContext{}, errors.New("agent workflow: manifest authority mismatch")
	}
	for _, id := range prior {
		if !id.Valid() {
			return ManifestContext{}, errors.New("agent workflow: invalid prior evidence")
		}
	}
	context := ManifestContext{
		Plan: plan.ID, CandidateManifest: plan.CandidateManifest,
		AffectedSymbols: slices.Clone(impact.Reachable), RiskBoundaries: slices.Clone(impact.Uncertainty),
		OwnedPackages: slices.Clone(impact.Packages), ProvenExclusions: slices.Clone(plan.Exclusions),
		PriorEvidence: slices.Clone(prior), RequiredChecks: make([]RequiredCheck, len(plan.Invocations)),
	}
	for index, invocation := range plan.Invocations {
		context.RequiredChecks[index] = RequiredCheck{
			Name: invocation.Check.Name, Definition: invocation.ID, Matched: slices.Clone(invocation.Matched),
		}
	}
	return context, nil
}
