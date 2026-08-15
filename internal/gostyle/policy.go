// Package gostyle provides Google-aligned Go policy census and candidate
// comparison. It reports rule-local evidence; enforcement belongs to cmd/gate.
package gostyle

import "overgo/internal/repoanalysis"

type Tier string

const (
	HardCandidateDelta Tier = "hard-candidate-delta"
	Advisory           Tier = "advisory"
)

type Mechanism string

const (
	Syntax    Mechanism = "syntax"
	TypeAware Mechanism = "type-aware"
	Toolchain Mechanism = "toolchain"
	Delegated Mechanism = "delegated"
)

type Rule struct {
	ID               string    `json:"id"`
	Tier             Tier      `json:"tier"`
	SourceSection    string    `json:"source_section"`
	Rationale        string    `json:"rationale"`
	Mechanism        Mechanism `json:"mechanism"`
	Coverage         string    `json:"coverage"`
	DeterministicFix bool      `json:"deterministic_fix"`
}

const (
	guide     = "https://google.github.io/styleguide/go/guide"
	decisions = "https://google.github.io/styleguide/go/decisions"
	practices = "https://google.github.io/styleguide/go/best-practices"
)

var policy = []Rule{
	{"gofmt-vet", HardCandidateDelta, guide + "#formatting", "Toolchain formatting and analysis are the common mechanical baseline.", Toolchain, "gofmt and go vet", true},
	{"package-imports", HardCandidateDelta, guide + "#package-names", "Package and import conventions keep ownership and dependencies legible.", Syntax, "package names, dot imports, and blank-import context", false},
	{"mixed-caps", HardCandidateDelta, guide + "#mixed-caps", "Go identifiers use MixedCaps rather than underscores.", Syntax, "new declarations with underscore-separated names; reviewed initialisms remain a later type-aware extension", false},
	{"receivers", HardCandidateDelta, decisions + "#receiver-names", "Short consistent receiver names reduce method-local noise.", Syntax, "receiver consistency, discouraged names, and unused receivers", false},
	{"context", HardCandidateDelta, decisions + "#contexts", "Context stays explicit, first, and request-scoped.", Syntax, "context.Context fields and parameter order", false},
	{"errors", HardCandidateDelta, decisions + "#errors", "Conventional error shapes preserve predictable callers and messages.", Syntax, "return order, exported concrete error pointers, and static error strings", false},
	{"discarded-error", HardCandidateDelta, practices + "#handle-errors", "Known discarded errors require explicit local ownership.", TypeAware, "requires type information and adjacent-comment attribution", false},
	{"library-flags", HardCandidateDelta, decisions + "#flags", "Importable libraries must not mutate command-line behavior as an import side effect.", Syntax, "flag package registration calls outside package main", false},
	{"module-tidy", HardCandidateDelta, guide + "#dependencies", "Module metadata should describe the actual dependency graph.", Toolchain, "go mod tidy -diff", true},
	{"interface-ownership", Advisory, practices + "#interfaces", "Interfaces should be introduced by demonstrated consumers.", Syntax, "interface declarations; consumer need remains reviewer judgment", false},
	{"global-state", Advisory, practices + "#global-state", "Mutable package state obscures ownership and concurrency.", Syntax, "package-level var declarations", false},
	{"background-context", Advisory, decisions + "#contexts", "Library call chains should propagate caller cancellation.", Syntax, "context.Background calls", false},
	{"panic-must", Advisory, decisions + "#dont-panic", "Panic-style control flow belongs only to initialization or invariants.", Syntax, "panic and Must-prefixed calls; path suitability remains reviewer judgment", false},
	{"get-prefix", Advisory, decisions + "#getters", "API names should expose cost or blocking where useful.", Syntax, "Get-prefixed function and method declarations", false},
	{"returns-copy", Advisory, decisions + "#naked-returns", "Explicit returns and deliberate receiver copying improve local readability.", Syntax, "naked returns in nontrivial functions; receiver copy cost requires types", false},
	{"test-quality", Advisory, practices + "#tests", "Tests should expose ownership and useful got/want evidence.", Syntax, "reflect.DeepEqual and testing helpers that fail without Helper", false},
	{"goroutine-ownership", Advisory, practices + "#goroutine-lifetimes", "Every goroutine needs visible lifetime ownership.", Syntax, "go statements; cancellation and join adequacy remain reviewer judgment", false},
	{"structural-growth", Advisory, practices + "#complexity", "Measured growth should justify new implementation surface.", Delegated, "internal/codeprofile owns functions, branches, clones, exports, and consumer evidence", false},
	{"discard-justification", Advisory, practices + "#handle-errors", "A discard comment must explain why loss is safe, not merely exist.", TypeAware, "reviewer judgment over type-known discarded errors", false},
}

// Policy returns the reviewed rule catalog in stable priority order.
func Policy() []Rule { return append([]Rule(nil), policy...) }

type Diagnostic struct {
	RuleID           string `json:"rule_id"`
	Tier             Tier   `json:"tier"`
	SourceSection    string `json:"source_section"`
	File             string `json:"file"`
	Line             int    `json:"line"`
	Symbol           string `json:"symbol,omitempty"`
	Rationale        string `json:"rationale"`
	DeterministicFix bool   `json:"deterministic_fix"`
}

type RuleCensus struct {
	Rule        Rule         `json:"rule"`
	Available   bool         `json:"available"`
	SkipReason  string       `json:"skip_reason,omitempty"`
	Diagnostics []Diagnostic `json:"diagnostics,omitempty"`
}

type Exclusion struct {
	File   string `json:"file"`
	Scope  string `json:"scope"`
	Reason string `json:"reason"`
}

type BuildConstraint struct {
	File       string `json:"file"`
	Expression string `json:"expression"`
}

type CensusReport struct {
	Identity         string            `json:"identity"`
	Rules            []RuleCensus      `json:"rules"`
	Exclusions       []Exclusion       `json:"exclusions,omitempty"`
	BuildConstraints []BuildConstraint `json:"build_constraints,omitempty"`
}

type DeltaState string

const (
	Introduced DeltaState = "introduced"
	Resolved   DeltaState = "resolved"
)

type DiagnosticDelta struct {
	State      DeltaState `json:"state"`
	Diagnostic Diagnostic `json:"diagnostic"`
}

type RuleDelta struct {
	Rule           Rule              `json:"rule"`
	Available      bool              `json:"available"`
	SkipReason     string            `json:"skip_reason,omitempty"`
	BaseCount      int               `json:"base_count"`
	CandidateCount int               `json:"candidate_count"`
	Changes        []DiagnosticDelta `json:"changes,omitempty"`
}

type BaselineReport struct {
	BaseIdentity              string            `json:"base_identity"`
	CandidateIdentity         string            `json:"candidate_identity"`
	Rules                     []RuleDelta       `json:"rules"`
	BaseExclusions            []Exclusion       `json:"base_exclusions,omitempty"`
	CandidateExclusions       []Exclusion       `json:"candidate_exclusions,omitempty"`
	BaseBuildConstraints      []BuildConstraint `json:"base_build_constraints,omitempty"`
	CandidateBuildConstraints []BuildConstraint `json:"candidate_build_constraints,omitempty"`
}

// Compare binds rule-local deltas to complete base and candidate snapshots.
// It intentionally has no aggregate score: improvement in one rule cannot hide
// regression in another.
func Compare(base, candidate repoanalysis.SourceSnapshot) (BaselineReport, error) {
	baseCensus, err := Census(base)
	if err != nil {
		return BaselineReport{}, err
	}
	candidateCensus, err := Census(candidate)
	if err != nil {
		return BaselineReport{}, err
	}
	report := BaselineReport{
		BaseIdentity: baseCensus.Identity, CandidateIdentity: candidateCensus.Identity,
		BaseExclusions: baseCensus.Exclusions, CandidateExclusions: candidateCensus.Exclusions,
		BaseBuildConstraints: baseCensus.BuildConstraints, CandidateBuildConstraints: candidateCensus.BuildConstraints,
		Rules: make([]RuleDelta, len(baseCensus.Rules)),
	}
	for index := range baseCensus.Rules {
		baseRule, candidateRule := baseCensus.Rules[index], candidateCensus.Rules[index]
		delta := RuleDelta{
			Rule: baseRule.Rule, Available: baseRule.Available, SkipReason: baseRule.SkipReason,
			BaseCount: len(baseRule.Diagnostics), CandidateCount: len(candidateRule.Diagnostics),
		}
		delta.Changes = diagnosticChanges(baseRule.Diagnostics, candidateRule.Diagnostics)
		report.Rules[index] = delta
	}
	return report, nil
}
