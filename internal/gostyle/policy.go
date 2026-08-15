// Package gostyle provides Google-aligned Go policy census and candidate
// comparison. It reports rule-local evidence; enforcement belongs to cmd/gate.
package gostyle

import (
	"fmt"
	"time"

	"overgo/internal/repoanalysis"
)

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

type Maturity string

const (
	Declared      Maturity = "declared"
	Observed      Maturity = "observed"
	Authoritative Maturity = "authoritative"
	Enforceable   Maturity = "enforceable"
)

type Rule struct {
	ID               string    `json:"id"`
	Tier             Tier      `json:"tier"`
	SourceID         string    `json:"source_id"`
	SourceSection    string    `json:"source_section"`
	Rationale        string    `json:"rationale"`
	Mechanism        Mechanism `json:"mechanism"`
	Maturity         Maturity  `json:"maturity"`
	Coverage         string    `json:"coverage"`
	DeterministicFix bool      `json:"deterministic_fix"`
}

var reviewedSources = map[string]string{
	"guide/formatting":              "https://google.github.io/styleguide/go/guide#formatting",
	"guide/mixed-caps":              "https://google.github.io/styleguide/go/guide#mixed-caps",
	"guide/simplicity":              "https://google.github.io/styleguide/go/guide#simplicity",
	"decisions/package-names":       "https://google.github.io/styleguide/go/decisions#package-names",
	"decisions/receiver-names":      "https://google.github.io/styleguide/go/decisions#receiver-names",
	"decisions/contexts":            "https://google.github.io/styleguide/go/decisions#contexts",
	"decisions/errors":              "https://google.github.io/styleguide/go/decisions#errors",
	"decisions/handle-errors":       "https://google.github.io/styleguide/go/decisions#handle-errors",
	"decisions/flags":               "https://google.github.io/styleguide/go/decisions#flags",
	"decisions/dont-panic":          "https://google.github.io/styleguide/go/decisions#dont-panic",
	"decisions/getters":             "https://google.github.io/styleguide/go/decisions#getters",
	"decisions/naked-returns":       "https://google.github.io/styleguide/go/decisions#naked-returns",
	"decisions/goroutine-lifetimes": "https://google.github.io/styleguide/go/decisions#goroutine-lifetimes",
	"practices/interfaces":          "https://google.github.io/styleguide/go/best-practices#interfaces",
	"practices/global-state":        "https://google.github.io/styleguide/go/best-practices#global-state",
	"practices/testing":             "https://google.github.io/styleguide/go/best-practices#testing",
	"go/module-tidy":                "https://go.dev/ref/mod#go-mod-tidy",
}

func policyRule(id string, tier Tier, sourceID, rationale string, mechanism Mechanism, maturity Maturity, coverage string, fix bool) Rule {
	return Rule{
		ID: id, Tier: tier, SourceID: sourceID, SourceSection: reviewedSources[sourceID], Rationale: rationale,
		Mechanism: mechanism, Maturity: maturity, Coverage: coverage, DeterministicFix: fix,
	}
}

var policy = []Rule{
	policyRule("gofmt-vet", HardCandidateDelta, "guide/formatting", "Toolchain formatting and analysis are the common mechanical baseline.", Toolchain, Authoritative, "gofmt and go vet", true),
	policyRule("package-imports", HardCandidateDelta, "decisions/package-names", "Package and import conventions keep ownership and dependencies legible.", Syntax, Observed, "package names, dot imports, and blank-import context", false),
	policyRule("mixed-caps", HardCandidateDelta, "guide/mixed-caps", "Go identifiers use MixedCaps rather than underscores.", Syntax, Observed, "declaration names with underscore-separated words; reviewed initialisms remain a later type-aware extension", false),
	policyRule("receivers", HardCandidateDelta, "decisions/receiver-names", "Short consistent receiver names reduce method-local noise.", Syntax, Observed, "receiver consistency, discouraged names, and parser-object-resolved unused receivers", false),
	policyRule("context", HardCandidateDelta, "decisions/contexts", "Context stays explicit, first, and request-scoped.", Syntax, Observed, "import-resolved context.Context fields and parameter order", false),
	policyRule("errors", HardCandidateDelta, "decisions/errors", "Conventional error shapes preserve predictable callers and messages.", Syntax, Observed, "return order and import-resolved static error strings; concrete error implementation requires types", false),
	policyRule("discarded-error", HardCandidateDelta, "decisions/handle-errors", "Known discarded errors require explicit local ownership.", TypeAware, Declared, "requires type information and adjacent-comment attribution", false),
	policyRule("library-flags", HardCandidateDelta, "decisions/flags", "Importable libraries must not mutate command-line behavior as an import side effect.", Syntax, Observed, "import-resolved flag package registration calls outside package main", false),
	policyRule("module-tidy", HardCandidateDelta, "go/module-tidy", "Module metadata should describe the actual dependency graph.", Toolchain, Authoritative, "go mod tidy -diff", true),
	policyRule("interface-ownership", Advisory, "practices/interfaces", "Interfaces should be introduced by demonstrated consumers.", Syntax, Observed, "interface declarations; consumer need remains reviewer judgment", false),
	policyRule("global-state", Advisory, "practices/global-state", "Mutable package state obscures ownership and concurrency.", Syntax, Observed, "package-level var declarations", false),
	policyRule("background-context", Advisory, "decisions/contexts", "Library call chains should propagate caller cancellation.", Syntax, Observed, "import-resolved context.Background calls", false),
	policyRule("panic-must", Advisory, "decisions/dont-panic", "Panic-style control flow belongs only to initialization or invariants.", Syntax, Observed, "panic and Must-prefixed calls; path suitability remains reviewer judgment", false),
	policyRule("get-prefix", Advisory, "decisions/getters", "API names should expose cost or blocking where useful.", Syntax, Observed, "Get-prefixed function and method declarations", false),
	policyRule("returns-copy", Advisory, "decisions/naked-returns", "Explicit returns and deliberate receiver copying improve local readability.", Syntax, Observed, "naked returns in nontrivial functions; receiver copy cost requires types", false),
	policyRule("test-quality", Advisory, "practices/testing", "Tests should expose ownership and useful got/want evidence.", Syntax, Observed, "import-resolved reflect.DeepEqual and testing helpers that fail without Helper", false),
	policyRule("goroutine-ownership", Advisory, "decisions/goroutine-lifetimes", "Every goroutine needs visible lifetime ownership.", Syntax, Observed, "go statements; cancellation and join adequacy remain reviewer judgment", false),
	policyRule("structural-growth", Advisory, "guide/simplicity", "Measured growth should justify new implementation surface.", Delegated, Authoritative, "internal/codeprofile owns functions, branches, clones, exports, and consumer evidence", false),
	policyRule("discard-justification", Advisory, "decisions/handle-errors", "A discard comment must explain why loss is safe, not merely exist.", TypeAware, Declared, "reviewer judgment over type-known discarded errors", false),
}

// Policy returns the reviewed rule catalog in stable priority order.
func Policy() []Rule { return append([]Rule(nil), policy...) }

func validatePolicy() error {
	seen := map[string]bool{}
	for _, rule := range policy {
		if rule.ID == "" || seen[rule.ID] {
			return fmt.Errorf("go style policy: missing or duplicate rule %q", rule.ID)
		}
		seen[rule.ID] = true
		if source, ok := reviewedSources[rule.SourceID]; !ok || source == "" || source != rule.SourceSection {
			return fmt.Errorf("go style policy: rule %s has unreviewed source %q", rule.ID, rule.SourceID)
		}
		if rule.Maturity == Enforceable && rule.Tier != HardCandidateDelta {
			return fmt.Errorf("go style policy: advisory rule %s cannot be enforceable", rule.ID)
		}
	}
	return nil
}

type Diagnostic struct {
	File    string `json:"file"`
	Line    int    `json:"line"`
	Symbol  string `json:"symbol,omitempty"`
	Message string `json:"message"`
}

type RuleCensus struct {
	Rule        Rule         `json:"rule"`
	Selected    bool         `json:"selected"`
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
	Selected   bool   `json:"selected"`
}

type AnalysisStats struct {
	FilesScanned int           `json:"files_scanned"`
	FilesReused  int           `json:"files_reused"`
	SyntaxPasses int           `json:"syntax_passes"`
	Diagnostics  int           `json:"diagnostics"`
	Duration     time.Duration `json:"duration_ns"`
}

type CensusReport struct {
	Identity         string            `json:"identity"`
	BuildContext     string            `json:"build_context,omitempty"`
	Rules            []RuleCensus      `json:"rules"`
	Exclusions       []Exclusion       `json:"exclusions,omitempty"`
	BuildConstraints []BuildConstraint `json:"build_constraints,omitempty"`
	Analysis         AnalysisStats     `json:"analysis"`
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
	Selected       bool              `json:"selected"`
	SkipReason     string            `json:"skip_reason,omitempty"`
	BaseCount      int               `json:"base_count"`
	CandidateCount int               `json:"candidate_count"`
	Changes        []DiagnosticDelta `json:"changes,omitempty"`
}

type SnapshotEvidence struct {
	Identity         string            `json:"identity"`
	BuildContext     string            `json:"build_context,omitempty"`
	Exclusions       []Exclusion       `json:"exclusions,omitempty"`
	BuildConstraints []BuildConstraint `json:"build_constraints,omitempty"`
}

type BaselineReport struct {
	Base      SnapshotEvidence `json:"base"`
	Candidate SnapshotEvidence `json:"candidate"`
	Rules     []RuleDelta      `json:"rules"`
	Analysis  AnalysisStats    `json:"analysis"`
}

type Options struct {
	Build repoanalysis.BuildSelection
}
