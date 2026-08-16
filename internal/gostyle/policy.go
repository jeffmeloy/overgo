// Package gostyle provides Google-aligned Go policy census and candidate
// comparison. It reports rule-local evidence; enforcement belongs to cmd/gate.
package gostyle

import (
	"fmt"
	"time"
)

type tier string

const (
	hardCandidateDelta tier = "hard-candidate-delta"
	advisory           tier = "advisory"
)

type mechanism string

const (
	syntaxMechanism    mechanism = "syntax"
	typeAwareMechanism mechanism = "type-aware"
	toolchainMechanism mechanism = "toolchain"
	delegatedMechanism mechanism = "delegated"
)

type maturity string

const (
	declared      maturity = "declared"
	observed      maturity = "observed"
	authoritative maturity = "authoritative"
	enforceable   maturity = "enforceable"
)

type rule struct {
	ID               string    `json:"id"`
	Tier             tier      `json:"tier"`
	SourceID         string    `json:"source_id"`
	SourceSection    string    `json:"source_section"`
	Rationale        string    `json:"rationale"`
	Mechanism        mechanism `json:"mechanism"`
	Maturity         maturity  `json:"maturity"`
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

func policyRule(id string, tier tier, sourceID, rationale string, mechanism mechanism, maturity maturity, coverage string, fix bool) rule {
	return rule{
		ID: id, Tier: tier, SourceID: sourceID, SourceSection: reviewedSources[sourceID], Rationale: rationale,
		Mechanism: mechanism, Maturity: maturity, Coverage: coverage, DeterministicFix: fix,
	}
}

var policy = []rule{
	policyRule("gofmt-vet", hardCandidateDelta, "guide/formatting", "Toolchain formatting and analysis are the common mechanical baseline.", toolchainMechanism, authoritative, "gofmt and go vet", true),
	policyRule("package-imports", hardCandidateDelta, "decisions/package-names", "Package and import conventions keep ownership and dependencies legible.", syntaxMechanism, observed, "package names, dot imports, and blank-import context", false),
	policyRule("mixed-caps", hardCandidateDelta, "guide/mixed-caps", "Go identifiers use MixedCaps rather than underscores.", syntaxMechanism, observed, "declaration names with underscore-separated words; reviewed initialisms remain a later type-aware extension", false),
	policyRule("receivers", hardCandidateDelta, "decisions/receiver-names", "Short consistent receiver names reduce method-local noise.", syntaxMechanism, observed, "receiver consistency, discouraged names, and parser-object-resolved unused receivers", false),
	policyRule("context", hardCandidateDelta, "decisions/contexts", "Context stays explicit, first, and request-scoped.", syntaxMechanism, observed, "import-resolved context.Context fields and parameter order", false),
	policyRule("errors", hardCandidateDelta, "decisions/errors", "Conventional error shapes preserve predictable callers and messages.", syntaxMechanism, observed, "return order and import-resolved static error strings; concrete error implementation requires types", false),
	policyRule("discarded-error", hardCandidateDelta, "decisions/handle-errors", "Known discarded errors require explicit local ownership.", typeAwareMechanism, declared, "requires type information and adjacent-comment attribution", false),
	policyRule("library-flags", hardCandidateDelta, "decisions/flags", "Importable libraries must not mutate command-line behavior as an import side effect.", syntaxMechanism, observed, "import-resolved flag package registration calls outside package main", false),
	policyRule("module-tidy", hardCandidateDelta, "go/module-tidy", "Module metadata should describe the actual dependency graph.", toolchainMechanism, authoritative, "go mod tidy -diff", true),
	policyRule("interface-ownership", advisory, "practices/interfaces", "Interfaces should be introduced by demonstrated consumers.", syntaxMechanism, observed, "interface declarations; consumer need remains reviewer judgment", false),
	policyRule("global-state", advisory, "practices/global-state", "Mutable package state obscures ownership and concurrency.", syntaxMechanism, observed, "package-level var declarations", false),
	policyRule("background-context", advisory, "decisions/contexts", "Library call chains should propagate caller cancellation.", syntaxMechanism, observed, "import-resolved context.Background calls", false),
	policyRule("panic-must", advisory, "decisions/dont-panic", "Panic-style control flow belongs only to initialization or invariants.", syntaxMechanism, observed, "panic and Must-prefixed calls; path suitability remains reviewer judgment", false),
	policyRule("get-prefix", advisory, "decisions/getters", "API names should expose cost or blocking where useful.", syntaxMechanism, observed, "Get-prefixed function and method declarations", false),
	policyRule("returns-copy", advisory, "decisions/naked-returns", "Explicit returns and deliberate receiver copying improve local readability.", syntaxMechanism, observed, "naked returns in nontrivial functions; receiver copy cost requires types", false),
	policyRule("test-quality", advisory, "practices/testing", "Tests should expose ownership and useful got/want evidence.", syntaxMechanism, observed, "import-resolved reflect.DeepEqual and testing helpers that fail without Helper", false),
	policyRule("goroutine-ownership", advisory, "decisions/goroutine-lifetimes", "Every goroutine needs visible lifetime ownership.", syntaxMechanism, observed, "go statements; cancellation and join adequacy remain reviewer judgment", false),
	policyRule("structural-growth", advisory, "guide/simplicity", "Measured growth should justify new implementation surface.", delegatedMechanism, authoritative, "internal/codeprofile owns functions, branches, clones, exports, and consumer evidence", false),
	policyRule("discard-justification", advisory, "decisions/handle-errors", "A discard comment must explain why loss is safe, not merely exist.", typeAwareMechanism, declared, "reviewer judgment over type-known discarded errors", false),
}

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
		if rule.Maturity == enforceable && rule.Tier != hardCandidateDelta {
			return fmt.Errorf("go style policy: advisory rule %s cannot be enforceable", rule.ID)
		}
	}
	return nil
}

type diagnostic struct {
	File    string `json:"file"`
	Line    int    `json:"line"`
	Symbol  string `json:"symbol,omitempty"`
	Message string `json:"message"`
}

type censusRule struct {
	Rule        rule         `json:"rule"`
	Selected    bool         `json:"selected"`
	SkipReason  string       `json:"skip_reason,omitempty"`
	Diagnostics []diagnostic `json:"diagnostics,omitempty"`
}

type exclusion struct {
	File   string `json:"file"`
	Scope  string `json:"scope"`
	Reason string `json:"reason"`
}

type buildConstraint struct {
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

type censusReport struct {
	Identity         string            `json:"identity"`
	BuildContext     string            `json:"build_context,omitempty"`
	Rules            []censusRule      `json:"rules"`
	Exclusions       []exclusion       `json:"exclusions,omitempty"`
	BuildConstraints []buildConstraint `json:"build_constraints,omitempty"`
	Analysis         AnalysisStats     `json:"analysis"`
}

type ruleDelta struct {
	Rule           rule         `json:"rule"`
	Selected       bool         `json:"selected"`
	SkipReason     string       `json:"skip_reason,omitempty"`
	BaseCount      int          `json:"base_count"`
	CandidateCount int          `json:"candidate_count"`
	Introduced     []diagnostic `json:"introduced,omitempty"`
	Resolved       []diagnostic `json:"resolved,omitempty"`
	Blocking       bool         `json:"blocking,omitempty"`
}

type snapshotFacts struct {
	Identity         string            `json:"identity"`
	BuildContext     string            `json:"build_context,omitempty"`
	Exclusions       []exclusion       `json:"exclusions,omitempty"`
	BuildConstraints []buildConstraint `json:"build_constraints,omitempty"`
}

type BaselineReport struct {
	Base      snapshotFacts `json:"base"`
	Candidate snapshotFacts `json:"candidate"`
	Rules     []ruleDelta   `json:"rules"`
	Analysis  AnalysisStats `json:"analysis"`
}
