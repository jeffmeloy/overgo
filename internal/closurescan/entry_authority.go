package closurescan

import (
	"fmt"
	"go/ast"
	"path"
	"sort"
	"strings"

	"overgo/internal/repoanalysis"
)

// EntryAuthorityDomain names one established production entry authority the
// permanent architecture ratchet protects.
type EntryAuthorityDomain string

const (
	// EntryAuthorityStorage guards the canonical journal layout owned by internal/overgodb.
	EntryAuthorityStorage EntryAuthorityDomain = "storage"
	// EntryAuthorityProcess guards external process execution owned by internal/processcontrol.
	EntryAuthorityProcess EntryAuthorityDomain = "process"
	// EntryAuthorityCapability guards serving-runner construction owned by internal/inference.
	EntryAuthorityCapability EntryAuthorityDomain = "capability"
	// EntryAuthorityTool guards tool-manual entry owned by internal/agenttool registration.
	EntryAuthorityTool EntryAuthorityDomain = "tool"
	// EntryAuthorityTrigger guards execution-trigger admission owned by internal/workflowruntime.
	EntryAuthorityTrigger EntryAuthorityDomain = "trigger"
	// EntryAuthorityPromotion guards the activation alias namespace owned by internal/modelrecipe.
	EntryAuthorityPromotion EntryAuthorityDomain = "promotion"
)

// EntryAuthorityRule binds one domain's bypass detection to its single owner.
// Exceptions grandfather the production files still holding a direct site,
// each with the reason and the work that retires it. The list only shrinks: a
// new bypass site anywhere else refuses, and a stale entry whose sites are
// gone refuses too, so every boundary ratchets toward its one owner. Retire
// names the only condition under which the rule itself may be deleted.
type EntryAuthorityRule struct {
	Domain     EntryAuthorityDomain
	Owner      string
	OwnerPaths []string
	Exceptions map[string]string
	Detect     func(file *ast.File) int
	Retire     string
}

// EntryAuthorityDomainReport measures one applied rule.
type EntryAuthorityDomainReport struct {
	Domain          EntryAuthorityDomain `json:"domain"`
	ProductionFiles int                  `json:"production_files"`
	BypassSites     int                  `json:"bypass_sites"`
	ExceptionFiles  int                  `json:"exception_files"`
}

// EntryAuthorityReport summarizes the complete architecture ratchet pass.
type EntryAuthorityReport struct {
	Domains []EntryAuthorityDomainReport `json:"domains"`
}

func selectorSites(file *ast.File, match func(*ast.SelectorExpr) bool) int {
	sites := 0
	ast.Inspect(file, func(node ast.Node) bool {
		if selector, ok := node.(*ast.SelectorExpr); ok && match(selector) {
			sites++
		}
		return true
	})
	return sites
}

func selectorIs(selector *ast.SelectorExpr, base, name string) bool {
	identifier, ok := selector.X.(*ast.Ident)
	return ok && identifier.Name == base && selector.Sel.Name == name
}

func compositeSites(file *ast.File, base, name string) int {
	sites := 0
	ast.Inspect(file, func(node ast.Node) bool {
		composite, ok := node.(*ast.CompositeLit)
		if !ok {
			return true
		}
		if selector, ok := composite.Type.(*ast.SelectorExpr); ok && selectorIs(selector, base, name) {
			sites++
		}
		return true
	})
	return sites
}

func literalSites(file *ast.File, fragment string) int {
	sites := 0
	ast.Inspect(file, func(node ast.Node) bool {
		if literal, ok := node.(*ast.BasicLit); ok && strings.Contains(literal.Value, fragment) {
			sites++
		}
		return true
	})
	return sites
}

func underAny(filePath string, prefixes []string) bool {
	directory := path.Dir(filePath)
	for _, prefix := range prefixes {
		if directory == prefix || strings.HasPrefix(directory+"/", prefix+"/") {
			return true
		}
	}
	return false
}

// EntryAuthorityRules is the single policy table for the permanent
// architecture ratchet. The gate consumes it on every commit, including
// documentation-only commits, and each protected package's
// TestProductionAuthorityBoundaries exercises its own domain against this
// same table, so there is exactly one policy owner and no second guard
// process.
func EntryAuthorityRules() []EntryAuthorityRule {
	return []EntryAuthorityRule{
		{
			Domain:     EntryAuthorityStorage,
			Owner:      "internal/overgodb owns the canonical journal layout; recovery lanes are enumerated",
			OwnerPaths: []string{"internal/overgodb"},
			Exceptions: map[string]string{
				"cmd/overgodb-rebuild/main.go":            "sanctioned offline rebuild lane; reads and swaps the journal it rebuilds",
				"cmd/overgodb-repair/main.go":             "sanctioned torn-tail repair lane; truncates the journal at the last valid frame",
				"internal/closurescan/entry_authority.go": "single policy owner; the rule table names the journal layout it guards",
			},
			Detect: func(file *ast.File) int {
				return literalSites(file, "overgodb.log")
			},
			Retire: "internal/overgodb stops owning a journal file layout",
		},
		{
			Domain:     EntryAuthorityProcess,
			Owner:      "internal/processcontrol owns external process execution and termination",
			OwnerPaths: []string{"internal/processcontrol"},
			Exceptions: map[string]string{
				"internal/runrecord/verifying_commit.go":       "read-only git fact reads; execution-semantics binds repository facts to supervised receipts",
				"internal/repoanalysis/build.go":               "read-only go toolchain fact reads; execution-semantics migration",
				"internal/repoanalysis/imports.go":             "read-only go toolchain fact reads; execution-semantics migration",
				"internal/clioptions/command.go":               "command self-identification fact; execution-semantics migration",
				"internal/testutil/process.go":                 "test-support process helpers consumed only by suites",
				"cmd/plan/sync.go":                             "read-only git fact reads; execution-semantics migration",
				"cmd/gate/main.go":                             "gate check invocations; execution-semantics binds gate steps to supervised receipts",
				"cmd/loophook/main.go":                         "hook-time tooling; execution-semantics migration",
				"cmd/release/main.go":                          "release lane tooling; execution-semantics migration",
				"cmd/reverify/main.go":                         "reverify lane tooling; execution-semantics migration",
				"cmd/build-kernels/main.go":                    "kernel toolchain invocations; execution-semantics migration",
				"cmd/smoke-lane/main.go":                       "smoke lane tooling; execution-semantics migration",
				"cmd/test-lane/main.go":                        "test lane tooling; execution-semantics migration",
				"cmd/eval-lane/main.go":                        "eval lane tooling; execution-semantics migration",
				"cmd/evaluate/main.go":                         "evaluation tooling; execution-semantics migration",
				"cmd/evaluate/all.go":                          "evaluation tooling; execution-semantics migration",
				"cmd/sbom/main.go":                             "read-only go toolchain fact reads; execution-semantics migration",
				"cmd/recipe/main.go":                           "read-only git fact reads; execution-semantics migration",
				"cmd/advisories/main.go":                       "read-only git fact reads; execution-semantics migration",
				"cmd/compatibility/training.go":                "training lane tooling; execution-semantics migration",
				"cmd/composite-generation-lane/run_windows.go": "generation lane tooling; execution-semantics migration",
			},
			Detect: func(file *ast.File) int {
				return selectorSites(file, func(selector *ast.SelectorExpr) bool {
					if selectorIs(selector, "exec", "Command") || selectorIs(selector, "exec", "CommandContext") {
						return true
					}
					if selector.Sel.Name == "Kill" {
						inner, ok := selector.X.(*ast.SelectorExpr)
						return ok && inner.Sel.Name == "Process"
					}
					return false
				})
			},
			Retire: "internal/processcontrol stops owning external process execution",
		},
		{
			Domain:     EntryAuthorityCapability,
			Owner:      "internal/inference owns the serving runner; OpenWithProgram over a resolved active recipe is the only door",
			OwnerPaths: []string{"internal/inference"},
			Exceptions: map[string]string{},
			Detect: func(file *ast.File) int {
				return compositeSites(file, "inference", "Runner")
			},
			Retire: "internal/inference stops owning the serving runner lifecycle",
		},
		{
			Domain:     EntryAuthorityTool,
			Owner:      "internal/agenttool owns tool manuals; manuals enter through registration and resolve from the store catalog",
			OwnerPaths: []string{"internal/agenttool"},
			Exceptions: map[string]string{},
			Detect: func(file *ast.File) int {
				return compositeSites(file, "agenttool", "Manual")
			},
			Retire: "internal/agenttool stops owning the registered tool catalog",
		},
		{
			Domain:     EntryAuthorityTrigger,
			Owner:      "internal/workflowruntime admits execution triggers; webhook deliveries are constructed only by the runrecord ledger and the dispatch authority",
			OwnerPaths: []string{"internal/runrecord", "internal/workflowruntime"},
			Exceptions: map[string]string{},
			Detect: func(file *ast.File) int {
				return compositeSites(file, "runrecord", "WebhookDelivery")
			},
			Retire: "internal/workflowruntime stops owning trigger admission",
		},
		{
			Domain:     EntryAuthorityPromotion,
			Owner:      "internal/modelrecipe owns capability activation; the recipe.active alias namespace changes only through its lifecycle",
			OwnerPaths: []string{"internal/modelrecipe"},
			Exceptions: map[string]string{
				"internal/closurescan/entry_authority.go": "single policy owner; the rule table names the activation alias namespace it guards",
			},
			Detect: func(file *ast.File) int {
				return literalSites(file, "recipe.active.")
			},
			Retire: "internal/modelrecipe stops owning capability activation aliases",
		},
	}
}

// EntryAuthorityRuleFor returns one domain's rule from the single policy
// table, so a protected package can exercise exactly its own boundary.
func EntryAuthorityRuleFor(domain EntryAuthorityDomain) (EntryAuthorityRule, bool) {
	for _, rule := range EntryAuthorityRules() {
		if rule.Domain == domain {
			return rule, true
		}
	}
	return EntryAuthorityRule{}, false
}

// ValidateEntryAuthorities applies every rule in the single policy table to
// the complete production source inventory and refuses any bypass around an
// established authority, and any stale grandfathered exception.
func ValidateEntryAuthorities(
	snapshot repoanalysis.SourceSnapshot,
	rules []EntryAuthorityRule,
) (EntryAuthorityReport, error) {
	report := EntryAuthorityReport{}
	for _, rule := range rules {
		domain, err := validateEntryAuthorityRule(snapshot, rule)
		if err != nil {
			return EntryAuthorityReport{}, err
		}
		report.Domains = append(report.Domains, domain)
	}
	sort.Slice(report.Domains, func(left, right int) bool {
		return report.Domains[left].Domain < report.Domains[right].Domain
	})
	return report, nil
}

func validateEntryAuthorityRule(
	snapshot repoanalysis.SourceSnapshot,
	rule EntryAuthorityRule,
) (EntryAuthorityDomainReport, error) {
	if rule.Domain == "" || rule.Owner == "" || rule.Detect == nil || rule.Retire == "" {
		return EntryAuthorityDomainReport{}, fmt.Errorf("entry authority: rule %q is incomplete", rule.Domain)
	}
	domain := EntryAuthorityDomainReport{Domain: rule.Domain}
	found := map[string]int{}
	for _, file := range snapshot.Files {
		filePath := path.Clean(strings.ReplaceAll(file.Path, "\\", "/"))
		if file.Test || underAny(filePath, rule.OwnerPaths) {
			continue
		}
		domain.ProductionFiles++
		syntax, err := file.Syntax()
		if err != nil {
			return EntryAuthorityDomainReport{}, fmt.Errorf("entry authority %s: parse %s: %w", rule.Domain, filePath, err)
		}
		sites := rule.Detect(syntax)
		if sites == 0 {
			continue
		}
		if _, grandfathered := rule.Exceptions[filePath]; !grandfathered {
			return EntryAuthorityDomainReport{}, fmt.Errorf(
				"entry authority %s: %s bypasses the owner (%s); route through it or record a reviewed exception",
				rule.Domain, filePath, rule.Owner,
			)
		}
		found[filePath] += sites
		domain.BypassSites += sites
	}
	for exception := range rule.Exceptions {
		if found[exception] == 0 {
			return EntryAuthorityDomainReport{}, fmt.Errorf(
				"entry authority %s: exception %s no longer holds a direct site; delete its entry",
				rule.Domain, exception,
			)
		}
	}
	domain.ExceptionFiles = len(found)
	return domain, nil
}
