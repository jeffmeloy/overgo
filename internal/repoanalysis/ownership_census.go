package repoanalysis

import (
	"fmt"
	"go/ast"
	"path"
	"sort"
	"strings"
)

// The RSI ownership census computes, from syntax alone, which package
// currently owns each mechanic family the storage campaign
// consolidates -- and who else re-implements it. The census is fully
// derived: markers name observable syntax (imported call selectors,
// declaration suffixes, sleep-in-loop retry shapes), the owner is the
// package with the most marked sites, and every other package holding
// a site is a consumer. Findings feed consolidation rows; nothing here
// is a hand-maintained list of files.

// CensusCall marks calls of one imported package's functions. Function
// "*" marks every selector call through the import.
type CensusCall struct {
	Import   string
	Function string
}

// CensusFamily declares one mechanic family by its observable syntax.
type CensusFamily struct {
	Name string
	// Calls mark call expressions through named imports.
	Calls []CensusCall
	// DeclSuffixes mark type declarations whose name carries the
	// family's shape (a local re-implementation of the mechanic).
	DeclSuffixes []string
	// SleepLoops marks for-statements that call time.Sleep -- the
	// hand-rolled retry/backoff shape.
	SleepLoops bool
	// ScopedCalls mark selector calls by name prefix, counted only in
	// files that import ScopedImport (a heuristic for method calls the
	// parser cannot type-resolve).
	ScopedCalls  []string
	ScopedImport string
}

// CensusFinding reports one family's computed ownership.
type CensusFinding struct {
	Family     string   `json:"family"`
	Owner      string   `json:"owner"`
	Sites      int      `json:"sites"`
	OwnerSites int      `json:"owner_sites"`
	Recurrence int      `json:"recurrence"`
	Consumers  []string `json:"consumers,omitempty"`
}

// rsiCensusFamilies is the declared marker table for the ten mechanic
// families the campaign consolidates. Markers are observable syntax;
// the families and their consolidation targets live in docs/plan.json.
func rsiCensusFamilies() []CensusFamily {
	return []CensusFamily{
		{Name: "storage", Calls: []CensusCall{
			{Import: "overgo/internal/overgodb", Function: "Open"},
			{Import: "overgo/internal/overgodb", Function: "OpenReadOnly"},
		}},
		{Name: "publication", ScopedCalls: []string{"Commit"}, ScopedImport: "overgo/internal/overgodb"},
		{Name: "process", Calls: []CensusCall{
			{Import: "os/exec", Function: "Command"},
			{Import: "os/exec", Function: "CommandContext"},
		}},
		{Name: "outcome", DeclSuffixes: []string{"Outcome", "Verdict"}},
		{Name: "failure", DeclSuffixes: []string{"Failure"}},
		{Name: "retry", SleepLoops: true},
		{Name: "capability", DeclSuffixes: []string{"Capability"}},
		{Name: "trigger", DeclSuffixes: []string{"Trigger"}},
		{Name: "resource", Calls: []CensusCall{
			{Import: "overgo/internal/processmeasure", Function: "*"},
			{Import: "runtime", Function: "ReadMemStats"},
		}},
		{Name: "query", ScopedCalls: []string{"Query"}, ScopedImport: "overgo/internal/overgodb"},
	}
}

// RSIOwnershipCensus computes the census over every production file in
// the snapshot. Owner selection is derived: the package with the most
// marked sites owns the family (ties resolve to the lexically first
// package name), every other package with a site is a consumer, and
// recurrence counts the sites outside the owner.
func RSIOwnershipCensus(snapshot SourceSnapshot) ([]CensusFinding, error) {
	families := rsiCensusFamilies()
	sites := make([]map[string]int, len(families))
	for index := range sites {
		sites[index] = map[string]int{}
	}
	for _, file := range snapshot.Files {
		if file.Test {
			continue
		}
		syntax, err := file.Syntax()
		if err != nil {
			return nil, fmt.Errorf("census parse %s: %w", file.Path, err)
		}
		pkg := path.Dir(file.Path)
		imports := importBases(syntax)
		for index, family := range families {
			count := countFamilySites(family, syntax, imports)
			if count > 0 {
				sites[index][pkg] += count
			}
		}
	}
	findings := make([]CensusFinding, 0, len(families))
	for index, family := range families {
		findings = append(findings, summarizeFamily(family.Name, sites[index]))
	}
	return findings, nil
}

// importBases maps local selector identifiers to import paths.
func importBases(file *ast.File) map[string]string {
	bases := map[string]string{}
	for _, spec := range file.Imports {
		importPath := strings.Trim(spec.Path.Value, `"`)
		name := path.Base(importPath)
		if spec.Name != nil {
			name = spec.Name.Name
		}
		bases[name] = importPath
	}
	return bases
}

func countFamilySites(family CensusFamily, file *ast.File, imports map[string]string) int {
	scoped := family.ScopedImport == ""
	for _, importPath := range imports {
		if importPath == family.ScopedImport {
			scoped = true
			break
		}
	}
	count := 0
	ast.Inspect(file, func(node ast.Node) bool {
		switch typed := node.(type) {
		case *ast.CallExpr:
			selector, ok := typed.Fun.(*ast.SelectorExpr)
			if !ok {
				return true
			}
			if base, ok := selector.X.(*ast.Ident); ok {
				for _, call := range family.Calls {
					if imports[base.Name] == call.Import &&
						(call.Function == "*" || selector.Sel.Name == call.Function) {
						count++
						return true
					}
				}
			}
			if scoped && family.ScopedImport != "" {
				for _, prefix := range family.ScopedCalls {
					if strings.HasPrefix(selector.Sel.Name, prefix) {
						count++
						return true
					}
				}
			}
		case *ast.TypeSpec:
			for _, suffix := range family.DeclSuffixes {
				if strings.HasSuffix(typed.Name.Name, suffix) {
					count++
					return true
				}
			}
		case *ast.ForStmt, *ast.RangeStmt:
			if family.SleepLoops && loopSleeps(node, imports) {
				count++
				return false
			}
		}
		return true
	})
	return count
}

func loopSleeps(loop ast.Node, imports map[string]string) bool {
	sleeps := false
	ast.Inspect(loop, func(node ast.Node) bool {
		call, ok := node.(*ast.CallExpr)
		if !ok {
			return true
		}
		selector, ok := call.Fun.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		if base, ok := selector.X.(*ast.Ident); ok &&
			imports[base.Name] == "time" && selector.Sel.Name == "Sleep" {
			sleeps = true
			return false
		}
		return true
	})
	return sleeps
}

func summarizeFamily(name string, packageSites map[string]int) CensusFinding {
	finding := CensusFinding{Family: name}
	packages := make([]string, 0, len(packageSites))
	for pkg, count := range packageSites {
		packages = append(packages, pkg)
		finding.Sites += count
	}
	sort.Strings(packages)
	for _, pkg := range packages {
		if finding.Owner == "" || packageSites[pkg] > packageSites[finding.Owner] {
			finding.Owner = pkg
		}
	}
	for _, pkg := range packages {
		if pkg == finding.Owner {
			continue
		}
		finding.Consumers = append(finding.Consumers, pkg)
		finding.Recurrence += packageSites[pkg]
	}
	if finding.Owner != "" {
		finding.OwnerSites = packageSites[finding.Owner]
	}
	return finding
}
