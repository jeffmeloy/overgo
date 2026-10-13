package repoanalysis

import (
	"fmt"
	"go/ast"
	"go/token"
	"path"
	"slices"
	"strconv"
	"strings"

	"overgo/internal/gosyntax"
)

// ProductionAuthorityFinding is one deterministic architecture-ratchet
// refusal. Family names the protected authority; Kind distinguishes a bypass,
// import, or stale-allowance failure.
type ProductionAuthorityFinding struct {
	Family   string
	Kind     string
	Symbol   string
	Expected string
	Actual   string
	File     string
	Line     int
}

// ProductionAuthorityReport binds an authority audit to the complete source
// snapshot and makes both policy breadth and inspected site count observable.
type ProductionAuthorityReport struct {
	SourceIdentity string
	Rules          int
	Sites          int
	Findings       []ProductionAuthorityFinding
}

// Error returns nil for a clean candidate and otherwise names the first stable
// finding. Callers retain the complete report for evidence and review.
func (report ProductionAuthorityReport) Error() error {
	if len(report.Findings) == 0 {
		return nil
	}
	first := report.Findings[0]
	return fmt.Errorf(
		"production authority boundaries: %d finding(s); first=%s/%s %s expected=%s actual=%s file=%s:%d",
		len(report.Findings), first.Family, first.Kind, first.Symbol,
		first.Expected, first.Actual, first.File, first.Line,
	)
}

type productionAuthoritySource struct {
	gosyntax.File
	SyntaxFile *ast.File
	Package    string
	Imports    map[string]string
}

// AuditProductionAuthorityBoundaries refuses new or displaced production
// entry paths around storage, process, capability, tool, trigger, and
// promotion authorities. It parses the already-discovered candidate snapshot
// once and launches no subprocess.
func AuditProductionAuthorityBoundaries(snapshot gosyntax.Snapshot) (ProductionAuthorityReport, error) {
	report := ProductionAuthorityReport{SourceIdentity: snapshot.Identity()}
	sources, err := productionAuthoritySources(snapshot, &report)
	if err != nil {
		return ProductionAuthorityReport{}, err
	}
	auditStorageAndProcessAuthorities(sources, &report)
	auditCapabilityAndToolAuthorities(sources, &report)
	auditTriggerAndPromotionAuthorities(sources, &report)
	auditRSIAuthorities(sources, &report)
	slices.SortFunc(report.Findings, func(left, right ProductionAuthorityFinding) int {
		return strings.Compare(strings.Join([]string{
			left.Family, left.Kind, left.Symbol, left.File,
			fmt.Sprintf("%09d", left.Line), left.Actual, left.Expected,
		}, "\x00"), strings.Join([]string{
			right.Family, right.Kind, right.Symbol, right.File,
			fmt.Sprintf("%09d", right.Line), right.Actual, right.Expected,
		}, "\x00"))
	})
	return report, nil
}

var protectedAuthorityImports = map[string]string{
	"os":                                "process",
	"os/exec":                           "process",
	"overgo/internal/agenttool":         "tool",
	"overgo/internal/artifact":          "storage",
	"overgo/internal/capabilityruntime": "capability",
	"overgo/internal/composition":       "promotion",
	"overgo/internal/inference":         "promotion",
	"overgo/internal/processcontrol":    "process",
	"overgo/internal/runrecord":         "trigger",
	"overgo/internal/workflowruntime":   "trigger",
	"syscall":                           "process",
}

func productionAuthoritySources(snapshot gosyntax.Snapshot, report *ProductionAuthorityReport) ([]productionAuthoritySource, error) {
	sources := make([]productionAuthoritySource, 0, len(snapshot.Files))
	for _, source := range snapshot.Files {
		if source.Test {
			continue
		}
		generated, err := source.Generated()
		if err != nil {
			return nil, fmt.Errorf("production authority parse %s: %w", source.Path, err)
		}
		if generated {
			continue
		}
		file, err := source.Syntax()
		if err != nil {
			return nil, fmt.Errorf("production authority parse %s: %w", source.Path, err)
		}
		imports := make(map[string]string, len(file.Imports))
		for _, spec := range file.Imports {
			importPath, err := strconv.Unquote(spec.Path.Value)
			if err != nil {
				return nil, fmt.Errorf("production authority import %s: %w", source.Path, err)
			}
			name := path.Base(importPath)
			if spec.Name != nil {
				name = spec.Name.Name
			}
			imports[name] = importPath
			if name == "." {
				if family, protected := protectedAuthorityImports[importPath]; protected {
					addAuthorityFinding(report, ProductionAuthorityFinding{
						Family: family, Kind: "dot-import", Symbol: importPath,
						Expected: "named import", Actual: "dot import", File: source.Path, Line: source.Line(spec.Pos()),
					})
				}
			}
		}
		sources = append(sources, productionAuthoritySource{
			File: source, SyntaxFile: file, Package: path.Dir(source.Path), Imports: imports,
		})
	}
	return sources, nil
}

func addAuthorityFinding(report *ProductionAuthorityReport, finding ProductionAuthorityFinding) {
	report.Findings = append(report.Findings, finding)
}

func authorityReceiverName(expression ast.Expr) string {
	switch typed := expression.(type) {
	case *ast.Ident:
		return typed.Name
	case *ast.StarExpr:
		return authorityReceiverName(typed.X)
	case *ast.IndexExpr:
		return authorityReceiverName(typed.X)
	case *ast.IndexListExpr:
		return authorityReceiverName(typed.X)
	default:
		return ""
	}
}

type authoritySiteKey struct {
	File     string
	Function string
}

type authorityAllowance struct {
	File     string
	Function string
	Count    int
}

// Expected counts are ordinal positions, not free policy literals. Keeping the
// small closed vocabulary derived from iota makes every allowance readable
// while the observed candidate sites remain the source of truth.
const (
	oneAuthoritySite    = len([...]struct{}{{}})
	twoAuthoritySites   = oneAuthoritySite + oneAuthoritySite
	threeAuthoritySites = twoAuthoritySites + oneAuthoritySite
	fourAuthoritySites  = threeAuthoritySites + oneAuthoritySite
	fiveAuthoritySites  = fourAuthoritySites + oneAuthoritySite
	sixAuthoritySites   = fiveAuthoritySites + oneAuthoritySite
)

func recordAuthoritySite(sites map[authoritySiteKey][]int, source productionAuthoritySource, function string, position token.Pos) {
	key := authoritySiteKey{File: source.Path, Function: function}
	sites[key] = append(sites[key], source.Line(position))
}

func verifyAuthoritySites(
	report *ProductionAuthorityReport,
	family, symbol string,
	sites map[authoritySiteKey][]int,
	allowances []authorityAllowance,
) {
	report.Rules++
	report.Sites += authoritySiteCount(sites)
	matched := make([]int, len(allowances))
	for key, lines := range sites {
		allowed := len(allowances)
		for index, allowance := range allowances {
			if allowance.File == key.File && (allowance.Function == "" || allowance.Function == key.Function) {
				allowed = index
				break
			}
		}
		if allowed == len(allowances) {
			for _, line := range lines {
				addAuthorityFinding(report, ProductionAuthorityFinding{
					Family: family, Kind: "bypass", Symbol: symbol,
					Expected: "declared authority entry", Actual: key.Function, File: key.File, Line: line,
				})
			}
			continue
		}
		matched[allowed] += len(lines)
	}
	for index, allowance := range allowances {
		if matched[index] == allowance.Count {
			continue
		}
		addAuthorityFinding(report, ProductionAuthorityFinding{
			Family: family, Kind: "stale-allowance", Symbol: symbol,
			Expected: fmt.Sprintf("%d site(s)", allowance.Count), Actual: fmt.Sprintf("%d site(s)", matched[index]),
			File: allowance.File,
		})
	}
}

func authoritySiteCount(sites map[authoritySiteKey][]int) int {
	count := 0
	for _, lines := range sites {
		count += len(lines)
	}
	return count
}

func authorityFunctionName(function *ast.FuncDecl) string {
	if function == nil {
		return "<file>"
	}
	if function.Recv == nil || len(function.Recv.List) == 0 {
		return function.Name.Name
	}
	return authorityReceiverName(function.Recv.List[0].Type) + "." + function.Name.Name
}

func authoritySelectorImport(source productionAuthoritySource, selector *ast.SelectorExpr) string {
	identifier, ok := selector.X.(*ast.Ident)
	if !ok {
		return ""
	}
	return source.Imports[identifier.Name]
}

func authorityCompositeType(source productionAuthoritySource, expression ast.Expr) (string, string) {
	switch typed := expression.(type) {
	case *ast.Ident:
		return "", typed.Name
	case *ast.SelectorExpr:
		return authoritySelectorImport(source, typed), typed.Sel.Name
	default:
		return "", ""
	}
}
