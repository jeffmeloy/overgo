package repoanalysis

import (
	"go/ast"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
)

// auditRSIAuthorities classifies every capability invocation site and keeps
// runtime behavior compiled into Go.
func auditRSIAuthorities(sources []productionAuthoritySource, report *ProductionAuthorityReport) {
	auditCapabilityInvocationBoundaries(sources, report)
	auditGoOnlyRuntime(sources, report)
}

type invocationBoundary struct {
	API      string
	File     string
	Function string
	Class    string
}

var capabilityInvocationBoundaries = []invocationBoundary{
	{API: "Invoke", File: "cmd/agent-tool/main.go", Function: "run", Class: "agent-initiated"},
	{API: "InvokeWithEffect", File: "internal/agentloop/coordinator.go", Function: "Coordinator.propose", Class: "agent-initiated"},
	{API: "InvokeWithEffect", File: "internal/agenttool/proxy.go", Function: "RegisterCapabilityProxy", Class: "cross-boundary"},
	{API: "InvokeWithEffect", File: "internal/agenttool/transport.go", Function: "Executor.Invoke", Class: "transport-owner"},
	{API: "Invoke", File: "internal/workflowruntime/tool_workflow.go", Function: "toolWorkflowAdapter", Class: "agent-initiated"},
}

var directGoOrchestrationPackages = map[string]bool{
	"internal/composition":      true,
	"internal/controlleraction": true,
	"internal/evaluation":       true,
	"internal/loop":             true,
	"internal/modelrecipe":      true,
	"internal/operation":        true,
	"internal/recipe":           true,
	"internal/runrecord":        true,
	"internal/trainingprogram":  true,
	"internal/trainingworkflow": true,
}

func auditCapabilityInvocationBoundaries(sources []productionAuthoritySource, report *ProductionAuthorityReport) {
	report.Rules += len(capabilityInvocationBoundaries) + oneAuthoritySite
	matched := make([]int, len(capabilityInvocationBoundaries))
	for _, source := range sources {
		inspectProductionAuthoritySource(source, func(function string, node ast.Node) {
			call, ok := node.(*ast.CallExpr)
			if !ok {
				return
			}
			selector, ok := call.Fun.(*ast.SelectorExpr)
			if !ok || !isCapabilityInvocation(source, selector) {
				return
			}
			report.Sites++
			boundary := len(capabilityInvocationBoundaries)
			for index, candidate := range capabilityInvocationBoundaries {
				if candidate.API == selector.Sel.Name && candidate.File == source.Path && candidate.Function == function {
					boundary = index
					break
				}
			}
			if boundary == len(capabilityInvocationBoundaries) {
				addAuthorityFinding(report, ProductionAuthorityFinding{
					Family: "invocation-boundary", Kind: "unclassified", Symbol: selector.Sel.Name,
					Expected: "agent-initiated or cross-boundary transport", Actual: function,
					File: source.Path, Line: source.Line(call.Pos()),
				})
			} else {
				matched[boundary]++
			}
			if directGoOrchestrationPackages[source.Package] {
				addAuthorityFinding(report, ProductionAuthorityFinding{
					Family: "direct-go", Kind: "transport-bypass", Symbol: selector.Sel.Name,
					Expected: "direct compiled Go owner call", Actual: function,
					File: source.Path, Line: source.Line(call.Pos()),
				})
			}
		})
	}
	for index, boundary := range capabilityInvocationBoundaries {
		if matched[index] == oneAuthoritySite {
			continue
		}
		addAuthorityFinding(report, ProductionAuthorityFinding{
			Family: "invocation-boundary", Kind: "stale-classification", Symbol: boundary.API,
			Expected: boundary.Class, Actual: strconv.Itoa(matched[index]) + " site(s)",
			File: boundary.File,
		})
	}
}

func isCapabilityInvocation(source productionAuthoritySource, selector *ast.SelectorExpr) bool {
	switch selector.Sel.Name {
	case "Invoke", "InvokeWithEffect", "OpenStream":
		return source.Package == agentToolPackage || sourceImportsAuthority(source, agentToolImport)
	default:
		return false
	}
}

var goOnlyRuntimePackages = map[string]bool{
	"internal/agentloop":         true,
	"internal/agenttool":         true,
	"internal/capabilityruntime": true,
	"internal/composition":       true,
	"internal/controlleraction":  true,
	"internal/evaluation":        true,
	"internal/invocation":        true,
	"internal/loop":              true,
	"internal/modelrecipe":       true,
	"internal/operation":         true,
	"internal/recipe":            true,
	"internal/runrecord":         true,
	"internal/trainingprogram":   true,
	"internal/trainingworkflow":  true,
	"internal/workflowruntime":   true,
}

var capabilityDiscoveryPackages = map[string]bool{
	"internal/agenttool":         true,
	"internal/capabilityruntime": true,
	"internal/invocation":        true,
	"internal/modelrecipe":       true,
	"internal/workflowruntime":   true,
}

var forbiddenRuntimeImportFragments = []string{
	"/goja", "/gopher-lua", "/otto", "/starlark", "/wazero", "/yaegi",
}

var interpretedExecutables = map[string]bool{
	"bash": true, "cmd": true, "node": true, "perl": true, "php": true,
	"powershell": true, "pwsh": true, "python": true, "python3": true,
	"ruby": true, "sh": true,
}

var nonGoBehaviorExtensions = map[string]bool{
	".js": true, ".lua": true, ".markdown": true, ".md": true, ".mjs": true,
	".ps1": true, ".py": true, ".pyc": true, ".rb": true, ".sh": true,
}

func auditGoOnlyRuntime(sources []productionAuthoritySource, report *ProductionAuthorityReport) {
	// Imports, interpreter execution, non-Go behavior reads, embedded behavior,
	// and filesystem catalog discovery are separate policy clauses.
	report.Rules += fiveAuthoritySites
	for _, source := range sources {
		for _, imported := range source.Imports {
			if imported == "plugin" || slices.ContainsFunc(forbiddenRuntimeImportFragments, func(fragment string) bool {
				return strings.Contains(imported, fragment)
			}) {
				report.Sites++
				addAuthorityFinding(report, ProductionAuthorityFinding{
					Family: "go-only", Kind: "runtime-loader", Symbol: imported,
					Expected: "compiled Go registration", Actual: "runtime import", File: source.Path,
				})
			}
		}
		if !goOnlyRuntimePackages[source.Package] {
			continue
		}
		for _, group := range source.SyntaxFile.Comments {
			for _, comment := range group.List {
				if strings.HasPrefix(comment.Text, "//go:embed ") && hasNonGoBehaviorExtension(strings.TrimSpace(strings.TrimPrefix(comment.Text, "//go:embed "))) {
					report.Sites++
					addAuthorityFinding(report, ProductionAuthorityFinding{
						Family: "go-only", Kind: "embedded-behavior", Symbol: comment.Text,
						Expected: "compiled Go registration", Actual: "non-Go embedded behavior",
						File: source.Path, Line: source.Line(comment.Pos()),
					})
				}
			}
		}
		inspectProductionAuthoritySource(source, func(function string, node ast.Node) {
			call, ok := node.(*ast.CallExpr)
			if !ok {
				return
			}
			if executable, found := directInterpreterExecutable(source, call); found {
				report.Sites++
				addAuthorityFinding(report, ProductionAuthorityFinding{
					Family: "go-only", Kind: "interpreted-runtime", Symbol: executable,
					Expected: "compiled Go capability", Actual: function,
					File: source.Path, Line: source.Line(call.Pos()),
				})
			}
			if nonGoBehaviorRead(source, call) {
				report.Sites++
				addAuthorityFinding(report, ProductionAuthorityFinding{
					Family: "go-only", Kind: "non-go-behavior", Symbol: "filesystem read",
					Expected: "typed evidence or compiled Go registration", Actual: function,
					File: source.Path, Line: source.Line(call.Pos()),
				})
			}
			if capabilityDiscoveryPackages[source.Package] && filesystemEnumeration(source, call) {
				report.Sites++
				addAuthorityFinding(report, ProductionAuthorityFinding{
					Family: "go-only", Kind: "filesystem-capability-discovery", Symbol: authorityCallName(call.Fun),
					Expected: "compiled Go registration", Actual: function,
					File: source.Path, Line: source.Line(call.Pos()),
				})
			}
		})
	}
}

func directInterpreterExecutable(source productionAuthoritySource, call *ast.CallExpr) (string, bool) {
	if len(call.Args) == 0 || !authorityCallFromImport(source, call.Fun, "os/exec", "Command", "CommandContext") {
		return "", false
	}
	literal, ok := stringLiteral(call.Args[0])
	if !ok {
		return "", false
	}
	name := strings.ToLower(filepath.Base(strings.TrimSpace(literal)))
	name = strings.TrimSuffix(name, filepath.Ext(name))
	if interpretedExecutables[name] || hasNonGoBehaviorExtension(literal) {
		return literal, true
	}
	for _, argument := range call.Args[1:] {
		if value, constant := stringLiteral(argument); constant && hasNonGoBehaviorExtension(value) {
			return value, true
		}
	}
	return "", false
}

func nonGoBehaviorRead(source productionAuthoritySource, call *ast.CallExpr) bool {
	if len(call.Args) == 0 || !authorityCallFromImport(source, call.Fun, "os", "Open", "OpenFile", "ReadFile") {
		return false
	}
	value, ok := stringLiteral(call.Args[0])
	return ok && hasNonGoBehaviorExtension(value)
}

func filesystemEnumeration(source productionAuthoritySource, call *ast.CallExpr) bool {
	return authorityCallFromImport(source, call.Fun, "os", "ReadDir") ||
		authorityCallFromImport(source, call.Fun, "io/fs", "ReadDir", "WalkDir") ||
		authorityCallFromImport(source, call.Fun, "path/filepath", "Walk", "WalkDir")
}

func authorityCallFromImport(source productionAuthoritySource, expression ast.Expr, importPath string, names ...string) bool {
	selector, ok := expression.(*ast.SelectorExpr)
	if !ok || authoritySelectorImport(source, selector) != importPath {
		return false
	}
	return slices.Contains(names, selector.Sel.Name)
}

func authorityCallName(expression ast.Expr) string {
	if selector, ok := expression.(*ast.SelectorExpr); ok {
		return selector.Sel.Name
	}
	return "call"
}

func stringLiteral(expression ast.Expr) (string, bool) {
	literal, ok := expression.(*ast.BasicLit)
	if !ok || literal.Kind.String() != "STRING" {
		return "", false
	}
	value, err := strconv.Unquote(literal.Value)
	return value, err == nil
}

func hasNonGoBehaviorExtension(value string) bool {
	value = strings.TrimSpace(value)
	if fields := strings.Fields(value); len(fields) == oneAuthoritySite {
		value = fields[0]
	}
	value = strings.TrimSuffix(strings.SplitN(value, "?", twoAuthoritySites)[0], "/")
	return nonGoBehaviorExtensions[strings.ToLower(filepath.Ext(value))]
}
