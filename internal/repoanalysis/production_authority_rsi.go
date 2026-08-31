package repoanalysis

import (
	"go/ast"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
)

const (
	modelRecipePackage       = "internal/modelrecipe"
	recipePackage            = "internal/recipe"
	invocationPackage        = "internal/invocation"
	trainingWorkflowPackage  = "internal/trainingworkflow"
	sequentialControlPackage = "internal/sequentialcontrol"
	runRecordImport          = "overgo/internal/runrecord"
)

// auditRSIAuthorities reserves the cross-domain RSI spines before their
// consumers are constructed and keeps runtime behavior compiled into Go. The
// reservation rules tolerate a future symbol being absent, but refuse the
// moment it appears outside its declared owner.
func auditRSIAuthorities(sources []productionAuthoritySource, report *ProductionAuthorityReport) {
	auditCrossDomainOwners(sources, report)
	auditRunrecordConstructorEntries(sources, report)
	auditCapabilityInvocationBoundaries(sources, report)
	auditGoOnlyRuntime(sources, report)
}

func auditCrossDomainOwners(sources []productionAuthoritySource, report *ProductionAuthorityReport) {
	requireAuthorityOwners(sources, report,
		authorityOwnerRule{Family: "candidate-admission", Kind: "type", Symbol: "AdmissionBinding", Owner: runRecordPackage},
		authorityOwnerRule{Family: "candidate-admission", Kind: "func", Symbol: "NewAdmissionBinding", Owner: runRecordPackage},
		authorityOwnerRule{Family: "candidate-admission", Kind: "func", Symbol: "ValidateAdmissionSuccession", Owner: runRecordPackage},
		authorityOwnerRule{Family: "candidate-admission", Kind: "type", Symbol: "CandidateAdmission", Owner: runRecordPackage},
		authorityOwnerRule{Family: "candidate-admission", Kind: "func", Symbol: "AdmitCandidate", Owner: runRecordPackage},
		authorityOwnerRule{Family: "promotion-lifecycle", Kind: "type", Symbol: "LifecycleEvent", Owner: recipePackage},
		authorityOwnerRule{Family: "promotion-lifecycle", Kind: "func", Symbol: "NewLifecycleEvent", Owner: recipePackage},
		authorityOwnerRule{Family: "promotion-lifecycle", Kind: "func", Symbol: "Transition", Owner: modelRecipePackage},
		authorityOwnerRule{Family: "invocation-contract", Kind: "type", Symbol: "ReceiptBinding", Owner: invocationPackage},
	)
	requireConcreteTypeOwner(sources, report, "invocation-contract", "Effect", invocationPackage)

	reserveAuthorityOwners(sources, report,
		authorityOwnerRule{Family: "evidence-driver", Kind: "type", Symbol: "DriverDecision", Owner: runRecordPackage},
		authorityOwnerRule{Family: "evidence-driver", Kind: "func", Symbol: "NewDriverDecision", Owner: runRecordPackage},
		authorityOwnerRule{Family: "training-evidence-publication", Kind: "type", Symbol: "TrainingEvidencePublication", Owner: trainingWorkflowPackage},
		authorityOwnerRule{Family: "training-evidence-publication", Kind: "func", Symbol: "PublishTrainingEvidence", Owner: trainingWorkflowPackage},
		authorityOwnerRule{Family: "sequential-control", Kind: "type", Symbol: "SequentialControlPlan", Owner: sequentialControlPackage},
		authorityOwnerRule{Family: "sequential-control", Kind: "func", Symbol: "CalibrateSequentialControl", Owner: sequentialControlPackage},
		authorityOwnerRule{Family: "sequential-control", Kind: "func", Symbol: "EvaluateSequentialControl", Owner: sequentialControlPackage},
	)

	// Candidate is intentionally a domain-scoped reservation: unrelated local
	// helper types may use the ordinary word, but RSI domains cannot establish
	// another cross-domain candidate envelope.
	reserveAuthorityOwnersInPackages(sources, report, rsiCandidatePackages,
		authorityOwnerRule{Family: "candidate-admission", Kind: "type", Symbol: "Candidate", Owner: modelRecipePackage},
	)
}

func auditRunrecordConstructorEntries(sources []productionAuthoritySource, report *ProductionAuthorityReport) {
	auditRunrecordConstructorEntry(
		sources, report, "candidate-admission", "AdmitCandidate",
		authorityAllowance{File: "internal/controlleraction/action.go", Function: "compileAction", Count: oneAuthoritySite},
	)
	auditRunrecordConstructorEntry(
		sources, report, "evidence-driver", "NewDriverDecision",
		authorityAllowance{
			File: "internal/evaluation/evidence_driver.go", Function: "CompileEvidenceDriverDecision", Count: oneAuthoritySite,
		},
	)
}

func auditRunrecordConstructorEntry(
	sources []productionAuthoritySource,
	report *ProductionAuthorityReport,
	family, constructor string,
	allowance authorityAllowance,
) {
	sites := map[authoritySiteKey][]int{}
	for _, source := range sources {
		inspectProductionAuthoritySource(source, func(function string, node ast.Node) {
			call, ok := node.(*ast.CallExpr)
			if ok && authorityCallNames(source, call.Fun, runRecordPackage, runRecordImport, constructor) {
				recordAuthoritySite(sites, source, function, call.Pos())
			}
		})
	}
	verifyAuthoritySites(report, family, constructor+" call", sites, []authorityAllowance{allowance})
}

// requireConcreteTypeOwner distinguishes the canonical definition from API
// aliases used while callers migrate. An alias may expose the type, but cannot
// become a second implementation owner.
func requireConcreteTypeOwner(
	sources []productionAuthoritySource,
	report *ProductionAuthorityReport,
	family, symbol, owner string,
) {
	report.Rules++
	var matches []ProductionAuthorityFinding
	for _, source := range sources {
		for _, declaration := range source.SyntaxFile.Decls {
			generic, ok := declaration.(*ast.GenDecl)
			if !ok {
				continue
			}
			for _, specification := range generic.Specs {
				typeSpec, ok := specification.(*ast.TypeSpec)
				if !ok || typeSpec.Name.Name != symbol || typeSpec.Assign.IsValid() {
					continue
				}
				matches = append(matches, ProductionAuthorityFinding{
					Actual: source.Package, File: source.Path, Line: source.Line(typeSpec.Pos()),
				})
			}
		}
	}
	report.Sites += len(matches)
	if len(matches) == 0 {
		addAuthorityFinding(report, ProductionAuthorityFinding{
			Family: family, Kind: "missing-owner", Symbol: symbol, Expected: owner, Actual: "absent",
		})
		return
	}
	for _, match := range matches {
		if match.Actual == owner && len(matches) == 1 {
			continue
		}
		match.Family, match.Kind, match.Symbol, match.Expected = family, "owner", symbol, owner
		addAuthorityFinding(report, match)
	}
}

var rsiCandidatePackages = map[string]bool{
	"internal/composition":      true,
	"internal/controlleraction": true,
	"internal/dataset":          true,
	"internal/loop":             true,
	"internal/modelrecipe":      true,
	"internal/representation":   true,
	"internal/runrecord":        true,
	"internal/trainingprogram":  true,
	"internal/trainingworkflow": true,
}

func reserveAuthorityOwners(sources []productionAuthoritySource, report *ProductionAuthorityReport, rules ...authorityOwnerRule) {
	reserveAuthorityOwnersInPackages(sources, report, nil, rules...)
}

func reserveAuthorityOwnersInPackages(
	sources []productionAuthoritySource,
	report *ProductionAuthorityReport,
	packages map[string]bool,
	rules ...authorityOwnerRule,
) {
	for _, rule := range rules {
		report.Rules++
		var matches []ProductionAuthorityFinding
		for _, source := range sources {
			if packages != nil && !packages[source.Package] {
				continue
			}
			matches = append(matches, authorityDeclarationMatches(source, rule)...)
		}
		report.Sites += len(matches)
		for _, match := range matches {
			if match.Actual == rule.Owner && len(matches) == 1 {
				continue
			}
			match.Family, match.Kind, match.Symbol, match.Expected =
				rule.Family, "reserved-owner", authorityOwnerSymbol(rule), rule.Owner
			addAuthorityFinding(report, match)
		}
	}
}

func authorityDeclarationMatches(source productionAuthoritySource, rule authorityOwnerRule) []ProductionAuthorityFinding {
	var matches []ProductionAuthorityFinding
	for _, declaration := range source.SyntaxFile.Decls {
		switch typed := declaration.(type) {
		case *ast.FuncDecl:
			kind := "func"
			receiver := ""
			if typed.Recv != nil && len(typed.Recv.List) != 0 {
				kind, receiver = "method", authorityReceiverName(typed.Recv.List[0].Type)
			}
			if rule.Kind == kind && rule.Symbol == typed.Name.Name && rule.Receiver == receiver {
				matches = append(matches, ProductionAuthorityFinding{
					Actual: source.Package, File: source.Path, Line: source.Line(typed.Pos()),
				})
			}
		case *ast.GenDecl:
			for _, spec := range typed.Specs {
				name, kind := "", ""
				switch value := spec.(type) {
				case *ast.TypeSpec:
					name, kind = value.Name.Name, "type"
				case *ast.ValueSpec:
					kind = "var"
					for _, candidate := range value.Names {
						if candidate.Name == rule.Symbol {
							name = candidate.Name
							break
						}
					}
				}
				if rule.Kind == kind && rule.Symbol == name {
					matches = append(matches, ProductionAuthorityFinding{
						Actual: source.Package, File: source.Path, Line: source.Line(spec.Pos()),
					})
				}
			}
		}
	}
	return matches
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
	{API: "Invoke", File: "internal/workflowruntime/automation.go", Function: "AutomationRuntime.deliverAutomation", Class: "agent-initiated"},
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
