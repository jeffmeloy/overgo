package repoanalysis

import "go/ast"

const (
	agentToolPackage         = "internal/agenttool"
	capabilityRuntimePackage = "internal/capabilityruntime"
	runRecordPackage         = "internal/runrecord"

	agentToolImport         = "overgo/internal/agenttool"
	artifactImport          = "overgo/internal/artifact"
	capabilityRuntimeImport = "overgo/internal/capabilityruntime"
)

// auditCapabilityAndToolAuthorities holds capability construction, catalog
// activation, and native tool execution to their existing Go owners. The
// call-site rules deliberately use exact enclosing functions and counts: an
// allowed file cannot silently accumulate a second bypass, and a displaced
// path leaves a stale allowance that must be removed with it.
func auditCapabilityAndToolAuthorities(sources []productionAuthoritySource, report *ProductionAuthorityReport) {
	requireAuthorityOwners(sources, report,
		authorityOwnerRule{Family: "capability", Kind: "type", Symbol: "CapabilityIdentity", Owner: runRecordPackage},
		authorityOwnerRule{Family: "capability", Kind: "method", Symbol: "Identify", Receiver: "CapabilityIdentity", Owner: runRecordPackage},
		authorityOwnerRule{Family: "capability", Kind: "func", Symbol: "RequireCapabilityIdentity", Owner: runRecordPackage},
		authorityOwnerRule{Family: "capability", Kind: "type", Symbol: "ExactCapabilityPlacement", Owner: capabilityRuntimePackage},
		authorityOwnerRule{Family: "capability", Kind: "func", Symbol: "ResolveExactCapabilityPlacement", Owner: capabilityRuntimePackage},
		authorityOwnerRule{Family: "capability", Kind: "type", Symbol: "ExecutorCatalog", Owner: capabilityRuntimePackage},
		authorityOwnerRule{Family: "capability", Kind: "method", Symbol: "Execute", Receiver: "ExecutorCatalog", Owner: capabilityRuntimePackage},
		authorityOwnerRule{Family: "capability", Kind: "method", Symbol: "ExecutePlaced", Receiver: "ExecutorCatalog", Owner: capabilityRuntimePackage},
		authorityOwnerRule{Family: "tool", Kind: "type", Symbol: "Manual", Owner: agentToolPackage},
		authorityOwnerRule{Family: "tool", Kind: "type", Symbol: "InvocationEffect", Owner: agentToolPackage},
		authorityOwnerRule{Family: "tool", Kind: "type", Symbol: "CatalogSnapshot", Owner: agentToolPackage},
		authorityOwnerRule{Family: "tool", Kind: "type", Symbol: "CandidateCatalog", Owner: agentToolPackage},
		authorityOwnerRule{Family: "tool", Kind: "var", Symbol: "ActiveCatalogAlias", Owner: agentToolPackage},
		authorityOwnerRule{Family: "tool", Kind: "func", Symbol: "NewManual", Owner: agentToolPackage},
		authorityOwnerRule{Family: "tool", Kind: "func", Symbol: "RequireManual", Owner: agentToolPackage},
		authorityOwnerRule{Family: "tool", Kind: "func", Symbol: "DeriveInvocationEffect", Owner: agentToolPackage},
		authorityOwnerRule{Family: "tool", Kind: "func", Symbol: "CheckArgvAuthority", Owner: agentToolPackage},
		authorityOwnerRule{Family: "tool", Kind: "func", Symbol: "ActivateCandidateCatalog", Owner: agentToolPackage},
		authorityOwnerRule{Family: "tool", Kind: "method", Symbol: "InvokeWithEffect", Receiver: "Executor", Owner: agentToolPackage},
		authorityOwnerRule{Family: "tool", Kind: "method", Symbol: "OpenStream", Receiver: "Executor", Owner: agentToolPackage},
	)

	manualNew := map[authoritySiteKey][]int{}
	manualRequire := map[authoritySiteKey][]int{}
	executorCatalogExecute := map[authoritySiteKey][]int{}
	placements := map[authoritySiteKey][]int{}
	activeCatalogBindings := map[authoritySiteKey][]int{}
	adapterInvoke := map[authoritySiteKey][]int{}
	invoke := map[authoritySiteKey][]int{}
	invokeWithEffect := map[authoritySiteKey][]int{}
	manualCatalogBatch := map[authoritySiteKey][]int{}
	argvAuthority := map[authoritySiteKey][]int{}
	openStream := map[authoritySiteKey][]int{}

	for _, source := range sources {
		inspectProductionAuthoritySource(source, func(function string, node ast.Node) {
			switch typed := node.(type) {
			case *ast.CallExpr:
				selector, selected := typed.Fun.(*ast.SelectorExpr)
				if selected && source.Package == agentToolPackage {
					if base, ok := selector.X.(*ast.Ident); ok && base.Name == "manualCodec" {
						switch selector.Sel.Name {
						case "New":
							recordAuthoritySite(manualNew, source, function, typed.Pos())
						case "Require":
							recordAuthoritySite(manualRequire, source, function, typed.Pos())
						}
					}
					if selector.Sel.Name == "invoke" {
						recordAuthoritySite(adapterInvoke, source, function, typed.Pos())
					}
				}
				if selected && selector.Sel.Name == "Execute" &&
					(source.Package == capabilityRuntimePackage || sourceImportsAuthority(source, capabilityRuntimeImport)) {
					recordAuthoritySite(executorCatalogExecute, source, function, typed.Pos())
				}
				if selected && (source.Package == agentToolPackage || sourceImportsAuthority(source, agentToolImport)) {
					switch selector.Sel.Name {
					case "Invoke":
						recordAuthoritySite(invoke, source, function, typed.Pos())
					case "InvokeWithEffect":
						recordAuthoritySite(invokeWithEffect, source, function, typed.Pos())
					case "OpenStream":
						recordAuthoritySite(openStream, source, function, typed.Pos())
					}
				}
				if authorityCallNames(source, typed.Fun, agentToolPackage, agentToolImport, "CheckArgvAuthority") {
					recordAuthoritySite(argvAuthority, source, function, typed.Pos())
				}
				if identifier, ok := typed.Fun.(*ast.Ident); ok &&
					source.Package == agentToolPackage && identifier.Name == "manualCatalogBatch" {
					recordAuthoritySite(manualCatalogBatch, source, function, typed.Pos())
				}
			case *ast.CompositeLit:
				importPath, name := authorityCompositeType(source, typed.Type)
				if len(typed.Elts) != 0 && name == "ExactCapabilityPlacement" &&
					(importPath == capabilityRuntimeImport || importPath == "" && source.Package == capabilityRuntimePackage) {
					recordAuthoritySite(placements, source, function, typed.Pos())
				}
				if name == "AliasBinding" && importPath == artifactImport &&
					compositeBindsActiveCatalog(source, typed) {
					recordAuthoritySite(activeCatalogBindings, source, function, typed.Pos())
				}
			case *ast.AssignStmt:
				for index, right := range typed.Rhs {
					if !activeCatalogAliasExpression(source, right) {
						continue
					}
					var target ast.Expr
					switch {
					case len(typed.Lhs) == len(typed.Rhs) && index < len(typed.Lhs):
						target = typed.Lhs[index]
					case len(typed.Lhs) == oneAuthoritySite:
						for _, candidate := range typed.Lhs {
							target = candidate
						}
					default:
						continue
					}
					if selector, ok := target.(*ast.SelectorExpr); ok && selector.Sel.Name == "Name" {
						recordAuthoritySite(activeCatalogBindings, source, function, typed.Pos())
					}
				}
			}
		})
	}

	verifyAuthoritySites(report, "tool", "manualCodec.New", manualNew, []authorityAllowance{{
		File: "internal/agenttool/manual.go", Function: "NewManual", Count: oneAuthoritySite,
	}})
	verifyAuthoritySites(report, "tool", "manualCodec.Require", manualRequire, []authorityAllowance{{
		File: "internal/agenttool/manual.go", Function: "RequireManual", Count: oneAuthoritySite,
	}})
	// The recipe command's verify and run verbs and the server's store
	// generation workspace execute a resolved local activation through the
	// shared media capability catalog; each is one declared site.
	verifyAuthoritySites(report, "capability", "ExecutorCatalog.Execute", executorCatalogExecute, []authorityAllowance{
		{File: "internal/capabilityruntime/placement.go", Function: "ExecutorCatalog.ExecutePlaced", Count: oneAuthoritySite},
		{File: "cmd/recipe/main.go", Function: "verifyCapability", Count: oneAuthoritySite},
		{File: "cmd/recipe/main.go", Function: "executeCapability", Count: oneAuthoritySite},
		{File: "internal/server/generation_store_workspace.go", Function: "StoreGenerationWorkspace.ExecuteWorkflow", Count: oneAuthoritySite},
	})
	verifyAuthoritySites(report, "capability", "ExactCapabilityPlacement construction", placements, []authorityAllowance{{
		File: "internal/capabilityruntime/placement.go", Function: "ResolveExactCapabilityPlacement", Count: oneAuthoritySite,
	}})
	verifyAuthoritySites(report, "tool", "ActiveCatalogAlias binding", activeCatalogBindings, []authorityAllowance{{
		File: "internal/agenttool/candidate.go", Function: "ActivateCandidateCatalog", Count: oneAuthoritySite,
	}})
	verifyAuthoritySites(report, "tool", "transportAdapter.invoke", adapterInvoke, []authorityAllowance{{
		File: "internal/agenttool/transport.go", Function: "Executor.InvokeWithEffect", Count: oneAuthoritySite,
	}})
	verifyAuthoritySites(report, "tool", "Executor.Invoke", invoke, []authorityAllowance{
		{File: "cmd/agent-tool/main.go", Function: "run", Count: oneAuthoritySite},
		{File: "internal/workflowruntime/automation.go", Function: "AutomationRuntime.deliverAutomation", Count: oneAuthoritySite},
		{File: "internal/workflowruntime/tool_workflow.go", Function: "toolWorkflowAdapter", Count: oneAuthoritySite},
	})
	verifyAuthoritySites(report, "tool", "Executor.InvokeWithEffect", invokeWithEffect, []authorityAllowance{
		{File: "internal/agentloop/coordinator.go", Function: "Coordinator.propose", Count: oneAuthoritySite},
		{File: "internal/agenttool/proxy.go", Function: "RegisterCapabilityProxy", Count: oneAuthoritySite},
		{File: "internal/agenttool/transport.go", Function: "Executor.Invoke", Count: oneAuthoritySite},
	})
	verifyAuthoritySites(report, "tool", "manualCatalogBatch", manualCatalogBatch, []authorityAllowance{
		{File: "internal/agenttool/candidate.go", Function: "ActivateCandidateCatalog", Count: oneAuthoritySite},
		{File: "internal/agenttool/catalog.go", Function: "PublishManualCatalog", Count: oneAuthoritySite},
	})
	verifyAuthoritySites(report, "tool", "CheckArgvAuthority", argvAuthority, []authorityAllowance{
		{File: "cmd/agent-tool/main.go", Function: "run", Count: oneAuthoritySite},
		{File: "internal/agentloop/coordinator.go", Function: "Coordinator.propose", Count: oneAuthoritySite},
		{File: "internal/agenttool/candidate.go", Function: "inspectCandidate", Count: oneAuthoritySite},
		{File: "internal/agenttool/catalog.go", Function: "PublishManualCatalog", Count: oneAuthoritySite},
		{File: "internal/agenttool/proxy.go", Function: "RegisterCapabilityProxy", Count: oneAuthoritySite},
		{File: "internal/workflowruntime/automation.go", Function: "AutomationRuntime.deliverAutomation", Count: oneAuthoritySite},
		{File: "internal/workflowruntime/tool_workflow.go", Function: "toolWorkflowAdapter", Count: oneAuthoritySite},
	})
	verifyAuthoritySites(report, "tool", "Executor.OpenStream", openStream, nil)
}

func inspectProductionAuthoritySource(source productionAuthoritySource, visit func(string, ast.Node)) {
	for _, declaration := range source.SyntaxFile.Decls {
		function, ok := declaration.(*ast.FuncDecl)
		name := "<file>"
		if ok {
			name = authorityFunctionName(function)
		}
		ast.Inspect(declaration, func(node ast.Node) bool {
			if node != nil {
				visit(name, node)
			}
			return true
		})
	}
}

func sourceImportsAuthority(source productionAuthoritySource, importPath string) bool {
	for _, imported := range source.Imports {
		if imported == importPath {
			return true
		}
	}
	return false
}

func authorityCallNames(
	source productionAuthoritySource,
	expression ast.Expr,
	localPackage, importPath, name string,
) bool {
	switch typed := expression.(type) {
	case *ast.Ident:
		return source.Package == localPackage && typed.Name == name
	case *ast.SelectorExpr:
		return typed.Sel.Name == name && authoritySelectorImport(source, typed) == importPath
	default:
		return false
	}
}

func compositeBindsActiveCatalog(source productionAuthoritySource, literal *ast.CompositeLit) bool {
	for _, element := range literal.Elts {
		field, ok := element.(*ast.KeyValueExpr)
		if !ok {
			continue
		}
		name, ok := field.Key.(*ast.Ident)
		if ok && name.Name == "Name" && activeCatalogAliasExpression(source, field.Value) {
			return true
		}
	}
	return false
}

func activeCatalogAliasExpression(source productionAuthoritySource, expression ast.Expr) bool {
	switch typed := expression.(type) {
	case *ast.Ident:
		return source.Package == agentToolPackage && typed.Name == "ActiveCatalogAlias"
	case *ast.SelectorExpr:
		return typed.Sel.Name == "ActiveCatalogAlias" && authoritySelectorImport(source, typed) == agentToolImport
	default:
		return false
	}
}
