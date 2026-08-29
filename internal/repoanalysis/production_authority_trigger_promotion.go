package repoanalysis

import "go/ast"

const (
	runrecordImport       = "overgo/internal/runrecord"
	compositionImport     = "overgo/internal/composition"
	workflowruntimeImport = "overgo/internal/workflowruntime"
	inferenceImport       = "overgo/internal/inference"
	loopImport            = "overgo/internal/loop"
)

// auditTriggerAndPromotionAuthorities pins the constructors and callers that
// may originate execution causality or select production model artifacts.
// The audit is deliberately syntax-only: it is cheap enough to run for every
// candidate snapshot and conservative when a protected selector is ambiguous.
func auditTriggerAndPromotionAuthorities(sources []productionAuthoritySource, report *ProductionAuthorityReport) {
	requireAuthorityOwners(sources, report,
		authorityOwnerRule{Family: "trigger", Kind: "type", Symbol: "CausalContext", Owner: "internal/runrecord"},
		authorityOwnerRule{Family: "trigger", Kind: "func", Symbol: "NewCausalRoot", Owner: "internal/runrecord"},
		authorityOwnerRule{Family: "trigger", Kind: "method", Symbol: "Derive", Owner: "internal/runrecord", Receiver: "CausalContext"},
		authorityOwnerRule{Family: "trigger", Kind: "var", Symbol: "TriggerRegistry", Owner: "internal/runrecord"},
		authorityOwnerRule{Family: "trigger", Kind: "func", Symbol: "ValidateTriggerRegistry", Owner: "internal/runrecord"},
		authorityOwnerRule{Family: "trigger", Kind: "var", Symbol: "ExecutionTriggers", Owner: "internal/workflowruntime"},
		authorityOwnerRule{Family: "trigger", Kind: "method", Symbol: "DispatchWebhook", Owner: "internal/workflowruntime", Receiver: "AutomationRuntime"},
		authorityOwnerRule{Family: "trigger", Kind: "method", Symbol: "admitCausal", Owner: "internal/workflowruntime", Receiver: "AutomationRuntime"},
		authorityOwnerRule{Family: "trigger", Kind: "type", Symbol: "StimulusFollowupAuthority", Owner: "internal/runrecord"},
		authorityOwnerRule{Family: "trigger", Kind: "method", Symbol: "Admit", Owner: "internal/runrecord", Receiver: "StimulusFollowupAuthority"},
		authorityOwnerRule{Family: "trigger", Kind: "func", Symbol: "ReconcileLateStimuli", Owner: "internal/workflowruntime"},
		authorityOwnerRule{Family: "promotion", Kind: "func", Symbol: "ActiveCompositeGeneration", Owner: "internal/composition"},
		authorityOwnerRule{Family: "promotion", Kind: "type", Symbol: "CompositeGenerationSurface", Owner: "internal/inference"},
		authorityOwnerRule{Family: "promotion", Kind: "func", Symbol: "ResolveCompositeGenerationSurface", Owner: "internal/inference"},
		authorityOwnerRule{Family: "promotion", Kind: "method", Symbol: "PromotedGeneration", Owner: "internal/inference", Receiver: "ProductionComposition"},
		authorityOwnerRule{Family: "promotion", Kind: "type", Symbol: "ModelBuildSession", Owner: "internal/workflowruntime"},
		authorityOwnerRule{Family: "promotion", Kind: "func", Symbol: "ExecuteModelBuild", Owner: "internal/workflowruntime"},
		authorityOwnerRule{Family: "promotion", Kind: "type", Symbol: "StrategyComparison", Owner: "internal/loop"},
		authorityOwnerRule{Family: "promotion", Kind: "func", Symbol: "CompareStrategyExperiment", Owner: "internal/loop"},
	)

	newCausalRoots := map[authoritySiteKey][]int{}
	causalDerivations := map[authoritySiteKey][]int{}
	causalConstructions := map[authoritySiteKey][]int{}
	webhookDispatches := map[authoritySiteKey][]int{}
	causalAdmissions := map[authoritySiteKey][]int{}
	stimulusAdmissions := map[authoritySiteKey][]int{}
	activeCompositeResolutions := map[authoritySiteKey][]int{}
	compositeSurfaceConstructions := map[authoritySiteKey][]int{}
	modelPromotions := map[authoritySiteKey][]int{}
	strategyComparisons := map[authoritySiteKey][]int{}
	strategyWinnerAssignments := map[authoritySiteKey][]int{}

	for _, source := range sources {
		// Package-level initializers have no enclosing FuncDecl, but can still
		// construct protected values (and are particularly easy bypasses to
		// overlook in a call-site-only audit).
		for _, declaration := range source.SyntaxFile.Decls {
			if _, isFunction := declaration.(*ast.FuncDecl); isFunction {
				continue
			}
			ast.Inspect(declaration, func(node ast.Node) bool {
				literal, ok := node.(*ast.CompositeLit)
				if !ok || len(literal.Elts) == 0 {
					return true
				}
				typeRef := triggerPromotionTypeReference(source, literal.Type)
				switch {
				case triggerPromotionTypeMatches(source, typeRef, runrecordImport, "internal/runrecord", "CausalContext"):
					recordAuthoritySite(causalConstructions, source, "<file>", literal.Pos())
				case triggerPromotionTypeMatches(source, typeRef, inferenceImport, "internal/inference", "CompositeGenerationSurface"):
					recordAuthoritySite(compositeSurfaceConstructions, source, "<file>", literal.Pos())
				case triggerPromotionTypeMatches(source, typeRef, loopImport, "internal/loop", "StrategyComparison"):
					recordAuthoritySite(strategyComparisons, source, "<file>", literal.Pos())
				}
				return true
			})
		}
		for _, declaration := range source.SyntaxFile.Decls {
			function, ok := declaration.(*ast.FuncDecl)
			if !ok || function.Body == nil {
				continue
			}
			functionName := authorityFunctionName(function)
			locals := triggerPromotionLocalTypes(source, function)
			ast.Inspect(function.Body, func(node ast.Node) bool {
				switch typed := node.(type) {
				case *ast.CallExpr:
					auditTriggerPromotionCall(
						source, functionName, locals, typed,
						newCausalRoots, causalDerivations, webhookDispatches,
						causalAdmissions, stimulusAdmissions, activeCompositeResolutions,
					)
				case *ast.CompositeLit:
					if len(typed.Elts) == 0 {
						return true
					}
					typeRef := triggerPromotionTypeReference(source, typed.Type)
					switch {
					case triggerPromotionTypeMatches(source, typeRef, runrecordImport, "internal/runrecord", "CausalContext"):
						recordAuthoritySite(causalConstructions, source, functionName, typed.Pos())
					case triggerPromotionTypeMatches(source, typeRef, inferenceImport, "internal/inference", "CompositeGenerationSurface"):
						recordAuthoritySite(compositeSurfaceConstructions, source, functionName, typed.Pos())
					case triggerPromotionTypeMatches(source, typeRef, loopImport, "internal/loop", "StrategyComparison"):
						recordAuthoritySite(strategyComparisons, source, functionName, typed.Pos())
					}
				case *ast.SelectorExpr:
					if typed.Sel.Name == "Promote" && triggerPromotionIdentName(typed.X) == "session" &&
						triggerPromotionTypeMatches(source, triggerPromotionExpressionType(source, typed.X, locals), workflowruntimeImport, "internal/workflowruntime", "ModelBuildSession") {
						recordAuthoritySite(modelPromotions, source, functionName, typed.Sel.Pos())
					}
				case *ast.AssignStmt:
					for _, target := range typed.Lhs {
						selector, ok := target.(*ast.SelectorExpr)
						if !ok || selector.Sel.Name != "Winner" ||
							!triggerPromotionTypeMatches(source, triggerPromotionExpressionType(source, selector.X, locals), loopImport, "internal/loop", "StrategyComparison") {
							continue
						}
						recordAuthoritySite(strategyWinnerAssignments, source, functionName, selector.Sel.Pos())
					}
				}
				return true
			})
		}
	}

	verifyAuthoritySites(report, "trigger", "runrecord.NewCausalRoot", newCausalRoots, []authorityAllowance{
		{File: "internal/runrecord/stimulus_followup.go", Function: "StimulusFollowupAuthority.Admit", Count: oneAuthoritySite},
		{File: "internal/workflowruntime/automation_webhook.go", Function: "webhookCausalContext", Count: twoAuthoritySites},
	})
	verifyAuthoritySites(report, "trigger", "runrecord.CausalContext.Derive", causalDerivations, []authorityAllowance{
		{File: "internal/runrecord/stimulus_followup.go", Function: "StimulusFollowupAuthority.Admit", Count: oneAuthoritySite},
		{File: "internal/workflowruntime/delegation.go", Function: "CompiledDelegation.DeriveCausal", Count: oneAuthoritySite},
	})
	verifyAuthoritySites(report, "trigger", "runrecord.CausalContext construction", causalConstructions, []authorityAllowance{
		{File: "internal/runrecord/causal_context.go", Function: "NewCausalRoot", Count: oneAuthoritySite},
		{File: "internal/runrecord/causal_context.go", Function: "CausalContext.Derive", Count: oneAuthoritySite},
	})
	verifyAuthoritySites(report, "trigger", "workflowruntime.AutomationRuntime.DispatchWebhook", webhookDispatches, []authorityAllowance{
		{File: "internal/server/automation_webhook.go", Function: "AutomationWorkspace.receiveAutomationWebhook", Count: oneAuthoritySite},
	})
	verifyAuthoritySites(report, "trigger", "workflowruntime.AutomationRuntime.admitCausal", causalAdmissions, []authorityAllowance{
		{File: "internal/workflowruntime/automation.go", Function: "AutomationRuntime.admit", Count: oneAuthoritySite},
		{File: "internal/workflowruntime/automation_webhook.go", Function: "AutomationRuntime.DispatchWebhook", Count: oneAuthoritySite},
	})
	verifyAuthoritySites(report, "trigger", "runrecord.StimulusFollowupAuthority.Admit", stimulusAdmissions, []authorityAllowance{
		{File: "internal/workflowruntime/stimulus_reconciliation.go", Function: "ReconcileLateStimuli", Count: oneAuthoritySite},
	})
	verifyAuthoritySites(report, "promotion", "composition.ActiveCompositeGeneration", activeCompositeResolutions, []authorityAllowance{
		{File: "internal/inference/composite_generation_surface.go", Function: "ResolveCompositeGenerationSurface", Count: oneAuthoritySite},
		{File: "internal/inference/composite_generation_surface.go", Function: "ProductionComposition.PromotedGeneration", Count: oneAuthoritySite},
	})
	verifyAuthoritySites(report, "promotion", "inference.CompositeGenerationSurface construction", compositeSurfaceConstructions, []authorityAllowance{
		{File: "internal/inference/composite_generation_surface.go", Function: "compositeGenerationSurface", Count: oneAuthoritySite},
	})
	verifyAuthoritySites(report, "promotion", "workflowruntime.ModelBuildSession.Promote", modelPromotions, []authorityAllowance{
		{File: "internal/workflowruntime/model_builder.go", Function: "ExecuteModelBuild", Count: oneAuthoritySite},
	})
	verifyAuthoritySites(report, "promotion", "loop.StrategyComparison construction", strategyComparisons, []authorityAllowance{
		{File: "internal/loop/experiment.go", Function: "CompareStrategyExperiment", Count: oneAuthoritySite},
	})
	verifyAuthoritySites(report, "promotion", "loop.StrategyComparison.Winner assignment", strategyWinnerAssignments, []authorityAllowance{
		{File: "internal/loop/experiment.go", Function: "CompareStrategyExperiment", Count: oneAuthoritySite},
	})
}

func auditTriggerPromotionCall(
	source productionAuthoritySource,
	function string,
	locals map[string]triggerPromotionType,
	call *ast.CallExpr,
	newCausalRoots,
	causalDerivations,
	webhookDispatches,
	causalAdmissions,
	stimulusAdmissions,
	activeCompositeResolutions map[authoritySiteKey][]int,
) {
	switch called := call.Fun.(type) {
	case *ast.Ident:
		switch {
		case called.Name == "NewCausalRoot" && source.Package == "internal/runrecord":
			recordAuthoritySite(newCausalRoots, source, function, called.Pos())
		case called.Name == "ActiveCompositeGeneration" && source.Package == "internal/composition":
			recordAuthoritySite(activeCompositeResolutions, source, function, called.Pos())
		}
	case *ast.SelectorExpr:
		selectorImport := authoritySelectorImport(source, called)
		switch {
		case called.Sel.Name == "NewCausalRoot" && selectorImport == runrecordImport:
			recordAuthoritySite(newCausalRoots, source, function, called.Sel.Pos())
		case called.Sel.Name == "Derive" && triggerPromotionMayUseRunrecord(source):
			recordAuthoritySite(causalDerivations, source, function, called.Sel.Pos())
		case called.Sel.Name == "DispatchWebhook":
			recordAuthoritySite(webhookDispatches, source, function, called.Sel.Pos())
		case called.Sel.Name == "admitCausal":
			recordAuthoritySite(causalAdmissions, source, function, called.Sel.Pos())
		case called.Sel.Name == "Admit" && triggerPromotionTypeMatches(
			source,
			triggerPromotionExpressionType(source, called.X, locals),
			runrecordImport,
			"internal/runrecord",
			"StimulusFollowupAuthority",
		):
			recordAuthoritySite(stimulusAdmissions, source, function, called.Sel.Pos())
		case called.Sel.Name == "ActiveCompositeGeneration" && selectorImport == compositionImport:
			recordAuthoritySite(activeCompositeResolutions, source, function, called.Sel.Pos())
		}
	}
}

func triggerPromotionMayUseRunrecord(source productionAuthoritySource) bool {
	if source.Package == "internal/runrecord" {
		return true
	}
	for _, importPath := range source.Imports {
		if importPath == runrecordImport {
			return true
		}
	}
	return false
}

type triggerPromotionType struct {
	Import string
	Name   string
}

func triggerPromotionLocalTypes(source productionAuthoritySource, function *ast.FuncDecl) map[string]triggerPromotionType {
	locals := map[string]triggerPromotionType{}
	if function.Type.Params != nil {
		for _, field := range function.Type.Params.List {
			typeRef := triggerPromotionTypeReference(source, field.Type)
			for _, name := range field.Names {
				locals[name.Name] = typeRef
			}
		}
	}
	ast.Inspect(function.Body, func(node ast.Node) bool {
		switch typed := node.(type) {
		case *ast.DeclStmt:
			declaration, ok := typed.Decl.(*ast.GenDecl)
			if !ok {
				return true
			}
			for _, spec := range declaration.Specs {
				value, ok := spec.(*ast.ValueSpec)
				if !ok {
					continue
				}
				for index, name := range value.Names {
					if value.Type != nil {
						locals[name.Name] = triggerPromotionTypeReference(source, value.Type)
					} else if index < len(value.Values) {
						locals[name.Name] = triggerPromotionExpressionType(source, value.Values[index], locals)
					}
				}
			}
		case *ast.AssignStmt:
			if typed.Tok.String() != ":=" {
				return true
			}
			for index, target := range typed.Lhs {
				name, ok := target.(*ast.Ident)
				if ok && index < len(typed.Rhs) {
					locals[name.Name] = triggerPromotionExpressionType(source, typed.Rhs[index], locals)
				}
			}
		}
		return true
	})
	return locals
}

func triggerPromotionTypeReference(source productionAuthoritySource, expression ast.Expr) triggerPromotionType {
	switch typed := expression.(type) {
	case *ast.Ident:
		return triggerPromotionType{Name: typed.Name}
	case *ast.SelectorExpr:
		return triggerPromotionType{Import: authoritySelectorImport(source, typed), Name: typed.Sel.Name}
	case *ast.StarExpr:
		return triggerPromotionTypeReference(source, typed.X)
	case *ast.IndexExpr:
		return triggerPromotionTypeReference(source, typed.X)
	case *ast.IndexListExpr:
		return triggerPromotionTypeReference(source, typed.X)
	default:
		return triggerPromotionType{}
	}
}

func triggerPromotionExpressionType(
	source productionAuthoritySource,
	expression ast.Expr,
	locals map[string]triggerPromotionType,
) triggerPromotionType {
	switch typed := expression.(type) {
	case *ast.Ident:
		return locals[typed.Name]
	case *ast.CompositeLit:
		return triggerPromotionTypeReference(source, typed.Type)
	case *ast.ParenExpr:
		return triggerPromotionExpressionType(source, typed.X, locals)
	case *ast.UnaryExpr:
		return triggerPromotionExpressionType(source, typed.X, locals)
	case *ast.CallExpr:
		return triggerPromotionTypeReference(source, typed.Fun)
	default:
		return triggerPromotionType{}
	}
}

func triggerPromotionTypeMatches(
	source productionAuthoritySource,
	typeRef triggerPromotionType,
	importPath, owner, name string,
) bool {
	if typeRef.Name != name {
		return false
	}
	return typeRef.Import == importPath || (typeRef.Import == "" && source.Package == owner)
}

func triggerPromotionIdentName(expression ast.Expr) string {
	identifier, _ := expression.(*ast.Ident)
	if identifier == nil {
		return ""
	}
	return identifier.Name
}
