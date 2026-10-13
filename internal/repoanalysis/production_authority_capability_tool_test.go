package repoanalysis

import (
	"go/ast"
	"slices"
)

const (
	agentToolPackage         = "internal/agenttool"
	capabilityRuntimePackage = "internal/capabilityruntime"

	agentToolImport         = "overgo/internal/agenttool"
	artifactImport          = "overgo/internal/artifact"
	capabilityRuntimeImport = "overgo/internal/capabilityruntime"
)

// auditCapabilityAndToolAuthorities holds exact capability placement and
// active tool-catalog binding to their resolvers. The call-site rules use
// exact enclosing functions and counts: an allowed file cannot silently
// accumulate a second bypass, and a displaced path leaves a stale allowance
// that must be removed with it.
func auditCapabilityAndToolAuthorities(sources []productionAuthoritySource, report *ProductionAuthorityReport) {
	placements := map[authoritySiteKey][]int{}
	activeCatalogBindings := map[authoritySiteKey][]int{}

	for _, source := range sources {
		inspectProductionAuthoritySource(source, func(function string, node ast.Node) {
			switch typed := node.(type) {
			case *ast.CallExpr:
				// The alias move has one owner in artifact; a binding of the
				// active catalog through it is the same site as the literal.
				for _, owner := range []string{"AliasMove", "AliasMoveFrom", "MoveAlias", "AliasRemoval"} {
					if authorityCallNames(source, typed.Fun, "artifact", artifactImport, owner) &&
						slices.ContainsFunc(typed.Args, func(argument ast.Expr) bool { return activeCatalogAliasExpression(source, argument) }) {
						recordAuthoritySite(activeCatalogBindings, source, function, typed.Pos())
					}
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

	verifyAuthoritySites(report, "capability", "ExactCapabilityPlacement construction", placements, []authorityAllowance{{
		File: "internal/capabilityruntime/placement.go", Function: "ResolveExactCapabilityPlacement", Count: oneAuthoritySite,
	}})
	verifyAuthoritySites(report, "tool", "ActiveCatalogAlias binding", activeCatalogBindings, []authorityAllowance{{
		File: "internal/agenttool/candidate.go", Function: "ActivateCandidateCatalog", Count: oneAuthoritySite,
	}})
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
