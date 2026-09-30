package repoanalysis

import "go/ast"

func auditStorageAndProcessAuthorities(sources []productionAuthoritySource, report *ProductionAuthorityReport) {
	producerCommitSites := map[authoritySiteKey][]int{}
	producerMintSites := map[authoritySiteKey][]int{}
	processSites := map[authoritySiteKey][]int{}
	for _, source := range sources {
		visitProductionAuthorityCalls(source, func(function string, call *ast.CallExpr, selector *ast.SelectorExpr) {
			switch selector.Sel.Name {
			case "CommitAs":
				recordAuthoritySite(producerCommitSites, source, function, call.Pos())
			case "NewProducer":
				recordAuthoritySite(producerMintSites, source, function, call.Pos())
			}
			if isProcessPrimitive(source, selector) {
				recordAuthoritySite(processSites, source, function, call.Pos())
			}
		})
		visitProductionAuthorityComposites(source, func(function string, literal *ast.CompositeLit) {
			importPath, name := authorityCompositeType(source, literal.Type)
			if importPath == "os/exec" && name == "Cmd" {
				recordAuthoritySite(processSites, source, function, literal.Pos())
			}
		})
	}

	// A producer's door and its mint are the capability to commit a guarded
	// kind: each site is a reviewed entry here, never a caller's choice.
	verifyAuthoritySites(report, "storage", ".CommitAs", producerCommitSites, []authorityAllowance{
		{File: "cmd/overgodb/precheck.go", Count: oneAuthoritySite},
		{File: "internal/overgodb/rebuild.go", Count: oneAuthoritySite},
		{File: "internal/overgodb/store.go", Function: "Store.Commit", Count: oneAuthoritySite},
	})
	verifyAuthoritySites(report, "storage", "overgodb.NewProducer", producerMintSites, []authorityAllowance{
		{File: "cmd/overgodb/precheck.go", Count: oneAuthoritySite},
	})
	verifyAuthoritySites(report, "process", "external process primitive", processSites, processPrimitiveAllowances)
}

func visitProductionAuthorityCalls(
	source productionAuthoritySource,
	visit func(string, *ast.CallExpr, *ast.SelectorExpr),
) {
	for _, declaration := range source.SyntaxFile.Decls {
		function, _ := declaration.(*ast.FuncDecl)
		name := authorityFunctionName(function)
		ast.Inspect(declaration, func(node ast.Node) bool {
			call, ok := node.(*ast.CallExpr)
			if !ok {
				return true
			}
			selector, ok := call.Fun.(*ast.SelectorExpr)
			if ok {
				visit(name, call, selector)
			}
			return true
		})
	}
}

func visitProductionAuthorityComposites(
	source productionAuthoritySource,
	visit func(string, *ast.CompositeLit),
) {
	for _, declaration := range source.SyntaxFile.Decls {
		function, _ := declaration.(*ast.FuncDecl)
		name := authorityFunctionName(function)
		ast.Inspect(declaration, func(node ast.Node) bool {
			literal, ok := node.(*ast.CompositeLit)
			if ok {
				visit(name, literal)
			}
			return true
		})
	}
}

func isProcessPrimitive(source productionAuthoritySource, selector *ast.SelectorExpr) bool {
	importPath := authoritySelectorImport(source, selector)
	if importPath == "os/exec" && (selector.Sel.Name == "Command" || selector.Sel.Name == "CommandContext") {
		return true
	}
	if (importPath == "os" || importPath == "syscall") && selector.Sel.Name == "StartProcess" {
		return true
	}
	process, ok := selector.X.(*ast.SelectorExpr)
	return ok && process.Sel.Name == "Process" &&
		(selector.Sel.Name == "Kill" || selector.Sel.Name == "Signal" || selector.Sel.Name == "Release")
}

var processPrimitiveAllowances = []authorityAllowance{
	{File: "cmd/advisories/main.go", Count: oneAuthoritySite},
	{File: "cmd/build-kernels/main.go", Count: twoAuthoritySites},
	{File: "cmd/composite-generation-lane/run_windows.go", Count: oneAuthoritySite},
	{File: "cmd/eval-lane/main.go", Count: oneAuthoritySite},
	{File: "cmd/loophook/main.go", Count: sixAuthoritySites},
	{File: "cmd/plan/git.go", Count: oneAuthoritySite},
	{File: "cmd/release/main.go", Count: fourAuthoritySites},
	{File: "cmd/reverify/main.go", Count: twoAuthoritySites},
	{File: "cmd/sbom/main.go", Count: oneAuthoritySite},
	{File: "internal/clioptions/command.go", Count: oneAuthoritySite},
	{File: "internal/processcontrol/supervisor.go", Function: "Start", Count: twoAuthoritySites},
	{File: "internal/runrecord/verifying_commit.go", Count: threeAuthoritySites},
	{File: "internal/testprocess/process.go", Count: twoAuthoritySites},
}
