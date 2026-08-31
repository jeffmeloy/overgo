package repoanalysis

import "go/ast"

func auditStorageAndProcessAuthorities(sources []productionAuthoritySource, report *ProductionAuthorityReport) {
	requireAuthorityOwners(sources, report,
		authorityOwnerRule{Family: "storage", Kind: "method", Symbol: "Commit", Owner: "internal/overgodb", Receiver: "Store"},
		authorityOwnerRule{Family: "storage", Kind: "method", Symbol: "commitPublished", Owner: "internal/overgodb", Receiver: "Store"},
		authorityOwnerRule{Family: "storage", Kind: "type", Symbol: "commitCoordinator", Owner: "internal/overgodb"},
		authorityOwnerRule{Family: "storage", Kind: "method", Symbol: "commit", Owner: "internal/overgodb", Receiver: "commitCoordinator"},
		authorityOwnerRule{Family: "storage", Kind: "method", Symbol: "append", Owner: "internal/overgodb", Receiver: "recordLog"},
		authorityOwnerRule{Family: "storage", Kind: "method", Symbol: "prepare", Owner: "internal/overgodb", Receiver: "blobStore"},
		authorityOwnerRule{Family: "process", Kind: "type", Symbol: "Receipt", Owner: "internal/processcontrol"},
		authorityOwnerRule{Family: "process", Kind: "type", Symbol: "Supervised", Owner: "internal/processcontrol"},
		authorityOwnerRule{Family: "process", Kind: "func", Symbol: "Start", Owner: "internal/processcontrol"},
		authorityOwnerRule{Family: "process", Kind: "method", Symbol: "Interrupt", Owner: "internal/processcontrol", Receiver: "Supervised"},
		authorityOwnerRule{Family: "process", Kind: "method", Symbol: "Terminate", Owner: "internal/processcontrol", Receiver: "Supervised"},
		authorityOwnerRule{Family: "process", Kind: "method", Symbol: "Wait", Owner: "internal/processcontrol", Receiver: "Supervised"},
		authorityOwnerRule{Family: "process", Kind: "method", Symbol: "Exited", Owner: "internal/processcontrol", Receiver: "Supervised"},
	)

	commitSites := map[authoritySiteKey][]int{}
	commitPublishedSites := map[authoritySiteKey][]int{}
	appendSites := map[authoritySiteKey][]int{}
	prepareSites := map[authoritySiteKey][]int{}
	processSites := map[authoritySiteKey][]int{}
	for _, source := range sources {
		visitProductionAuthorityCalls(source, func(function string, call *ast.CallExpr, selector *ast.SelectorExpr) {
			if selector.Sel.Name == "Commit" {
				recordAuthoritySite(commitSites, source, function, call.Pos())
			}
			if source.Package == "internal/overgodb" {
				switch selector.Sel.Name {
				case "commitPublished":
					recordAuthoritySite(commitPublishedSites, source, function, call.Pos())
				case "append":
					recordAuthoritySite(appendSites, source, function, call.Pos())
				case "prepare":
					recordAuthoritySite(prepareSites, source, function, call.Pos())
				}
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

	verifyAuthoritySites(report, "storage", ".Commit", commitSites, directCommitAllowances)
	verifyAuthoritySites(report, "storage", "Store.commitPublished", commitPublishedSites, []authorityAllowance{{
		File: "internal/overgodb/store.go", Function: "Store.Commit", Count: oneAuthoritySite,
	}})
	verifyAuthoritySites(report, "storage", "recordLog.append", appendSites, []authorityAllowance{{
		File: "internal/overgodb/commit_coordinator.go", Function: "commitCoordinator.commit", Count: oneAuthoritySite,
	}})
	verifyAuthoritySites(report, "storage", "blobStore.prepare", prepareSites, []authorityAllowance{{
		File: "internal/overgodb/commit_coordinator.go", Function: "commitCoordinator.commit", Count: oneAuthoritySite,
	}})
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

var directCommitAllowances = []authorityAllowance{
	{File: "cmd/admission/main.go", Count: oneAuthoritySite},
	{File: "cmd/advisories/main.go", Count: threeAuthoritySites},
	{File: "cmd/closure-scan/main.go", Count: twoAuthoritySites},
	{File: "cmd/compatibility/main.go", Count: oneAuthoritySite},
	{File: "cmd/composite-generation-lane/run_windows.go", Count: fourAuthoritySites},
	{File: "cmd/finding/main.go", Count: oneAuthoritySite},
	{File: "cmd/gate/main.go", Count: fourAuthoritySites},
	{File: "cmd/graft-probe/main.go", Count: oneAuthoritySite},
	{File: "cmd/hf-gguf-convert/main.go", Count: oneAuthoritySite},
	{File: "cmd/lora-extract/main.go", Count: oneAuthoritySite},
	{File: "cmd/model-characterize/main.go", Count: twoAuthoritySites},
	{File: "cmd/offline-artifact-validate/main.go", Count: oneAuthoritySite},
	{File: "cmd/plan/orchestration.go", Count: threeAuthoritySites},
	{File: "cmd/recipe/main.go", Count: twoAuthoritySites},
	{File: "cmd/smoke-lane/main.go", Count: oneAuthoritySite},
	{File: "internal/artifact/repositorytest/contract.go", Count: tenAuthoritySites},
	{File: "internal/artifact/repositorytest/counting.go", Count: oneAuthoritySite},
	{File: "internal/artifact/schema.go", Count: oneAuthoritySite},
	{File: "internal/codemanifest/repository.go", Count: twoAuthoritySites},
	{File: "internal/composition/record.go", Count: fourAuthoritySites},
	{File: "internal/overgodb/rebuild.go", Count: oneAuthoritySite},
	{File: "internal/overgodb/retention.go", Count: oneAuthoritySite},
	{File: "internal/runrecord/claim_commit.go", Count: oneAuthoritySite},
	{File: "internal/scratchmodel/derivation_profile.go", Count: oneAuthoritySite},
	{File: "internal/steering/proposal.go", Count: oneAuthoritySite},
	{File: "internal/testutil/fixture.go", Count: oneAuthoritySite},
	{File: "internal/trainingworkflow/bootstrap_recipe.go", Count: twoAuthoritySites},
	{File: "internal/trainingworkflow/session_observation.go", Count: oneAuthoritySite},
	{File: "internal/workflowruntime/runtime.go", Count: oneAuthoritySite},
}

var processPrimitiveAllowances = []authorityAllowance{
	{File: "cmd/advisories/main.go", Count: oneAuthoritySite},
	{File: "cmd/build-kernels/main.go", Count: twoAuthoritySites},
	{File: "cmd/compatibility/training.go", Count: oneAuthoritySite},
	{File: "cmd/composite-generation-lane/run_windows.go", Count: oneAuthoritySite},
	{File: "cmd/eval-lane/main.go", Count: oneAuthoritySite},
	{File: "cmd/evaluate/all.go", Count: oneAuthoritySite},
	{File: "cmd/evaluate/main.go", Count: oneAuthoritySite},
	{File: "cmd/gate/main.go", Count: threeAuthoritySites},
	{File: "cmd/loophook/main.go", Count: sixAuthoritySites},
	{File: "cmd/plan/sync.go", Count: threeAuthoritySites},
	{File: "cmd/recipe/main.go", Count: oneAuthoritySite},
	{File: "cmd/release/main.go", Count: fiveAuthoritySites},
	{File: "cmd/reverify/main.go", Count: twoAuthoritySites},
	{File: "cmd/sbom/main.go", Count: oneAuthoritySite},
	{File: "cmd/smoke-lane/main.go", Count: twoAuthoritySites},
	{File: "cmd/test-lane/main.go", Count: oneAuthoritySite},
	{File: "internal/clioptions/command.go", Count: oneAuthoritySite},
	{File: "internal/processcontrol/supervisor.go", Function: "Start", Count: twoAuthoritySites},
	{File: "internal/repoanalysis/build.go", Count: twoAuthoritySites},
	{File: "internal/repoanalysis/imports.go", Count: oneAuthoritySite},
	{File: "internal/runrecord/verifying_commit.go", Count: threeAuthoritySites},
	{File: "internal/testutil/process.go", Count: twoAuthoritySites},
}
