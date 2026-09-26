package repoanalysis

import "go/ast"

func auditStorageAndProcessAuthorities(sources []productionAuthoritySource, report *ProductionAuthorityReport) {
	requireAuthorityOwners(sources, report,
		authorityOwnerRule{Family: "storage", Kind: "method", Symbol: "Commit", Owner: "internal/overgodb", Receiver: "Store"},
		authorityOwnerRule{Family: "storage", Kind: "method", Symbol: "CommitAs", Owner: "internal/overgodb", Receiver: "Store"},
		authorityOwnerRule{Family: "storage", Kind: "func", Symbol: "NewProducer", Owner: "internal/overgodb"},
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
	producerCommitSites := map[authoritySiteKey][]int{}
	producerMintSites := map[authoritySiteKey][]int{}
	appendSites := map[authoritySiteKey][]int{}
	prepareSites := map[authoritySiteKey][]int{}
	processSites := map[authoritySiteKey][]int{}
	for _, source := range sources {
		visitProductionAuthorityCalls(source, func(function string, call *ast.CallExpr, selector *ast.SelectorExpr) {
			switch selector.Sel.Name {
			case "Commit":
				recordAuthoritySite(commitSites, source, function, call.Pos())
			case "CommitAs":
				recordAuthoritySite(producerCommitSites, source, function, call.Pos())
			case "NewProducer":
				recordAuthoritySite(producerMintSites, source, function, call.Pos())
			}
			if source.Package == "internal/overgodb" {
				switch selector.Sel.Name {
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
	// A producer's door and its mint are the capability to commit a guarded
	// kind: each site is a reviewed entry here, never a caller's choice.
	verifyAuthoritySites(report, "storage", ".CommitAs", producerCommitSites, []authorityAllowance{
		{File: "cmd/store-precheck/main.go", Count: oneAuthoritySite},
		{File: "internal/gate/batch_evidence.go", Count: oneAuthoritySite},
		{File: "internal/gate/deferred_lanes.go", Count: twoAuthoritySites},
		{File: "internal/gate/finalization.go", Count: oneAuthoritySite},
		{File: "internal/gate/preparation.go", Count: oneAuthoritySite},
		{File: "internal/gate/recovery.go", Count: threeAuthoritySites},
		{File: "internal/gate/terminal_evidence.go", Count: twoAuthoritySites},
		{File: "internal/overgodb/rebuild.go", Count: oneAuthoritySite},
		{File: "internal/overgodb/store.go", Function: "Store.Commit", Count: oneAuthoritySite},
	})
	verifyAuthoritySites(report, "storage", "overgodb.NewProducer", producerMintSites, []authorityAllowance{
		{File: "cmd/store-precheck/main.go", Count: oneAuthoritySite},
		{File: "internal/gate/finalization.go", Count: oneAuthoritySite},
	})
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
	{File: "cmd/graft-probe/main.go", Count: oneAuthoritySite},
	{File: "cmd/lora-extract/main.go", Count: oneAuthoritySite},
	{File: "cmd/model-characterize/main.go", Count: twoAuthoritySites},
	{File: "cmd/offline-artifact-validate/main.go", Count: oneAuthoritySite},
	{File: "cmd/plan/orchestration.go", Count: threeAuthoritySites},
	{File: "internal/artifact/repositorytest/contract.go", Count: tenAuthoritySites},
	{File: "internal/artifact/repositorytest/counting.go", Count: oneAuthoritySite},
	{File: "internal/artifact/schema.go", Count: oneAuthoritySite},
	{File: "internal/codemanifest/repository.go", Count: twoAuthoritySites},
	{File: "internal/composition/compositiontest/authority.go", Count: twoAuthoritySites},
	{File: "internal/composition/record.go", Count: fourAuthoritySites},
	{File: "internal/overgodb/retention.go", Count: oneAuthoritySite},
	{File: "internal/runrecord/claim_commit.go", Count: oneAuthoritySite},
	{File: "internal/scratchmodel/derivation_profile.go", Count: oneAuthoritySite},
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
	{File: "internal/gate/gate.go", Count: threeAuthoritySites},
	{File: "cmd/loophook/main.go", Count: sixAuthoritySites},
	{File: "cmd/plan/sync.go", Count: threeAuthoritySites},
	{File: "cmd/release/main.go", Count: fiveAuthoritySites},
	{File: "cmd/reverify/main.go", Count: twoAuthoritySites},
	{File: "cmd/sbom/main.go", Count: oneAuthoritySite},
	{File: "internal/clioptions/command.go", Count: oneAuthoritySite},
	{File: "internal/processcontrol/supervisor.go", Function: "Start", Count: twoAuthoritySites},
	{File: "internal/processcontrol/detached.go", Function: "StartDetached", Count: twoAuthoritySites},
	{File: "internal/repoanalysis/imports.go", Count: oneAuthoritySite},
	{File: "internal/runrecord/verifying_commit.go", Count: threeAuthoritySites},
	{File: "internal/testprocess/process.go", Count: twoAuthoritySites},
}
