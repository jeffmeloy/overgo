package gate

import (
	"maps"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/testscope"
)

type packageTestScope struct {
	direct, dependent, productionPaths []string
	unresolved                         []string
	opaqueRuntimeInputs                []string
	// opaqueReaders: packages bound to every root with the reason the
	// source left their runtime reach unnamed.
	opaqueReaders []string
	edited        []string
	excluded      int
}

func (g *gateContext) deriveTestScope() (packageTestScope, error) {
	graph, err := g.inputGraph()
	if err != nil {
		return packageTestScope{}, err
	}
	// Deleted inputs are absent from the candidate index. Retain their owner
	// conservatively so removing a resource cannot remove its acceptance too.
	graph.bindResourceFiles(g.paths)
	var roots []goPackageInput
	var packages []testscope.Package
	for _, node := range graph.nodes {
		// Match identifies real packages selected by ./...; test binaries have
		// no match, and ForTest identifies compiler-generated test variants.
		if len(node.Match) == 0 || node.ForTest != "" {
			continue
		}
		roots = append(roots, node)
		pkg := testscope.Package{
			ImportPath: node.ImportPath, Dir: node.Dir, ProductionFiles: node.productionFiles(),
			TestEmbedFiles: node.TestEmbedFiles, XTestEmbedFiles: node.XTestEmbedFiles,
		}
		for _, path := range graph.resourceFiles[node.Dir] {
			relative, err := filepath.Rel(node.Dir, path)
			if err != nil {
				return packageTestScope{}, err
			}
			pkg.ResourceFiles = append(pkg.ResourceFiles, relative)
		}
		// Paths the tests name select the package's own tests, like an
		// embedded test fixture; paths the compiled sources name at run
		// time select it too and taint importers that reach the repository.
		for _, path := range slices.Concat(graph.testResourceFiles[node.Dir], graph.runtimeResourceFiles[node.Dir]) {
			relative, err := filepath.Rel(node.Dir, path)
			if err != nil {
				return packageTestScope{}, err
			}
			pkg.TestEmbedFiles = append(pkg.TestEmbedFiles, relative)
		}
		packages = append(packages, pkg)
	}
	direct, production := testscope.DirectPackages(graph.root, g.paths, packages)
	scope := packageTestScope{direct: direct}
	// Keep physical source edits ahead of conservatively selected readers.
	// This changes order only; the full affected set remains required.
	scope.edited, _ = testscope.DirectPackages(graph.root, g.changedGoFiles(), packages)
	for _, path := range g.paths {
		if path == "go.mod" || path == "go.sum" {
			scope.unresolved = append(scope.unresolved, path)
			continue
		}
		if strings.EqualFold(filepath.Ext(path), ".go") {
			owners, _ := testscope.DirectPackages(graph.root, []string{path}, packages)
			if len(owners) == 0 {
				scope.unresolved = append(scope.unresolved, path)
			}
		}
	}
	productionDirs := map[string]bool{}
	for _, node := range roots {
		if len(scope.unresolved) == 0 && !slices.Contains(production, node.ImportPath) {
			continue
		}
		productionDirs[node.Dir] = true
		// The boundary owner consumes package directories through source paths.
		// Include actual compiled sources so nested embedded assets reach the
		// same assembly checks as their owning production package.
		for _, name := range append(slices.Clone(node.GoFiles), node.CgoFiles...) {
			relative, err := filepath.Rel(graph.root, filepath.Join(node.Dir, name))
			if err != nil {
				return packageTestScope{}, err
			}
			scope.productionPaths = append(scope.productionPaths, filepath.ToSlash(relative))
		}
	}
	// A dependency recompiled for another package's tests has that other
	// package in ForTest. Seed by physical owner, then follow the compiler's
	// actual imports; do not turn test imports into production dependencies.
	// A production change propagates to every importer. An opaque reader's
	// unnamed reach propagates only into importers whose own sources reach
	// the repository root: a test over a fixture under a temporary directory
	// cannot observe a repository file through the library it imports.
	affected, tainted, named := map[string]bool{}, map[string]bool{}, map[string]bool{}
	changedAbsolute := map[string]bool{}
	for _, name := range g.paths {
		changedAbsolute[filepath.Clean(filepath.Join(graph.root, filepath.FromSlash(name)))] = true
	}
	for _, node := range graph.nodes {
		if productionDirs[node.Dir] {
			affected[node.ImportPath] = true
		}
		// A changed path the compiled sources read at run time is a change
		// to the reader's own inputs; importers observe it only when their
		// tests reach the repository.
		if slices.ContainsFunc(graph.runtimeResourceFiles[node.Dir], func(path string) bool { return changedAbsolute[filepath.Clean(path)] }) {
			named[node.ImportPath], tainted[node.ImportPath] = true, true
		}
		// Opaque readers may inspect test source as data. A test-only edit
		// therefore reaches them even without a production import change;
		// a reader that names its inputs follows its named edges instead.
		if node.opaqueReader && len(g.paths) != 0 {
			tainted[node.ImportPath] = true
		}
	}
	for changed := true; changed; {
		changed = false
		for _, node := range graph.nodes {
			// A test-only unnamed reach binds every root for the package's
			// own identity and selection; it is not an edge its importers
			// observe, and the package is selected directly below.
			edges := node.Imports
			if !node.testOpaque && !node.opaqueReader {
				edges = slices.Concat(node.Imports, node.inputDependencies)
			}
			if !affected[node.ImportPath] && slices.ContainsFunc(edges, func(imported string) bool { return affected[imported] }) {
				affected[node.ImportPath] = true
				changed = true
			}
			if !tainted[node.ImportPath] && (node.escapes || node.opaqueReader || node.testOpaque) &&
				slices.ContainsFunc(edges, func(imported string) bool { return tainted[imported] }) {
				tainted[node.ImportPath] = true
				changed = true
			}
		}
	}
	changeAffected := maps.Clone(affected)
	maps.Copy(affected, tainted)
	testTargets := map[string]bool{}
	for _, node := range graph.nodes {
		if affected[node.ImportPath] && node.ForTest != "" {
			testTargets[node.ForTest] = true
		}
	}
	for _, node := range roots {
		if slices.Contains(direct, node.ImportPath) {
			continue
		}
		// A production change or a change to a file the package names at
		// run time runs the package in the complete group. A package whose
		// unnamed reach, or whose import's unnamed reach, might observe the
		// change runs in the short group: nothing it compiles changed.
		if changeAffected[node.ImportPath] || testTargets[node.ImportPath] || named[node.ImportPath] {
			scope.dependent = append(scope.dependent, node.ImportPath)
			continue
		}
		if tainted[node.ImportPath] || (node.testOpaque || node.opaqueReader) && len(g.paths) != 0 {
			scope.direct = append(scope.direct, node.ImportPath)
			continue
		}
		scope.excluded++
	}
	slices.Sort(scope.direct)
	slices.Sort(scope.dependent)
	for _, node := range graph.nodes {
		if len(node.inputDependencies) == 0 {
			continue
		}
		target := node.ImportPath
		if node.ForTest != "" {
			target = node.ForTest
		}
		if (slices.Contains(scope.direct, target) || slices.Contains(scope.dependent, target)) && !slices.Contains(scope.opaqueRuntimeInputs, target) {
			scope.opaqueRuntimeInputs = append(scope.opaqueRuntimeInputs, target)
		}
		if (node.opaqueReader || node.testOpaque) && !slices.Contains(scope.opaqueReaders, target+": "+node.runtimeReason) {
			scope.opaqueReaders = append(scope.opaqueReaders, target+": "+node.runtimeReason)
		}
	}
	slices.Sort(scope.opaqueReaders)
	slices.Sort(scope.opaqueRuntimeInputs)
	return scope, nil
}
