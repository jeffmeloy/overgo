package landing

import (
	"bytes"
	"cmp"
	"context"
	"errors"
	"fmt"
	"go/ast"
	"go/token"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"

	"overgo/internal/gitauthority"
	"overgo/internal/gosyntax"
	"overgo/internal/repoanalysis"
)

// rules compares the changed files with HEAD: no bytes a record pins move,
// no new style finding, no new inline policy number, no new legacy form go fix
// leaves, no new export without a consumer, no CUDA-gated test no lane runs,
// and no more production code than the landed rows may still add. It says
// what the change adds of each kind of written code; only production code is
// bounded.
func (l *landing) rules(ctx context.Context) error {
	stale, err := stalePins(l.Repo, l.paths)
	if err != nil {
		return err
	}
	if len(stale) != 0 {
		return fmt.Errorf("changes bytes a record pins: %s; repin the record for a behaviour-neutral edit, or restore the file", strings.Join(stale, "; "))
	}
	candidate, base, err := l.changedSource(ctx)
	if err != nil {
		return err
	}
	if err := repoanalysis.ValidateGoStyleDelta(candidate, base); err != nil {
		return err
	}
	findings := append(newPolicyLiterals(base, candidate), newLegacyForms(base, candidate)...)
	exports, err := l.newUnconsumedExports(base, candidate)
	if err != nil {
		return err
	}
	findings = append(findings, exports...)
	uncovered, err := uncoveredCUDATests(l.Repo)
	if err != nil {
		return err
	}
	for _, directory := range uncovered {
		findings = append(findings, directory+" has CUDA-gated tests that no lane runs; add it to a lane that sets "+cudaTestEnvironment)
	}
	growth := writtenNodes(candidate, false) - writtenNodes(base, false)
	left := l.charge(growth)
	if growth > left {
		findings = append(findings, fmt.Sprintf("adds %d production syntax nodes, and the row(s) %s may add %d; pay for the rest with deletions, or raise the count where this change's terms are read, in a change to the plan alone before it, or in the row of a merge", growth, l.Ref, left))
	}
	if len(findings) != 0 {
		return errors.New(strings.Join(findings, "; "))
	}
	l.details["rules"] = fmt.Sprintf("production %+d nodes of the %d its row may add, test code %+d nodes, record schemas %+d, %s",
		growth, left, writtenNodes(candidate, true)-writtenNodes(base, true), recordSchemas(candidate)-recordSchemas(base), l.outsideGo(ctx))
	return nil
}

// writtenNodes counts the syntax nodes of a snapshot's written Go: its
// production files, which a row's count bounds, or its test files, which a
// landing reports and nothing bounds, since a change still ships the test
// that would have caught its failure. Testdata and generated files are
// neither.
func writtenNodes(snapshot gosyntax.Snapshot, tests bool) int {
	nodes := 0
	for _, file := range snapshot.Files {
		if file.Test != tests || strings.Contains("/"+file.Path, "/testdata/") {
			continue
		}
		syntax, err := file.Syntax()
		if generated, _ := file.Generated(); err != nil || generated {
			continue
		}
		ast.Inspect(syntax, func(node ast.Node) bool {
			if node != nil {
				nodes++
			}
			return true
		})
	}
	return nodes
}

// recordSchemas counts the record schemas a snapshot's production files name:
// the strings that open with overgo and close with a version after a dot or a
// slash, as overgo.recipe-lifecycle.v1 and overgo/decision/v1 do.
func recordSchemas(snapshot gosyntax.Snapshot) (named int) {
	for _, file := range snapshot.Files {
		syntax, err := file.Syntax()
		if file.Test || err != nil {
			continue
		}
		ast.Inspect(syntax, func(node ast.Node) bool {
			if literal, ok := node.(*ast.BasicLit); ok && literal.Kind == token.STRING {
				name := strings.Trim(literal.Value, "\"`")
				version, found := strings.CutPrefix(name[strings.LastIndexAny(name, "./")+1:], "v")
				if _, err := strconv.Atoi(version); found && err == nil && strings.HasPrefix(name, "overgo") {
					named++
				}
			}
			return true
		})
	}
	return named
}

// Where the device kernels are written, the word that opens each of their
// entry points, and the mark of a kernel file that is generated output and not
// written code.
const (
	kernelSources   = "kernels/cuda/"
	kernelEntry     = "__global__"
	generatedKernel = "_generated"
)

// outsideGo says what a change adds outside Go source: lines of device kernel
// source and kernel entry points, and fixture files with their bytes, the
// generators kept beside them among them. A file HEAD lacks, or the tree no
// longer holds, counts as empty.
func (l *landing) outsideGo(ctx context.Context) string {
	var lines, entryPoints, fixtures, fixtureBytes int
	for _, entry := range l.entries {
		kernel := strings.HasPrefix(entry.Path, kernelSources) && !strings.Contains(entry.Path, generatedKernel)
		if !kernel && !strings.Contains("/"+entry.Path, "/testdata/") {
			continue
		}
		before, absent := gitauthority.Query(ctx, l.Repo, "show", l.builtOn()+":"+cmp.Or(entry.OriginalPath, entry.Path))
		after, gone := os.ReadFile(filepath.Join(l.Repo, filepath.FromSlash(entry.Path)))
		if kernel {
			lines += bytes.Count(after, []byte("\n")) - bytes.Count(before, []byte("\n"))
			entryPoints += bytes.Count(after, []byte(kernelEntry)) - bytes.Count(before, []byte(kernelEntry))
			continue
		}
		if absent == nil {
			fixtures--
		}
		if gone == nil {
			fixtures++
		}
		fixtureBytes += len(after) - len(before)
	}
	return fmt.Sprintf("CUDA source %+d lines and %+d entry points, fixtures %+d files and %+d bytes", lines, entryPoints, fixtures, fixtureBytes)
}

// changedSource is every Go file of the changed packages as the tree holds
// them, and the same packages with each changed file as HEAD held it: a
// package-level rule sees the whole package, not the files a change touched.
func (l *landing) changedSource(ctx context.Context) (gosyntax.Snapshot, gosyntax.Snapshot, error) {
	directories := map[string]bool{}
	before := map[string][]byte{}
	for _, entry := range l.entries {
		if !strings.HasSuffix(entry.Path, ".go") {
			continue
		}
		directories[path.Dir(entry.Path)] = true
		// A renamed file's base is what HEAD held at its old path; a file
		// HEAD lacks is absent from the base.
		data, err := gitauthority.Query(ctx, l.Repo, "show", l.builtOn()+":"+cmp.Or(entry.OriginalPath, entry.Path))
		if err != nil {
			data = nil
		}
		before[entry.Path] = data
	}
	var names []string
	for directory := range directories {
		entries, err := os.ReadDir(filepath.Join(l.Repo, filepath.FromSlash(directory)))
		if errors.Is(err, fs.ErrNotExist) {
			continue
		}
		if err != nil {
			return gosyntax.Snapshot{}, gosyntax.Snapshot{}, err
		}
		for _, entry := range entries {
			if !entry.IsDir() && strings.HasSuffix(entry.Name(), ".go") {
				names = append(names, path.Join(directory, entry.Name()))
			}
		}
	}
	candidate, err := gosyntax.LoadGo(l.Repo, names)
	if err != nil {
		return gosyntax.Snapshot{}, gosyntax.Snapshot{}, err
	}
	base, err := candidate.Overlay(before)
	return candidate, base, err
}

// newPolicyLiterals names the inline numbers a changed production file
// gained outside constant declarations. Zero and one are identities; any
// other number a change writes inline is a decision that belongs in a named
// constant whose comment says why it has that value.
func newPolicyLiterals(base, candidate gosyntax.Snapshot) []string {
	before := literalCounts(base)
	var findings []string
	for key, count := range literalCounts(candidate) {
		if count > before[key] {
			findings = append(findings, "new inline number "+key+"; name it as a documented constant")
		}
	}
	return findings
}

// literalCounts counts each number written inline by file and value. A
// count keyed by file, not by enclosing declaration, still rises when a
// change adds a number, and holds when a change renames or moves the
// function that already had it.
func literalCounts(snapshot gosyntax.Snapshot) map[string]int {
	counts := map[string]int{}
	for _, file := range snapshot.Files {
		if file.Test {
			continue
		}
		syntax, err := file.Syntax()
		if generated, _ := file.Generated(); err != nil || generated {
			continue
		}
		for _, declaration := range syntax.Decls {
			if general, ok := declaration.(*ast.GenDecl); ok && general.Tok == token.CONST {
				continue
			}
			ast.Inspect(declaration, func(node ast.Node) bool {
				literal, ok := node.(*ast.BasicLit)
				if ok && (literal.Kind == token.INT || literal.Kind == token.FLOAT) && literal.Value != "0" && literal.Value != "1" {
					counts[file.Path+" "+literal.Value]++
				}
				return true
			})
		}
	}
	return counts
}

// newLegacyForms names the forms go fix does not rewrite that a changed
// file gained: sort.Slice where slices.SortFunc serves, and sync.Once where
// sync.OnceFunc or sync.OnceValue serves. An if that falls back from an
// empty value is not one: cmp.Or evaluates its fallback even when it is not
// needed, and a fallback that indexes or dereferences can panic there.
func newLegacyForms(base, candidate gosyntax.Snapshot) []string {
	before := legacyFormCounts(base)
	var findings []string
	for key, count := range legacyFormCounts(candidate) {
		if count > before[key] {
			findings = append(findings, key)
		}
	}
	return findings
}

// legacyFormCounts counts each legacy form by file and enclosing
// declaration.
func legacyFormCounts(snapshot gosyntax.Snapshot) map[string]int {
	counts := map[string]int{}
	for _, file := range snapshot.Files {
		syntax, err := file.Syntax()
		if generated, _ := file.Generated(); err != nil || generated {
			continue
		}
		imports := map[string]string{}
		for _, spec := range syntax.Imports {
			importPath, _ := strconv.Unquote(spec.Path.Value)
			name := path.Base(importPath)
			if spec.Name != nil {
				name = spec.Name.Name
			}
			imports[name] = importPath
		}
		for _, declaration := range syntax.Decls {
			scope := "package"
			if function, ok := declaration.(*ast.FuncDecl); ok {
				scope = function.Name.Name
			}
			ast.Inspect(declaration, func(node ast.Node) bool {
				if form := legacyForm(node, imports); form != "" {
					counts["new "+form+" in "+file.Path+" "+scope]++
				}
				return true
			})
		}
	}
	return counts
}

// legacyForm names the legacy form node is, if any.
func legacyForm(node ast.Node, imports map[string]string) string {
	selector, ok := node.(*ast.SelectorExpr)
	if !ok {
		return ""
	}
	qualifier, ok := selector.X.(*ast.Ident)
	if !ok {
		return ""
	}
	switch imports[qualifier.Name] + "." + selector.Sel.Name {
	case "sort.Slice", "sort.SliceStable":
		return "sort." + selector.Sel.Name + " (use slices.SortFunc or slices.SortStableFunc)"
	case "sync.Once":
		return "sync.Once (use sync.OnceFunc, sync.OnceValue or sync.OnceValues)"
	}
	return ""
}

// modulePath is the import path the repository's go.mod declares; a
// package's import path is it joined with the package's directory.
const modulePath = "overgo"

// newUnconsumedExports names exported top-level functions, variables and
// constants a changed internal package gained that no other package's
// production code selects through an import of that package. An export only
// its own package uses is surface without a consumer. A test-support
// package exists for other packages' tests, so their selections count.
func (l *landing) newUnconsumedExports(base, candidate gosyntax.Snapshot) ([]string, error) {
	existing := exportedNames(base)
	corpus := sync.OnceValues(func() (gosyntax.Snapshot, error) { return gosyntax.DiscoverGo(l.Repo, "cmd", "internal") })
	// support holds, per directory, whether its package is test support:
	// every production file outside it that imports it is in a test-support
	// package, and none is a main package. Go's imports are acyclic; a
	// directory met again while its answer is pending reads as production.
	support := map[string]bool{}
	var testSupport func(sources gosyntax.Snapshot, directory string) bool
	testSupport = func(sources gosyntax.Snapshot, directory string) bool {
		if known, found := support[directory]; found {
			return known
		}
		support[directory] = false
		support[directory] = !slices.ContainsFunc(sources.Files, func(file gosyntax.File) bool {
			if file.Test || path.Dir(file.Path) == directory {
				return false
			}
			names, syntax, err := importNames(file, modulePath+"/"+directory, "")
			return err != nil || len(names) != 0 && (syntax.Name.Name == "main" || !testSupport(sources, path.Dir(file.Path)))
		})
		return support[directory]
	}
	var findings []string
	for key, pkg := range exportedNames(candidate) {
		if _, found := existing[key]; found {
			continue
		}
		sources, err := corpus()
		if err != nil {
			return nil, err
		}
		directory, name, _ := strings.Cut(key, " ")
		production, tested := false, false
		for _, file := range sources.Files {
			if path.Dir(file.Path) == directory {
				continue
			}
			names, syntax, err := importNames(file, modulePath+"/"+directory, pkg)
			selected := err != nil || selects(syntax, names, name)
			if file.Test {
				tested = tested || selected
			} else {
				production = production || selected
			}
		}
		if !production && !(tested && testSupport(sources, directory)) {
			findings = append(findings, "new export "+pkg+"."+name+" in "+directory+" has no consumer outside its package; unexport it or land its consumer")
		}
	}
	return findings, nil
}

// exportedNames maps each exported top-level function, variable and constant
// of an internal production file, keyed by directory and name, to its
// package name.
func exportedNames(snapshot gosyntax.Snapshot) map[string]string {
	names := map[string]string{}
	for _, file := range snapshot.Files {
		if file.Test || !strings.HasPrefix(file.Path, "internal/") {
			continue
		}
		syntax, err := file.Syntax()
		if err != nil {
			continue
		}
		record := func(identifier *ast.Ident) {
			if identifier.IsExported() {
				names[path.Dir(file.Path)+" "+identifier.Name] = syntax.Name.Name
			}
		}
		for _, declaration := range syntax.Decls {
			switch typed := declaration.(type) {
			case *ast.FuncDecl:
				if typed.Recv == nil {
					record(typed.Name)
				}
			case *ast.GenDecl:
				// A type reaches its consumers through the functions and values
				// that carry it, often without its name being written, so only
				// those are held to a consumer.
				for _, spec := range typed.Specs {
					if values, ok := spec.(*ast.ValueSpec); ok {
						for _, identifier := range values.Names {
							record(identifier)
						}
					}
				}
			}
		}
	}
	return names
}

// importNames are the names a file imports the package at importPath, named
// pkg, as, and the file's syntax. A comment or a string that mentions the
// path, or a same-named package from another path, is no import; a file
// that names the path but does not parse returns its error, and the rule,
// which cannot read it, takes it to import the package and select from it.
func importNames(file gosyntax.File, importPath, pkg string) ([]string, *ast.File, error) {
	if !bytes.Contains(file.Data, []byte(importPath)) {
		return nil, nil, nil
	}
	syntax, err := file.Syntax()
	if err != nil {
		return nil, nil, err
	}
	var names []string
	for _, spec := range syntax.Imports {
		if imported, _ := strconv.Unquote(spec.Path.Value); imported == importPath {
			local := pkg
			if spec.Name != nil {
				local = spec.Name.Name
			}
			names = append(names, local)
		}
	}
	return names, syntax, nil
}

// selects reports whether syntax selects name through one of the names it
// imports a package as, where no declaration of the file shadows that name.
func selects(syntax *ast.File, names []string, name string) bool {
	if len(names) == 0 {
		return false
	}
	found := false
	ast.Inspect(syntax, func(node ast.Node) bool {
		if selector, ok := node.(*ast.SelectorExpr); ok && selector.Sel.Name == name {
			qualifier, ok := selector.X.(*ast.Ident)
			found = found || ok && slices.Contains(names, qualifier.Name) && qualifier.Obj == nil
		}
		return !found
	})
	return found
}
