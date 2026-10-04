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
	"overgo/internal/plan"
	"overgo/internal/repoanalysis"
)

// rules compares the changed files with HEAD: no bytes a record pins move,
// the pinned media runtime moves only with its reconciliation, no new style
// finding, no new inline policy number, no new legacy form go fix leaves,
// no new export without a consumer, and no CUDA-gated test no lane runs.
func (l *landing) rules(ctx context.Context) error {
	stale, err := stalePins(l.Repo, l.paths)
	if err != nil {
		return err
	}
	if len(stale) != 0 {
		return fmt.Errorf("changes bytes a record pins: %s; repin the record for a behaviour-neutral edit, or restore the file", strings.Join(stale, "; "))
	}
	if _, err := mediaMoves(l.Repo, l.paths); err != nil {
		return err
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
	if growth := productionNodes(candidate) - productionNodes(base); !admits(l.budget, growth) {
		findings = append(findings, fmt.Sprintf("adds %d production syntax nodes beyond row %s's budget %+v; pay for them with deletions, or declare the growth in the row's budget", growth, l.item, l.budget))
	}
	if len(findings) != 0 {
		return errors.New(strings.Join(findings, "; "))
	}
	return nil
}

// productionNodes counts the syntax nodes of a snapshot's production Go
// files; tests, testdata and generated files are not the code a row grows.
func productionNodes(snapshot gosyntax.Snapshot) int {
	nodes := 0
	for _, file := range snapshot.Files {
		if file.Test || strings.Contains("/"+file.Path, "/testdata/") {
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

// admits reports whether a row's budget admits a change that grows
// production code by growth nodes: a row without a budget may not grow, a
// stated reason admits the growth, and declared nodes cap it (negative for a
// row that must shrink).
func admits(budget plan.Budget, growth int) bool {
	switch {
	case budget.Nodes != 0:
		return growth <= budget.Nodes
	case budget.Reason != "":
		return true
	default:
		return growth <= 0
	}
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
		data, err := gitauthority.Query(ctx, l.Repo, "show", "HEAD:"+cmp.Or(entry.OriginalPath, entry.Path))
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
