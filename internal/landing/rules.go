package landing

import (
	"bytes"
	"cmp"
	"context"
	"errors"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"

	"overgo/internal/gitauthority"
	"overgo/internal/gosyntax"
	"overgo/internal/repoanalysis"
)

// rules compares the changed files with HEAD: no bytes a record pins move,
// the pinned media runtime moves only with its reconciliation, no new style
// finding, no new inline policy number, no new legacy form go fix leaves,
// and no new export without a consumer.
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
	if len(findings) != 0 {
		return errors.New(strings.Join(findings, "; "))
	}
	return nil
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
// file gained: sort.Slice where slices.SortFunc serves, sync.Once where
// sync.OnceFunc or sync.OnceValue serves, and an if that falls back from an
// empty string or zero where cmp.Or serves.
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
	switch typed := node.(type) {
	case *ast.SelectorExpr:
		qualifier, ok := typed.X.(*ast.Ident)
		if !ok {
			return ""
		}
		switch imports[qualifier.Name] + "." + typed.Sel.Name {
		case "sort.Slice", "sort.SliceStable":
			return "sort." + typed.Sel.Name + " (use slices.SortFunc or slices.SortStableFunc)"
		case "sync.Once":
			return "sync.Once (use sync.OnceFunc, sync.OnceValue or sync.OnceValues)"
		}
	case *ast.IfStmt:
		if zeroFallback(typed) {
			return "zero-value fallback if (use cmp.Or)"
		}
	}
	return ""
}

// zeroFallback reports an if with no init or else whose one statement
// assigns the value it compared with "" or 0.
func zeroFallback(statement *ast.IfStmt) bool {
	if statement.Init != nil || statement.Else != nil || len(statement.Body.List) != 1 {
		return false
	}
	comparison, ok := statement.Cond.(*ast.BinaryExpr)
	assignment, assigns := statement.Body.List[0].(*ast.AssignStmt)
	if !ok || comparison.Op != token.EQL || !assigns || assignment.Tok != token.ASSIGN || len(assignment.Lhs) != 1 {
		return false
	}
	zero := func(expression ast.Expr) bool {
		literal, ok := expression.(*ast.BasicLit)
		return ok && (literal.Value == `""` || literal.Value == "``" || literal.Value == "0")
	}
	value := comparison.X
	if zero(value) {
		value = comparison.Y
	} else if !zero(comparison.Y) {
		return false
	}
	return types.ExprString(assignment.Lhs[0]) == types.ExprString(value)
}

// newUnconsumedExports names exported top-level functions, variables and
// constants a changed internal package gained that no other package's
// production code names. An export only its own package uses is surface
// without a consumer.
func (l *landing) newUnconsumedExports(base, candidate gosyntax.Snapshot) ([]string, error) {
	existing := exportedNames(base)
	var findings []string
	var corpus map[string][]byte
	for key, pkg := range exportedNames(candidate) {
		if _, found := existing[key]; found {
			continue
		}
		if corpus == nil {
			var err error
			if corpus, err = l.productionSources(); err != nil {
				return nil, err
			}
		}
		directory, name, _ := strings.Cut(key, " ")
		reference := []byte(pkg + "." + name)
		consumed := false
		for file, data := range corpus {
			if path.Dir(file) != directory && bytes.Contains(data, reference) {
				consumed = true
				break
			}
		}
		if !consumed {
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

// productionSources reads every non-test Go file under cmd and internal.
func (l *landing) productionSources() (map[string][]byte, error) {
	sources := map[string][]byte{}
	for _, top := range []string{"cmd", "internal"} {
		err := filepath.WalkDir(filepath.Join(l.Repo, top), func(name string, entry fs.DirEntry, err error) error {
			if err != nil || entry.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
				return err
			}
			data, err := os.ReadFile(name)
			if err != nil {
				return err
			}
			relative, err := filepath.Rel(l.Repo, name)
			if err == nil {
				sources[filepath.ToSlash(relative)] = data
			}
			return err
		})
		if err != nil {
			return nil, err
		}
	}
	return sources, nil
}
