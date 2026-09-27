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
	"strings"

	"overgo/internal/gitauthority"
	"overgo/internal/gosyntax"
	"overgo/internal/repoanalysis"
)

// rules compares the changed files with HEAD: no bytes a record pins move,
// the pinned media runtime moves only with its reconciliation, no new style
// finding, no new inline policy number, and no new export without a consumer.
func (l *landing) rules(ctx context.Context) error {
	stale, err := StalePins(l.Repo, l.paths)
	if err != nil {
		return err
	}
	if len(stale) != 0 {
		return fmt.Errorf("changes bytes a record pins: %s; repin the record for a behaviour-neutral edit, or restore the file", strings.Join(stale, "; "))
	}
	if _, err := MediaMoves(l.Repo, l.paths); err != nil {
		return err
	}
	candidate, base, err := l.changedSource(ctx)
	if err != nil {
		return err
	}
	if err := repoanalysis.ValidateGoStyleDelta(candidate, base); err != nil {
		return err
	}
	findings := newPolicyLiterals(base, candidate)
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

// literalCounts counts each number written inline by file, enclosing
// declaration and value.
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
			scope := "package"
			switch typed := declaration.(type) {
			case *ast.GenDecl:
				if typed.Tok == token.CONST {
					continue
				}
			case *ast.FuncDecl:
				scope = typed.Name.Name
			}
			ast.Inspect(declaration, func(node ast.Node) bool {
				literal, ok := node.(*ast.BasicLit)
				if ok && (literal.Kind == token.INT || literal.Kind == token.FLOAT) && literal.Value != "0" && literal.Value != "1" {
					counts[file.Path+" "+scope+" "+literal.Value]++
				}
				return true
			})
		}
	}
	return counts
}

// newUnconsumedExports names exported top-level identifiers a changed
// internal package gained that no other package's production code names.
// An export only its own package uses is surface without a consumer.
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

// exportedNames maps each exported top-level identifier of an internal
// production file, keyed by directory and name, to its package name.
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
				for _, spec := range typed.Specs {
					switch spec := spec.(type) {
					case *ast.TypeSpec:
						record(spec.Name)
					case *ast.ValueSpec:
						for _, identifier := range spec.Names {
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
