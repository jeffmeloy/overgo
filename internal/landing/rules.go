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
	"strconv"
	"strings"

	"overgo/internal/gitauthority"
	"overgo/internal/gosyntax"
	"overgo/internal/repoanalysis"
)

// rules compares the changed files with HEAD: no bytes a record pins move,
// no new style finding, no new legacy form go fix leaves, no CUDA-gated test
// no lane runs, and, for a consolidation, no more production code than its
// count. A number written inline and an export its own package alone uses
// are no refusal: those rules refused 24 of the 30 refusals that named their
// rule by 2026-11-11 and guarded little (fewer-refusals/trim-rules). Each
// refusal prints the edit that clears it. It reads the candidate (checkOut),
// and names an ignored document the checkout holds outside its homes
// (strayDocuments) without refusing: git ignores it, so no commit holds it, and
// on 2026-11-08 another session's document stopped every landing. It says what
// the change adds of each kind of written code, which the landed rows' counts take.
func (l *landing) rules(ctx context.Context) error {
	stale, err := stalePins(l.stageRoot(), l.paths)
	if err != nil {
		return err
	}
	if len(stale) != 0 {
		return errors.New(pinsRefusal(stale))
	}
	candidate, base, err := l.changedSource(ctx)
	if err != nil {
		return err
	}
	if err := repoanalysis.ValidateGoStyleDelta(candidate, base); err != nil {
		return err
	}
	findings := newLegacyForms(base, candidate)
	uncovered, err := uncoveredCUDATests(l.stageRoot())
	if err != nil {
		return err
	}
	for _, directory := range uncovered {
		findings = append(findings, uncoveredRefusal(directory))
	}
	strays, err := strayDocuments(ctx, l.Repo)
	if err != nil {
		return err
	}
	if len(strays) != 0 {
		named, _ := rows(l.Ref)
		fmt.Fprintln(l.Log, "land: "+strayNote(strays, named[0].item))
	}
	// A row's growth is measured and recorded, its count taking it; only a consolidation, which must shrink, is held to
	// its count.
	growth := writtenNodes(candidate, false) - writtenNodes(base, false)
	left := l.charge(growth)
	if growth > left && l.consolidates() {
		findings = append(findings, shrinkRefusal(growth, l.Ref, left))
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
	// rootFixtures is the repository's own fixture directory, counted as a package's testdata is.
	rootFixtures = "fixtures/"
)

// outsideGo says what a change adds outside Go source: lines of device kernel
// source and kernel entry points, and fixture files with their bytes, the
// generators kept beside them among them. A file HEAD lacks, or the tree no
// longer holds, counts as empty.
func (l *landing) outsideGo(ctx context.Context) string {
	var lines, entryPoints, fixtures, fixtureBytes int
	for _, entry := range l.entries {
		kernel := strings.HasPrefix(entry.Path, kernelSources) && !strings.Contains(entry.Path, generatedKernel)
		if !kernel && !strings.HasPrefix(entry.Path, rootFixtures) && !strings.Contains("/"+entry.Path, "/testdata/") {
			continue
		}
		before, absent := gitauthority.Query(ctx, l.Repo, "show", l.builtOn()+":"+cmp.Or(entry.OriginalPath, entry.Path))
		after, gone := os.ReadFile(filepath.Join(l.stageRoot(), filepath.FromSlash(entry.Path)))
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
		entries, err := os.ReadDir(filepath.Join(l.stageRoot(), filepath.FromSlash(directory)))
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
	candidate, err := gosyntax.LoadGo(l.stageRoot(), names)
	if err != nil {
		return gosyntax.Snapshot{}, gosyntax.Snapshot{}, err
	}
	base, err := candidate.Overlay(before)
	return candidate, base, err
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
