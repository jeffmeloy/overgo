package repoanalysis

import (
	"fmt"
	"go/ast"
	"go/types"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// repeatLeast is the fewest syntax nodes a body holds for its repeats to count: a body smaller than that is an idiom,
// which a helper would not shorten.
const repeatLeast = 20

// repeatCeiling is the most syntax nodes the production bodies outside the inference surface may repeat beyond one
// copy of each: what compact-production/exact-folds-outside left, less the biased linear surface-folds gave a home in
// hostmath, repeats of types a helper would not unite.
const repeatCeiling = 296

// parameterLeast is the most values two bodies of one skeleton may differ in and still be one repeat with those values
// as its parameters.
const parameterLeast = 3

// parameterCeiling is the most syntax nodes the production bodies outside the inference surface may repeat beyond one
// copy of each, among bodies that differ in a value or more and at most parameterLeast: what
// compact-production/param-folds-outside left, repeats over different types or report fields, which a helper would
// not shorten.
const parameterCeiling = 1945

// surfaceRepeatCeiling and surfaceParameterCeiling are the most syntax nodes the production bodies inside the inference
// surface may repeat beyond one copy of each, alike and differing in at most parameterLeast values: what
// compact-production/surface-folds left, repeats on a per-token path, which no fold takes without a benchmark, the
// device cache's releases, which consolidate/release-storages folds once their callers' lifetimes are read, the CUDA
// executor's and driver's, which consolidate/cell-step-reads folds once their full device lane can pass, and typed
// accessors and constructors over different types, which a helper would not shorten.
const (
	surfaceRepeatCeiling    = 204
	surfaceParameterCeiling = 1493
)

// bodyShape reads a body as the checker resolves it: its skeleton, each node's kind and each operator, with each local
// variable named by the order it first appears and each literal and each other name a hole; and the values that fill
// the holes, in order. Two bodies that differ only in their locals' names have one skeleton and one list of values.
func bodyShape(body ast.Node, info *types.Info) (skeleton string, values []string, nodes int) {
	var out strings.Builder
	locals := map[types.Object]int{}
	ast.Inspect(body, func(node ast.Node) bool {
		if node == nil {
			out.WriteString(")")
			return true
		}
		nodes++
		out.WriteString("(" + reflect.TypeOf(node).Elem().Name())
		switch typed := node.(type) {
		case *ast.Ident:
			object := info.Defs[typed]
			if object == nil {
				object = info.Uses[typed]
			}
			if local, isVar := object.(*types.Var); isVar && !local.IsField() && local.Pkg() != nil && local.Parent() != nil && local.Parent() != local.Pkg().Scope() {
				order, seen := locals[object]
				if !seen {
					order = len(locals)
					locals[object] = order
				}
				out.WriteString(" L" + strconv.Itoa(order))
			} else {
				out.WriteString(" ?")
				values = append(values, typed.Name)
			}
		case *ast.BasicLit:
			out.WriteString(" ?")
			values = append(values, typed.Value)
		case *ast.BinaryExpr:
			out.WriteString(" " + typed.Op.String())
		case *ast.UnaryExpr:
			out.WriteString(" " + typed.Op.String())
		case *ast.AssignStmt:
			out.WriteString(" " + typed.Tok.String())
		case *ast.IncDecStmt:
			out.WriteString(" " + typed.Tok.String())
		case *ast.BranchStmt:
			out.WriteString(" " + typed.Tok.String())
		}
		return true
	})
	return out.String(), values, nodes
}

// bodySite is one body of repeatLeast nodes or more: where it stands, its nodes and the values its skeleton holds.
type bodySite struct {
	at     string
	nodes  int
	values []string
}

// linkedPackages loads the module at root and the import paths of the packages its commands link.
func linkedPackages(t *testing.T, root string) (*moduleLoad, map[string]bool) {
	t.Helper()
	load, err := loadModule(t.Context(), root)
	testcheck.NoError(t, err)
	linked := map[string]bool{}
	for importPath, checked := range load.module {
		if checked.listed.Name == "main" {
			for linkedPath := range load.closure(importPath) {
				linked[linkedPath] = true
			}
		}
	}
	return load, linked
}

// productionBodies reads the function and closure bodies of repeatLeast nodes or more of the module's linked packages,
// outside generated files and outside inside, by skeleton.
func productionBodies(t *testing.T, root string, inside func(directory string) bool) map[string][]bodySite {
	t.Helper()
	load, linked := linkedPackages(t, root)
	skeletons := map[string][]bodySite{}
	for importPath, checked := range load.module {
		directory, _ := filepath.Rel(load.root, checked.listed.Dir)
		if !linked[importPath] || inside(filepath.ToSlash(directory)) {
			continue
		}
		for _, file := range checked.files {
			if ast.IsGenerated(file) {
				continue
			}
			ast.Inspect(file, func(node ast.Node) bool {
				var body *ast.BlockStmt
				switch typed := node.(type) {
				case *ast.FuncDecl:
					body = typed.Body
				case *ast.FuncLit:
					body = typed.Body
				}
				if body == nil {
					return true
				}
				if skeleton, values, nodes := bodyShape(body, checked.info); nodes >= repeatLeast {
					place := load.fset.Position(body.Pos())
					relative, _ := filepath.Rel(load.root, place.Filename)
					skeletons[skeleton] = append(skeletons[skeleton], bodySite{fmt.Sprintf("%s:%d", filepath.ToSlash(relative), place.Line), nodes, values})
				}
				return true
			})
		}
	}
	return skeletons
}

// differing counts the values two bodies of one skeleton differ in.
func differing(left, right []string) int {
	count := 0
	for index := range left {
		if left[index] != right[index] {
			count++
		}
	}
	return count
}

// bodyRepeats groups the bodies of each skeleton into repeats: with most = 0 those of one list of values, and with most
// above it those whose values differ from the first of their group's in a value or more and at most most, a group
// whose values are all one counted with no parameter. Each skeleton's bodies are read in the order of where they
// stand, since the distance is not transitive and the first of a group decides who joins it. It returns each repeat
// with where its bodies stand, and the nodes repeated beyond one copy of each.
func bodyRepeats(skeletons map[string][]bodySite, most int) ([][]string, int) {
	var groups [][]string
	repeated := 0
	for _, sites := range skeletons {
		sites = slices.SortedFunc(slices.Values(sites), func(a, b bodySite) int { return strings.Compare(a.at, b.at) })
		var open [][]bodySite
		for _, site := range sites {
			at := slices.IndexFunc(open, func(group []bodySite) bool { return differing(group[0].values, site.values) <= most })
			if at < 0 {
				open = append(open, []bodySite{site})
			} else {
				open[at] = append(open[at], site)
			}
		}
		for _, group := range open {
			parameters := slices.ContainsFunc(group[1:], func(site bodySite) bool { return differing(group[0].values, site.values) > 0 })
			if len(group) < 2 || most > 0 && !parameters {
				continue
			}
			repeated += (len(group) - 1) * group[0].nodes
			named := []string{strconv.Itoa(group[0].nodes)}
			for _, site := range group {
				named = append(named, site.at)
			}
			slices.Sort(named[1:])
			groups = append(groups, named)
		}
	}
	slices.SortFunc(groups, func(a, b []string) int { return strings.Compare(a[1], b[1]) })
	return groups, repeated
}

// holdCeiling refuses repeated nodes of the production code where names over ceiling, and under it, since a ceiling
// only falls, logging each repeat.
func holdCeiling(t *testing.T, where, kind string, groups [][]string, repeated, ceiling int) {
	t.Helper()
	for _, group := range groups {
		t.Logf("%s nodes x%d: %s", group[0], len(group)-1, strings.Join(group[1:], " "))
	}
	if repeated > ceiling {
		t.Errorf("production %s repeats %d nodes beyond one copy of each %s, over the ceiling of %d, so fold the copy into the body it repeats", where, repeated, kind, ceiling)
	}
	if repeated < ceiling {
		t.Errorf("production %s repeats %d nodes beyond one copy of each %s, under the ceiling of %d, so lower the ceiling to %d, since it only falls", where, repeated, kind, ceiling, repeated)
	}
}

// repeatFixture is a module of bodies of one skeleton: A and B differ only in their locals' names, C from A in one
// literal and D from A in four, and E is too small to count.
func repeatFixture(t *testing.T) string {
	t.Helper()
	const body = "\tvar sum int\n\tfor _, each := range %s {\n\t\tif each > %d && each < %d && each != %d && each != %d {\n\t\t\tsum += each * each\n\t\t}\n\t}\n\treturn sum\n"
	return fixtureModule(t, map[string]string{
		"cmd/tool/main.go": "package main\n\nimport \"overgo/internal/lib\"\n\nfunc main() { println(lib.A(nil), lib.B(nil), lib.C(nil), lib.D(nil), lib.E(nil)) }\n",
		"internal/lib/lib.go": "package lib\n\n" +
			"func A(values []int) int {\n" + fmt.Sprintf(body, "values", 1, 9, 3, 5) + "}\n\n" +
			"func B(numbers []int) int {\n" + fmt.Sprintf(body, "numbers", 1, 9, 3, 5) + "}\n\n" +
			"func C(values []int) int {\n" + fmt.Sprintf(body, "values", 2, 9, 3, 5) + "}\n\n" +
			"func D(values []int) int {\n" + fmt.Sprintf(body, "values", 2, 8, 4, 6) + "}\n\n" +
			"func E(values []int) int { return len(values) }\n",
	})
}

// TestRepeatedProductionBodiesStayUnderTheirCeiling holds the repeats of production code outside the inference surface
// to repeatCeiling, so a copy folded into one place does not come back: a function or closure body of repeatLeast
// nodes or more that another has, once their locals' names are set aside, counts its nodes for each copy beyond the
// first. A fixture module pins the reading: two bodies that differ in their locals' names are one repeat, and a body
// that differs in a literal, or one too small, is none.
func TestRepeatedProductionBodiesStayUnderTheirCeiling(t *testing.T) {
	t.Parallel()
	groups, repeated := bodyRepeats(productionBodies(t, repeatFixture(t), func(string) bool { return false }), 0)
	if len(groups) != 1 || !slices.Equal(groups[0][1:], []string{"internal/lib/lib.go:13", "internal/lib/lib.go:3"}) || repeated == 0 {
		t.Fatalf("the fixture's repeats are %q (%d nodes beyond one copy), want A and B alone, which differ in their locals' names", groups, repeated)
	}
	inside := surfaceDirectories(t)
	groups, repeated = bodyRepeats(productionBodies(t, checkoutRoot, func(directory string) bool { return inside[directory] }), 0)
	holdCeiling(t, "outside the inference surface", "body", groups, repeated, repeatCeiling)
}

// TestParameterRepeatsStayUnderTheirCeiling holds the repeats of production code outside the inference surface that
// differ in a value or more and at most parameterLeast to parameterCeiling: bodies of one skeleton whose literals and
// names other than their locals differ in that many places, which one body with those values as its parameters would
// hold. A fixture module pins the reading: a body that differs from another in one literal is a repeat of it, and one
// that differs in four is none.
func TestParameterRepeatsStayUnderTheirCeiling(t *testing.T) {
	t.Parallel()
	skeletons := productionBodies(t, repeatFixture(t), func(string) bool { return false })
	groups, repeated := bodyRepeats(skeletons, parameterLeast)
	if len(groups) != 1 || !slices.Equal(groups[0][1:], []string{"internal/lib/lib.go:13", "internal/lib/lib.go:23", "internal/lib/lib.go:3"}) || repeated == 0 {
		t.Fatalf("the fixture's parameter repeats are %q (%d nodes beyond one copy), want A, B and C, which differ in at most one value, and not D, which differs in four", groups, repeated)
	}
	for _, sites := range skeletons {
		slices.Reverse(sites)
	}
	if reversed, again := bodyRepeats(skeletons, parameterLeast); !reflect.DeepEqual(reversed, groups) || again != repeated {
		t.Fatalf("the fixture's parameter repeats read in reverse are %q (%d nodes), want %q (%d nodes), since C, read first, would take D", reversed, again, groups, repeated)
	}
	inside := surfaceDirectories(t)
	groups, repeated = bodyRepeats(productionBodies(t, checkoutRoot, func(directory string) bool { return inside[directory] }), parameterLeast)
	holdCeiling(t, "outside the inference surface", "parameterised body", groups, repeated, parameterCeiling)
}

// TestSurfaceRepeatsStayUnderTheirCeiling holds the repeats of production code inside the inference surface, read as
// the two ceilings outside it read theirs, to surfaceRepeatCeiling for bodies alike once their locals' names are set
// aside and to surfaceParameterCeiling for bodies that differ in a value or more and at most parameterLeast.
func TestSurfaceRepeatsStayUnderTheirCeiling(t *testing.T) {
	t.Parallel()
	inside := surfaceDirectories(t)
	skeletons := productionBodies(t, checkoutRoot, func(directory string) bool { return !inside[directory] })
	groups, repeated := bodyRepeats(skeletons, 0)
	holdCeiling(t, "inside the inference surface", "body", groups, repeated, surfaceRepeatCeiling)
	groups, repeated = bodyRepeats(skeletons, parameterLeast)
	holdCeiling(t, "inside the inference surface", "parameterised body", groups, repeated, surfaceParameterCeiling)
}
