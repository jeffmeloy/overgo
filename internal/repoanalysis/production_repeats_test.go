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
// compact-production/surface-folds, consolidate/release-storages and consolidate/cell-step-reads left, repeats on a
// per-token path, which no fold takes without a benchmark, and typed accessors and constructors over different types,
// which a helper would not shorten.
const (
	surfaceRepeatCeiling    = 111
	surfaceParameterCeiling = 1429
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
	load := testcheck.Must(loadModule(t.Context(), root))(t)
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

// productionBodies reads the function and closure bodies of repeatLeast nodes or more of the loaded module's linked
// packages, outside generated files and outside inside, by skeleton.
func productionBodies(load *moduleLoad, linked map[string]bool, inside func(directory string) bool) map[string][]bodySite {
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

// TestRepeatedProductionBodiesStayUnderTheirCeiling holds the repeats and local forwarders of production code to their
// ceilings, reading the checkout once, so a copy folded into one place does not come back. Outside the inference
// surface a function or closure body of repeatLeast nodes or more that another has, once their locals' names are set
// aside, counts its nodes for each copy beyond the first toward repeatCeiling, and bodies of one skeleton whose
// literals and names other than their locals differ in a value or more and at most parameterLeast, which one body with
// those values as its parameters would hold, count toward parameterCeiling. Inside the surface the same two readings
// hold to surfaceRepeatCeiling and surfaceParameterCeiling. Outside it, a one-call function its package could say in
// place holds to forwarderCeiling. The module's tests hold to testRepeatMost bodies of one skeleton, alike or differing
// in at most parameterLeast values, so a body a third case repeats is folded into one the cases share, except the group
// testParameterHeld names, which must still stand. A fixture module pins each reading: two bodies that differ in their locals' names
// are one repeat, a body that differs from another in one literal is a parameter repeat of it, read in either order,
// and one that differs in four, or one too small, is none; a function that returns one call, called from its package,
// is a forwarder, and one that makes two calls, one that passes a function literal, one called from another package,
// one used as a value and one called past forwarderMost times are none.
func TestRepeatedProductionBodiesStayUnderTheirCeiling(t *testing.T) {
	t.Parallel()
	repeats, repeatsLinked := linkedPackages(t, repeatFixture(t))
	skeletons := productionBodies(repeats, repeatsLinked, func(string) bool { return false })
	groups, repeated := bodyRepeats(skeletons, 0)
	if len(groups) != 1 || !slices.Equal(groups[0][1:], []string{"internal/lib/lib.go:13", "internal/lib/lib.go:3"}) || repeated == 0 {
		t.Fatalf("the fixture's repeats are %q (%d nodes beyond one copy), want A and B alone, which differ in their locals' names", groups, repeated)
	}
	groups, repeated = bodyRepeats(skeletons, parameterLeast)
	if len(groups) != 1 || !slices.Equal(groups[0][1:], []string{"internal/lib/lib.go:13", "internal/lib/lib.go:23", "internal/lib/lib.go:3"}) || repeated == 0 {
		t.Fatalf("the fixture's parameter repeats are %q (%d nodes beyond one copy), want A, B and C, which differ in at most one value, and not D, which differs in four", groups, repeated)
	}
	for _, sites := range skeletons {
		slices.Reverse(sites)
	}
	if reversed, again := bodyRepeats(skeletons, parameterLeast); !reflect.DeepEqual(reversed, groups) || again != repeated {
		t.Fatalf("the fixture's parameter repeats read in reverse are %q (%d nodes), want %q (%d nodes), since C, read first, would take D", reversed, again, groups, repeated)
	}
	load, linked := linkedPackages(t, checkoutRoot)
	inside := surfaceDirectories(t)
	t.Run("outside the inference surface", func(t *testing.T) {
		outside := productionBodies(load, linked, func(directory string) bool { return inside[directory] })
		groups, repeated := bodyRepeats(outside, 0)
		holdCeiling(t, "outside the inference surface", "body", groups, repeated, repeatCeiling)
		groups, repeated = bodyRepeats(outside, parameterLeast)
		holdCeiling(t, "outside the inference surface", "parameterised body", groups, repeated, parameterCeiling)
	})
	t.Run("inside the inference surface", func(t *testing.T) {
		surface := productionBodies(load, linked, func(directory string) bool { return !inside[directory] })
		groups, repeated := bodyRepeats(surface, 0)
		holdCeiling(t, "inside the inference surface", "body", groups, repeated, surfaceRepeatCeiling)
		groups, repeated = bodyRepeats(surface, parameterLeast)
		holdCeiling(t, "inside the inference surface", "parameterised body", groups, repeated, surfaceParameterCeiling)
	})
	t.Run("local forwarders", func(t *testing.T) {
		fixture := fixtureModule(t, map[string]string{
			"cmd/tool/main.go": "package main\n\nimport \"overgo/internal/lib\"\n\nfunc main() { println(lib.Run(), lib.Shared(1)) }\n",
			"internal/lib/lib.go": "package lib\n\nimport (\n\t\"slices\"\n\t\"strconv\"\n)\n\n" +
				"func Run() int { apply(each); return len(twice(1)+both(2)+many(1)+many(2)+many(3)+many(4)+Shared(5)) + literal(6) }\n\n" +
				"func twice(value int) string { return strconv.Itoa(value * 2) }\n\n" +
				"func both(value int) string { return strconv.Quote(strconv.Itoa(value)) }\n\n" +
				"func literal(value int) int { return slices.IndexFunc([]int{value}, func(int) bool { return true }) }\n\n" +
				"func many(value int) string { return strconv.Itoa(value) }\n\n" +
				"func each(value int) int { return len(strconv.Itoa(value)) }\n\n" +
				"func apply(step func(int) int) { step(1) }\n\n" +
				"func Shared(value int) string { return strconv.Itoa(value) }\n",
		})
		forwarders, forwardersLinked := linkedPackages(t, fixture)
		found, total := localForwarders(forwarders, forwardersLinked, func(string) bool { return false })
		if len(found) != 2 || !strings.Contains(found[0], " twice ") || !strings.Contains(found[1], " apply ") || total == 0 {
			t.Fatalf("the fixture's forwarders are %q (%d nodes), want twice and apply alone", found, total)
		}
		found, total = localForwarders(load, linked, func(directory string) bool { return inside[directory] })
		for _, forwarder := range found {
			t.Log(forwarder)
		}
		if total > forwarderCeiling {
			t.Errorf("production outside the inference surface declares %d local forwarders of %d nodes, over the ceiling of %d, so say each call in place", len(found), total, forwarderCeiling)
		}
		if total < forwarderCeiling {
			t.Errorf("production outside the inference surface declares %d local forwarders of %d nodes, under the ceiling of %d, so lower the ceiling to %d, since it only falls", len(found), total, forwarderCeiling, total)
		}
	})
	tests := testBodies(t, load)
	t.Run("test bodies", func(t *testing.T) {
		groups, _ := bodyRepeats(tests, 0)
		over := 0
		for _, group := range groups {
			if len(group)-1 > testRepeatMost {
				over++
				t.Errorf("%s nodes x%d: %s", group[0], len(group)-1, strings.Join(group[1:], " "))
			}
		}
		if over > 0 {
			t.Errorf("%d test bodies stand more than %d times, so fold each into one body its cases share", over, testRepeatMost)
		}
	})
	t.Run("test bodies of a few values", func(t *testing.T) {
		groups, _ := bodyRepeats(tests, parameterLeast)
		over, held := 0, false
		for _, group := range groups {
			if len(group)-1 <= testRepeatMost {
				continue
			}
			files := make([]string, 0, len(group)-1)
			for _, site := range group[1:] {
				file, _, _ := strings.Cut(site, ":")
				files = append(files, file)
			}
			slices.Sort(files)
			if slices.Equal(files, testParameterHeld) {
				held = true
				continue
			}
			over++
			t.Errorf("%s nodes x%d: %s", group[0], len(group)-1, strings.Join(group[1:], " "))
		}
		if over > 0 {
			t.Errorf("%d test bodies of a few values stand more than %d times, so fold each into one body that takes them", over, testRepeatMost)
		}
		if !held {
			t.Errorf("the group testParameterHeld names, %q, no longer stands more than %d times, so drop it", testParameterHeld, testRepeatMost)
		}
	})
}
