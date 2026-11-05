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
)

// repeatLeast is the fewest syntax nodes a body holds for its repeats to count: a body smaller than that is an idiom,
// which a helper would not shorten.
const repeatLeast = 20

// repeatCeiling is the most syntax nodes the production bodies outside the inference surface may repeat beyond one
// copy of each: what compact-production/exact-folds-outside left, repeats of types a helper would not unite or that
// wait for a shared home inside the surface.
const repeatCeiling = 334

// bodyShape writes a body as the checker reads it, with each local variable named by the order it first appears, so
// two bodies that differ only in the names of their locals have one shape: each node's kind, each operator and literal,
// each name that is not a local's.
func bodyShape(body ast.Node, info *types.Info) (shape string, nodes int) {
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
				out.WriteString(" " + typed.Name)
			}
		case *ast.BasicLit:
			out.WriteString(" " + typed.Value)
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
	return out.String(), nodes
}

// productionRepeats groups the function and closure bodies of the module's linked packages, outside generated files
// and outside inside, by shape, and returns each shape that more than one body has with where they stand, and the
// nodes repeated beyond one copy of each.
func productionRepeats(t *testing.T, root string, inside func(directory string) bool) ([][]string, int) {
	t.Helper()
	load, err := loadModule(t.Context(), root)
	if err != nil {
		t.Fatal(err)
	}
	linked := map[string]bool{}
	for importPath, checked := range load.module {
		if checked.listed.Name == "main" {
			for linkedPath := range load.closure(importPath) {
				linked[linkedPath] = true
			}
		}
	}
	type site struct {
		at    string
		nodes int
	}
	shapes := map[string][]site{}
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
				if shape, nodes := bodyShape(body, checked.info); nodes >= repeatLeast {
					place := load.fset.Position(body.Pos())
					relative, _ := filepath.Rel(load.root, place.Filename)
					shapes[shape] = append(shapes[shape], site{fmt.Sprintf("%s:%d", filepath.ToSlash(relative), place.Line), nodes})
				}
				return true
			})
		}
	}
	var groups [][]string
	repeated := 0
	for _, sites := range shapes {
		if len(sites) < 2 {
			continue
		}
		repeated += (len(sites) - 1) * sites[0].nodes
		var at []string
		for _, held := range sites {
			at = append(at, held.at)
		}
		slices.Sort(at)
		groups = append(groups, append([]string{strconv.Itoa(sites[0].nodes)}, at...))
	}
	slices.SortFunc(groups, func(a, b []string) int { return strings.Compare(a[1], b[1]) })
	return groups, repeated
}

// TestRepeatedProductionBodiesStayUnderTheirCeiling holds the repeats of production code outside the inference surface
// to repeatCeiling, so a copy folded into one place does not come back: a function or closure body of repeatLeast
// nodes or more that another has, once their locals' names are set aside, counts its nodes for each copy beyond the
// first. A fixture module pins the reading: two bodies that differ in their locals' names are one shape, a body that
// calls another package's function is another, and a smaller body is no repeat.
func TestRepeatedProductionBodiesStayUnderTheirCeiling(t *testing.T) {
	t.Parallel()
	const body = "\tvar sum int\n\tfor _, each := range %s {\n\t\tif each > %d {\n\t\t\tsum += each * each\n\t\t}\n\t}\n\treturn sum\n"
	fixture := fixtureModule(t, map[string]string{
		"cmd/tool/main.go": "package main\n\nimport \"overgo/internal/lib\"\n\nfunc main() { println(lib.A(nil), lib.B(nil), lib.C(nil), lib.D(nil)) }\n",
		"internal/lib/lib.go": "package lib\n\n" +
			"func A(values []int) int {\n" + fmt.Sprintf(body, "values", 1) + "}\n\n" +
			"func B(numbers []int) int {\n" + fmt.Sprintf(body, "numbers", 1) + "}\n\n" +
			"func C(values []int) int {\n" + fmt.Sprintf(body, "values", 2) + "}\n\n" +
			"func D(values []int) int { return len(values) }\n",
	})
	groups, repeated := productionRepeats(t, fixture, func(string) bool { return false })
	if len(groups) != 1 || !slices.Equal(groups[0][1:], []string{"internal/lib/lib.go:13", "internal/lib/lib.go:3"}) || repeated == 0 {
		t.Fatalf("the fixture's repeats are %q (%d nodes beyond one copy), want A and B alone, which differ in their locals' names", groups, repeated)
	}

	inside := surfaceDirectories(t)
	groups, repeated = productionRepeats(t, checkoutRoot, func(directory string) bool { return inside[directory] })
	for _, group := range groups {
		t.Logf("%s nodes x%d: %s", group[0], len(group)-1, strings.Join(group[1:], " "))
	}
	if repeated > repeatCeiling {
		t.Errorf("production outside the inference surface repeats %d nodes beyond one copy of each body, over the ceiling of %d, so fold the copy into the body it repeats", repeated, repeatCeiling)
	}
	if repeated < repeatCeiling {
		t.Errorf("production outside the inference surface repeats %d nodes beyond one copy of each body, under the ceiling of %d, so lower repeatCeiling to %d, since the ceiling only falls", repeated, repeatCeiling, repeated)
	}
}
