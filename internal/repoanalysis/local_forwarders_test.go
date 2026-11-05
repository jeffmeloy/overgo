package repoanalysis

import (
	"fmt"
	"go/ast"
	"go/types"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// forwarderMost is the most calls a local forwarder may have: past it a name stands for enough code to keep.
const forwarderMost = 3

// forwarderCeiling is the most syntax nodes the local forwarders of production outside the inference surface may
// declare: what compact-production/inline-forwarders left, names that read better than their call, which binds a flag,
// a literal, a key's format or a formula, entries that tests call many times by name, and speech recognition's profile
// readers, whose callers' file a compatibility claim binds.
const forwarderCeiling = 1272

// localForwarders reads the local forwarders of the module's linked packages outside generated files and outside
// inside: functions, not methods, whose body is one statement that makes one call, a conversion aside, called from
// production code of their own package alone, at least once and at most forwarderMost times, and never used as a value.
// It returns each with where it stands and its nodes, and their nodes in all.
func localForwarders(t *testing.T, root string, inside func(directory string) bool) ([]string, int) {
	t.Helper()
	load, linked := linkedPackages(t, root)
	calls, values := map[types.Object]int{}, map[types.Object]bool{}
	callers := map[types.Object]map[*types.Package]bool{}
	for _, checked := range load.module {
		called := map[*ast.Ident]bool{}
		for _, file := range checked.files {
			ast.Inspect(file, func(node ast.Node) bool {
				if call, isCall := node.(*ast.CallExpr); isCall {
					switch fun := ast.Unparen(call.Fun).(type) {
					case *ast.Ident:
						called[fun] = true
					case *ast.SelectorExpr:
						called[fun.Sel] = true
					case *ast.IndexExpr:
						if name, isName := fun.X.(*ast.Ident); isName {
							called[name] = true
						}
					}
				}
				return true
			})
		}
		for ident, object := range checked.info.Uses {
			if _, isFunc := object.(*types.Func); !isFunc {
				continue
			}
			if !called[ident] {
				values[object] = true
				continue
			}
			calls[object]++
			if callers[object] == nil {
				callers[object] = map[*types.Package]bool{}
			}
			callers[object][checked.pkg] = true
		}
	}
	var found []string
	total := 0
	for importPath, checked := range load.module {
		directory, _ := filepath.Rel(load.root, checked.listed.Dir)
		if !linked[importPath] || inside(filepath.ToSlash(directory)) {
			continue
		}
		for _, file := range checked.files {
			if ast.IsGenerated(file) {
				continue
			}
			for _, declaration := range file.Decls {
				function, isFunction := declaration.(*ast.FuncDecl)
				if !isFunction || function.Recv != nil || function.Body == nil || len(function.Body.List) != 1 {
					continue
				}
				object := checked.info.Defs[function.Name]
				if calls[object] == 0 || calls[object] > forwarderMost || values[object] || len(callers[object]) != 1 || !callers[object][checked.pkg] {
					continue
				}
				var statement ast.Node = function.Body.List[0]
				if returned, isReturn := statement.(*ast.ReturnStmt); isReturn && len(returned.Results) == 1 {
					statement = returned.Results[0]
				} else if expression, isExpression := statement.(*ast.ExprStmt); isExpression {
					statement = expression.X
				}
				if _, isCall := statement.(*ast.CallExpr); !isCall || madeCalls(function.Body, checked.info) != 1 {
					continue
				}
				nodes := 0
				for _, part := range []ast.Node{function.Name, function.Type, function.Body} {
					ast.Inspect(part, func(node ast.Node) bool {
						if node != nil {
							nodes++
						}
						return true
					})
				}
				place := load.fset.Position(function.Pos())
				relative, _ := filepath.Rel(load.root, place.Filename)
				found = append(found, fmt.Sprintf("%s:%d %s %d nodes, %d calls", filepath.ToSlash(relative), place.Line, function.Name.Name, nodes, calls[object]))
				total += nodes
			}
		}
	}
	slices.Sort(found)
	return found, total
}

// madeCalls counts the calls node makes, conversions aside, and a function literal as two, since its body is code of
// its own.
func madeCalls(node ast.Node, info *types.Info) int {
	count := 0
	ast.Inspect(node, func(node ast.Node) bool {
		switch typed := node.(type) {
		case *ast.CallExpr:
			if !info.Types[typed.Fun].IsType() {
				count++
			}
		case *ast.FuncLit:
			count += 2
		}
		return true
	})
	return count
}

// TestLocalForwardersStayUnderTheirCeiling holds the local forwarders of production code outside the inference surface
// to forwarderCeiling, so a one-call function its package could say in place does not come back. A fixture module pins
// the reading: a function that returns one call, called from its package, is a forwarder, and one that makes two
// calls, one that passes a function literal, one called from another package, one used as a value and one called past
// forwarderMost times are none.
func TestLocalForwardersStayUnderTheirCeiling(t *testing.T) {
	t.Parallel()
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
	found, total := localForwarders(t, fixture, func(string) bool { return false })
	if len(found) != 2 || !strings.Contains(found[0], " twice ") || !strings.Contains(found[1], " apply ") || total == 0 {
		t.Fatalf("the fixture's forwarders are %q (%d nodes), want twice and apply alone", found, total)
	}
	inside := surfaceDirectories(t)
	found, total = localForwarders(t, checkoutRoot, func(directory string) bool { return inside[directory] })
	for _, forwarder := range found {
		t.Log(forwarder)
	}
	if total > forwarderCeiling {
		t.Errorf("production outside the inference surface declares %d local forwarders of %d nodes, over the ceiling of %d, so say each call in place", len(found), total, forwarderCeiling)
	}
	if total < forwarderCeiling {
		t.Errorf("production outside the inference surface declares %d local forwarders of %d nodes, under the ceiling of %d, so lower the ceiling to %d, since it only falls", len(found), total, forwarderCeiling, total)
	}
}
