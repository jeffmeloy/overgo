package repoanalysis

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"

	"overgo/internal/testcheck"
)

// threeLineStoreCeiling is the most three-line store openings the module's tests may keep: what compact-tests/store-
// helper and compact-tests/call-checks left, a helper's whose deferred close falls at its own return and whose err it
// reads again, which neither one call nor one expression takes.
const threeLineStoreCeiling = 1

// threeLineStores counts the store openings of a test file that say in three lines what overgodbtest says in one: a
// store and err assigned from overgodb's Open, Create or OpenReadOnly, the err checked by testcheck.NoError, and the
// store closed by a defer or a cleanup of the test.
func threeLineStores(file *ast.File) int {
	count := 0
	ast.Inspect(file, func(node ast.Node) bool {
		block, isBlock := node.(*ast.BlockStmt)
		if !isBlock {
			return true
		}
		for index := 0; index+2 < len(block.List); index++ {
			assign, isAssign := block.List[index].(*ast.AssignStmt)
			if !isAssign || len(assign.Lhs) != 2 || len(assign.Rhs) != 1 || !isNamed(assign.Lhs[1], "err") {
				continue
			}
			store, isIdent := assign.Lhs[0].(*ast.Ident)
			call, isCall := assign.Rhs[0].(*ast.CallExpr)
			if !isIdent || !isCall {
				continue
			}
			opener, isSelector := call.Fun.(*ast.SelectorExpr)
			if !isSelector || !isNamed(opener.X, "overgodb") || opener.Sel.Name != "Open" && opener.Sel.Name != "Create" && opener.Sel.Name != "OpenReadOnly" {
				continue
			}
			check, isExpression := block.List[index+1].(*ast.ExprStmt)
			if !isExpression {
				continue
			}
			checked, isCheck := check.X.(*ast.CallExpr)
			if !isCheck || len(checked.Args) != 2 || !isNamed(checked.Args[1], "err") {
				continue
			}
			if named, isSelector := checked.Fun.(*ast.SelectorExpr); !isSelector || !isNamed(named.X, "testcheck") || named.Sel.Name != "NoError" {
				continue
			}
			if closesAtTheEnd(block.List[index+2], store.Name) {
				count++
				index += 2
			}
		}
		return true
	})
	return count
}

// closesAtTheEnd reports whether statement closes store when its function or its test ends: defer store.Close(), or a
// cleanup whose one statement closes it.
func closesAtTheEnd(statement ast.Stmt, store string) bool {
	isClose := func(expression ast.Expr) bool {
		call, isCall := expression.(*ast.CallExpr)
		if !isCall {
			return false
		}
		selector, isSelector := call.Fun.(*ast.SelectorExpr)
		return isSelector && len(call.Args) == 0 && isNamed(selector.X, store) && selector.Sel.Name == "Close"
	}
	switch typed := statement.(type) {
	case *ast.DeferStmt:
		return isClose(typed.Call)
	case *ast.ExprStmt:
		call, isCall := typed.X.(*ast.CallExpr)
		if !isCall || len(call.Args) != 1 {
			return false
		}
		if selector, isSelector := call.Fun.(*ast.SelectorExpr); !isSelector || selector.Sel.Name != "Cleanup" {
			return false
		}
		cleanup, isLiteral := call.Args[0].(*ast.FuncLit)
		if !isLiteral || len(cleanup.Body.List) != 1 {
			return false
		}
		switch closing := cleanup.Body.List[0].(type) {
		case *ast.ExprStmt:
			return isClose(closing.X)
		case *ast.AssignStmt:
			return len(closing.Lhs) == 1 && isNamed(closing.Lhs[0], "_") && len(closing.Rhs) == 1 && isClose(closing.Rhs[0])
		}
	}
	return false
}

// TestATestOpensItsStoreInOneCall holds the module's tests to threeLineStoreCeiling three-line store openings, so a
// test opens its throwaway store in one overgodbtest call, which checks the open and closes the store when the test
// ends, and the three-line form does not come back. A fixture pins the reading: an opening closed by a defer and one
// closed by a cleanup count, and one left open, one whose check is another's and another package's opener do not.
func TestATestOpensItsStoreInOneCall(t *testing.T) {
	t.Parallel()
	const fixture = `package p

import "testing"

func opened(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	testcheck.NoError(t, err)
	defer store.Close()
	created, err := overgodb.Create(directory)
	testcheck.NoError(t, err)
	t.Cleanup(func() { _ = created.Close() })
	kept, err := overgodb.OpenReadOnly(directory)
	testcheck.NoError(t, err)
	use(kept)
	checked, err := overgodb.Open(directory)
	other.NoError(t, err)
	defer checked.Close()
	file, err := os.Open(path)
	testcheck.NoError(t, err)
	defer file.Close()
}
`
	parsed := testcheck.Must(parser.ParseFile(token.NewFileSet(), "p_test.go", fixture, parser.SkipObjectResolution))(t)
	if got := threeLineStores(parsed); got != 2 {
		t.Fatalf("the fixture holds %d three-line store openings, want the deferred close's and the cleanup's", got)
	}
	total := testFileCount(t, threeLineStores)
	if total > threeLineStoreCeiling {
		t.Errorf("the module's tests hold %d three-line store openings, over the ceiling of %d, so open each with overgodbtest", total, threeLineStoreCeiling)
	}
	if total < threeLineStoreCeiling {
		t.Errorf("the module's tests hold %d three-line store openings, under the ceiling of %d, so lower the ceiling to %d, since it only falls", total, threeLineStoreCeiling, total)
	}
}
