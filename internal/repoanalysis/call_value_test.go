package repoanalysis

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"

	"overgo/internal/testcheck"
)

// valueThenCheckCeiling is the most value-then-check pairs the module's tests may keep: what compact-tests/call-checks
// left, where the pair's err is read again or declared there and used again, where a comment stands between its two
// statements, in the tests a pinned record of docs/verification binds whole, which change with a verification of their
// own, and in the tests whose change runs the device lane in full.
const valueThenCheckCeiling = 179

// valueThenChecks counts the pairs of a test file that take a call's value and check its error in two statements,
// v, err := f() then testcheck.NoError(t, err), which testcheck.Must says in one expression.
func valueThenChecks(file *ast.File) int {
	count := 0
	ast.Inspect(file, func(node ast.Node) bool {
		block, isBlock := node.(*ast.BlockStmt)
		if !isBlock {
			return true
		}
		for index := 0; index+1 < len(block.List); index++ {
			assign, isAssign := block.List[index].(*ast.AssignStmt)
			if !isAssign || len(assign.Lhs) != 2 || len(assign.Rhs) != 1 || !isNamed(assign.Lhs[1], "err") {
				continue
			}
			if _, isCall := assign.Rhs[0].(*ast.CallExpr); !isCall {
				continue
			}
			check, isExpression := block.List[index+1].(*ast.ExprStmt)
			if !isExpression {
				continue
			}
			call, isCall := check.X.(*ast.CallExpr)
			if !isCall || len(call.Args) != 2 || !isNamed(call.Args[1], "err") {
				continue
			}
			if selector, isSelector := call.Fun.(*ast.SelectorExpr); isSelector && isNamed(selector.X, "testcheck") && selector.Sel.Name == "NoError" {
				count++
				index++
			}
		}
		return true
	})
	return count
}

// TestATestTakesACallsValueInOneExpression holds the module's tests to valueThenCheckCeiling value-then-check pairs,
// so a test takes a call's value and checks its error in one expression, testcheck.Must(f())(t), and the two-statement
// form does not come back. A fixture pins the reading: a defined and an assigned pair count, and a pair whose check is
// another's, a call that returns err alone and a value taken from no call do not.
func TestATestTakesACallsValueInOneExpression(t *testing.T) {
	t.Parallel()
	const fixture = `package p

func checked(t *testing.T) {
	data, err := os.ReadFile(path)
	testcheck.NoError(t, err)
	data, err = os.ReadFile(path)
	testcheck.NoError(t, err)
	other, err := os.ReadFile(path)
	require.NoError(t, err)
	err := run()
	testcheck.NoError(t, err)
	value, err := cached, nil
	testcheck.NoError(t, err)
}
`
	parsed, err := parser.ParseFile(token.NewFileSet(), "p_test.go", fixture, parser.SkipObjectResolution)
	testcheck.NoError(t, err)
	if got := valueThenChecks(parsed); got != 2 {
		t.Fatalf("the fixture holds %d value-then-check pairs, want the defined and the assigned one", got)
	}
	total := testFileCount(t, valueThenChecks)
	if total > valueThenCheckCeiling {
		t.Errorf("the module's tests hold %d value-then-check pairs, over the ceiling of %d, so take each value in one expression with testcheck.Must", total, valueThenCheckCeiling)
	}
	if total < valueThenCheckCeiling {
		t.Errorf("the module's tests hold %d value-then-check pairs, under the ceiling of %d, so lower the ceiling to %d, since it only falls", total, valueThenCheckCeiling, total)
	}
}
