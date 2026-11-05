package repoanalysis

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// threeLineCheckCeiling is the most three-line error checks the module's tests may keep: what
// compact-tests/fatal-check left, in tests that a compatibility claim or a pinned record binds, which change with a
// verification of their own, in the CUDA executor's and driver's, which change with the next landing that runs the
// device lane in full, and those whose init spans lines or assigns an err declared before it.
const threeLineCheckCeiling = 1962

// threeLineChecks counts the checks of a test file that say in three lines what testcheck.NoError says in one: an if
// whose condition is err != nil, with no else and no else-if before it, and whose body is t.Fatal(err), tb's or b's,
// bare or after an init that assigns err from one call.
func threeLineChecks(file *ast.File) int {
	elseIf := map[*ast.IfStmt]bool{}
	count := 0
	ast.Inspect(file, func(node ast.Node) bool {
		statement, isIf := node.(*ast.IfStmt)
		if !isIf {
			return true
		}
		if nested, isNested := statement.Else.(*ast.IfStmt); isNested {
			elseIf[nested] = true
		}
		if elseIf[statement] || statement.Else != nil || len(statement.Body.List) != 1 {
			return true
		}
		condition, isBinary := statement.Cond.(*ast.BinaryExpr)
		if !isBinary || condition.Op != token.NEQ || !isNamed(condition.X, "err") || !isNamed(condition.Y, "nil") {
			return true
		}
		if statement.Init != nil {
			assign, isAssign := statement.Init.(*ast.AssignStmt)
			if !isAssign || len(assign.Lhs) != 1 || len(assign.Rhs) != 1 || !isNamed(assign.Lhs[0], "err") {
				return true
			}
			if _, isCall := assign.Rhs[0].(*ast.CallExpr); !isCall {
				return true
			}
		}
		expression, isExpression := statement.Body.List[0].(*ast.ExprStmt)
		if !isExpression {
			return true
		}
		call, isCall := expression.X.(*ast.CallExpr)
		if !isCall || len(call.Args) != 1 || !isNamed(call.Args[0], "err") {
			return true
		}
		selector, isSelector := call.Fun.(*ast.SelectorExpr)
		if isSelector && selector.Sel.Name == "Fatal" && (isNamed(selector.X, "t") || isNamed(selector.X, "tb") || isNamed(selector.X, "b")) {
			count++
		}
		return true
	})
	return count
}

func isNamed(expression ast.Expr, name string) bool {
	ident, isIdent := expression.(*ast.Ident)
	return isIdent && ident.Name == name
}

// TestATestChecksAnErrorInOneCall holds the module's tests to threeLineCheckCeiling three-line error checks, so a test
// says the check in one call, testcheck.NoError, and the three-line form does not come back. A fixture pins the
// reading: the bare check and its init form count, and an else-if, a check with an else, another receiver's and a
// formatted failure do not.
func TestATestChecksAnErrorInOneCall(t *testing.T) {
	t.Parallel()
	const fixture = `package p

import "testing"

func checked(t *testing.T) {
	var err error
	if err != nil {
		t.Fatal(err)
	}
	if err := run(); err != nil {
		t.Fatal(err)
	}
	if err == nil {
	} else if err != nil {
		t.Fatal(err)
	}
	if err != nil {
		t.Fatal(err)
	} else {
		t.Log("ran")
	}
	if err != nil {
		other.Fatal(err)
	}
	if err != nil {
		t.Fatalf("run: %v", err)
	}
}
`
	parsed, err := parser.ParseFile(token.NewFileSet(), "p_test.go", fixture, parser.SkipObjectResolution)
	testcheck.NoError(t, err)
	if got := threeLineChecks(parsed); got != 2 {
		t.Fatalf("the fixture holds %d three-line checks, want the bare check and its init form", got)
	}
	total := 0
	for _, top := range []string{"cmd", "internal"} {
		testcheck.NoError(t, filepath.WalkDir(filepath.Join(checkoutRoot, top), func(path string, entry fs.DirEntry, err error) error {
			if err != nil || entry.IsDir() || !strings.HasSuffix(path, "_test.go") || strings.Contains(filepath.ToSlash(path), "/testdata/") {
				return err
			}
			source, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			file, err := parser.ParseFile(token.NewFileSet(), path, source, parser.SkipObjectResolution)
			if err != nil {
				return err
			}
			total += threeLineChecks(file)
			return nil
		}))
	}
	if total > threeLineCheckCeiling {
		t.Errorf("the module's tests hold %d three-line error checks, over the ceiling of %d, so say each in one call with testcheck.NoError", total, threeLineCheckCeiling)
	}
	if total < threeLineCheckCeiling {
		t.Errorf("the module's tests hold %d three-line error checks, under the ceiling of %d, so lower the ceiling to %d, since it only falls", total, threeLineCheckCeiling, total)
	}
}
