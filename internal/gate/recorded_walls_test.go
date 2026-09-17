package gate

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// TestRecordedWallsReadTheCounter holds every production wall to the
// performance counter: a time.Since difference converted to an integer or
// read in integer units is a tick-clock reading recorded as a wall, and
// processmeasure is the one package that reads a clock for a record.
func TestRecordedWallsReadTheCounter(t *testing.T) {
	t.Parallel()
	root := repositoryRoot(t)
	var offenders []string
	for _, dir := range []string{"cmd", "internal"} {
		err := filepath.WalkDir(filepath.Join(root, dir), func(path string, entry fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if entry.IsDir() {
				if entry.Name() == "processmeasure" || entry.Name() == "testdata" {
					return filepath.SkipDir
				}
				return nil
			}
			if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
				return nil
			}
			fileSet := token.NewFileSet()
			parsed, err := parser.ParseFile(fileSet, path, nil, parser.SkipObjectResolution)
			if err != nil {
				return err
			}
			relative, err := filepath.Rel(root, path)
			if err != nil {
				return err
			}
			ast.Inspect(parsed, func(node ast.Node) bool {
				call, ok := node.(*ast.CallExpr)
				if ok && convertsRuntimeClock(call) {
					offenders = append(offenders, fmt.Sprintf("%s:%d", filepath.ToSlash(relative), fileSet.Position(call.Pos()).Line))
				}
				return true
			})
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	if len(offenders) != 0 {
		t.Fatalf("runtime clock differences recorded as walls; read them through processmeasure.Stopwatch:\n%s", strings.Join(offenders, "\n"))
	}
}

// convertsRuntimeClock reports a time.Since result converted to a number or
// read in integer units.
func convertsRuntimeClock(call *ast.CallExpr) bool {
	switch fun := call.Fun.(type) {
	case *ast.SelectorExpr:
		return slices.Contains([]string{"Nanoseconds", "Microseconds", "Milliseconds"}, fun.Sel.Name) && isTimeSince(fun.X)
	case *ast.Ident:
		return len(call.Args) == 1 && slices.Contains([]string{"int", "int64", "uint", "uint64", "float32", "float64"}, fun.Name) && isTimeSince(call.Args[0])
	}
	return false
}

func isTimeSince(expression ast.Expr) bool {
	call, ok := expression.(*ast.CallExpr)
	if !ok {
		return false
	}
	selector, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return false
	}
	pkg, ok := selector.X.(*ast.Ident)
	return ok && pkg.Name == "time" && selector.Sel.Name == "Since"
}
