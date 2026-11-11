package landing

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"strings"
)

// wholeTreeTag builds the tests that read the whole tree, each in a file named with wholeTreeSuffix: a measurement's
// checkout copied from it, the lane discovery run over it. go test counts what a test reads among its package's
// inputs, so built with the rest of their package's tests they would run all of that suite again on any change; built
// apart, the invariants stage runs them on every landing, and their packages' other tests are replayed until their own
// code changes (test-reuse).
const wholeTreeTag = "wholetree"

// wholeTreeSuffix ends the name of a file of whole-tree tests, which carries wholeTreeTag.
const wholeTreeSuffix = "_wholetree_test.go"

// wholeTreePackages are the packages whose suites hold whole-tree tests: the commit tool's and the driver's.
var wholeTreePackages = []string{"./internal/landing", "./cmd/loop"}

// wholeTreeRun is the go test arguments that run the whole-tree tests of root, those its whole-tree files declare, and
// none when it has none (a fixture's tree).
func wholeTreeRun(root string) ([]string, error) {
	var names, packages []string
	for _, pattern := range wholeTreePackages {
		files, err := filepath.Glob(filepath.Join(root, filepath.FromSlash(pattern), "*"+wholeTreeSuffix))
		if err != nil {
			return nil, err
		}
		for _, name := range files {
			parsed, err := parser.ParseFile(token.NewFileSet(), name, nil, parser.SkipObjectResolution)
			if err != nil {
				return nil, err
			}
			for _, declaration := range parsed.Decls {
				if function, ok := declaration.(*ast.FuncDecl); ok && function.Recv == nil && strings.HasPrefix(function.Name.Name, "Test") {
					names = append(names, function.Name.Name)
				}
			}
		}
		if len(files) > 0 {
			packages = append(packages, pattern)
		}
	}
	if len(names) == 0 {
		return nil, nil
	}
	return append([]string{"-tags", wholeTreeTag, "-run", "^(" + strings.Join(names, "|") + ")$"}, packages...), nil
}
