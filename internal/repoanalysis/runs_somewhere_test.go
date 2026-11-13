package repoanalysis_test

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/gosyntax"
	"overgo/internal/landing"
	"overgo/internal/testcheck"
)

// awaitingPin are the gates of the tests that compare overgo with a llama.cpp build of the owner's pinning, the
// build and where it is kept not yet chosen (opt-in-tests/run-or-remove, 2026-11-13). Each gates its test until then,
// and the list only shrinks: an entry that gates nothing fails.
var awaitingPin = []string{"OVERGO_GGML_BASE_ORACLE", "OVERGO_GGUF_HASH_ORACLE", "OVERGO_GGUF_SPLIT_ORACLE", "OVERGO_LLAMA_CPP"}

// skipCalls name the calls that hold a test out: testing's skips and testskip's.
var skipCalls = []string{"Skip", "Skipf", "SkipNow", "NotApplicable"}

// gatedVariables are the environment variables a file's guards skip on: a guard whose condition reads a variable the
// file names, as a literal, a constant of the file or a name the file assigned from the environment, and whose body
// skips or returns. A variable read for anything else, such as the device a fixture opens on, gates nothing.
func gatedVariables(syntax *ast.File) []string {
	constants, assigned := fileConstants(syntax), map[string]string{}
	variable := func(call ast.Expr) string {
		read, ok := call.(*ast.CallExpr)
		if !ok || len(read.Args) != 1 || !readsEnvironment(read.Fun) {
			return ""
		}
		switch argument := read.Args[0].(type) {
		case *ast.BasicLit:
			name, _ := strconv.Unquote(argument.Value)
			return name
		case *ast.Ident:
			return constants[argument.Name]
		}
		return ""
	}
	ast.Inspect(syntax, func(node ast.Node) bool {
		if declared, ok := node.(*ast.AssignStmt); ok {
			for index, value := range declared.Rhs {
				if name, ok := declared.Lhs[min(index, len(declared.Lhs)-1)].(*ast.Ident); ok && variable(value) != "" {
					assigned[name.Name] = variable(value)
				}
			}
		}
		return true
	})
	var gated []string
	ast.Inspect(syntax, func(node ast.Node) bool {
		guard, ok := node.(*ast.IfStmt)
		if !ok || !slices.ContainsFunc(guard.Body.List, holdsOut) {
			return true
		}
		// A guard holds the test out while the variable is unset where it asks for it empty or for another value: a guard
		// that holds out once the variable is set, as a fixture's rewrite switch does, gates a mode, not the test.
		ast.Inspect(guard.Cond, func(node ast.Node) bool {
			compared, ok := node.(*ast.BinaryExpr)
			if !ok || compared.Op != token.EQL && compared.Op != token.NEQ {
				return true
			}
			literal, ok := compared.Y.(*ast.BasicLit)
			if !ok || literal.Kind != token.STRING {
				return true
			}
			value, _ := strconv.Unquote(literal.Value)
			if (compared.Op == token.EQL) != (value == "") {
				return true
			}
			if name := variable(compared.X); name != "" {
				gated = append(gated, name)
			} else if ident, ok := compared.X.(*ast.Ident); ok && assigned[ident.Name] != "" {
				gated = append(gated, assigned[ident.Name])
			}
			return true
		})
		return true
	})
	slices.Sort(gated)
	return slices.Compact(gated)
}

// setVariables are the environment variables a file sets: a string that opens "NAME=", a name joined to "=" (a literal
// or a constant of the file), or a Setenv call's first argument.
func setVariables(syntax *ast.File) map[string]bool {
	constants := fileConstants(syntax)
	named := func(expression ast.Expr) string {
		switch value := expression.(type) {
		case *ast.BasicLit:
			text, _ := strconv.Unquote(value.Value)
			return text
		case *ast.Ident:
			return constants[value.Name]
		}
		return ""
	}
	set := map[string]bool{}
	ast.Inspect(syntax, func(node ast.Node) bool {
		switch value := node.(type) {
		case *ast.BasicLit:
			if name, _, found := strings.Cut(named(value), "="); found && environmentName(name) {
				set[name] = true
			}
		case *ast.BinaryExpr:
			if joined := named(value.Y); value.Op == token.ADD && strings.HasPrefix(joined, "=") && environmentName(named(value.X)) {
				set[named(value.X)] = true
			}
		case *ast.CallExpr:
			if selector, ok := value.Fun.(*ast.SelectorExpr); ok && selector.Sel.Name == "Setenv" && len(value.Args) > 0 {
				set[named(value.Args[0])] = true
			}
		}
		return true
	})
	return set
}

// environmentName reports a name written as environment variables are: capitals, digits and underscores.
func environmentName(name string) bool {
	return name != "" && strings.Trim(name, "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") == ""
}

// fileConstants are the string constants and variables a file declares with a literal value.
func fileConstants(syntax *ast.File) map[string]string {
	constants := map[string]string{}
	ast.Inspect(syntax, func(node ast.Node) bool {
		if declared, ok := node.(*ast.ValueSpec); ok {
			for index, name := range declared.Names {
				if index < len(declared.Values) {
					if literal, ok := declared.Values[index].(*ast.BasicLit); ok && literal.Kind == token.STRING {
						constants[name.Name], _ = strconv.Unquote(literal.Value)
					}
				}
			}
		}
		return true
	})
	return constants
}

// asExpr is node as an expression, or nil.
func asExpr(node ast.Node) ast.Expr {
	expression, _ := node.(ast.Expr)
	return expression
}

// holdsOut reports a statement that ends a test there: a skip, or a return.
func holdsOut(statement ast.Stmt) bool {
	if _, returns := statement.(*ast.ReturnStmt); returns {
		return true
	}
	expression, ok := statement.(*ast.ExprStmt)
	if !ok {
		return false
	}
	call, ok := expression.X.(*ast.CallExpr)
	if !ok {
		return false
	}
	selector, ok := call.Fun.(*ast.SelectorExpr)
	return ok && slices.Contains(skipCalls, selector.Sel.Name)
}

// TestEveryTestRunsSomewhere holds every test to running somewhere: a test held out unless an environment variable is
// set runs only where a lane of the measurement run sets it (landing.CycleLanes), so a variable no lane sets leaves its
// test running nowhere. On 2026-11-12 31 tests were gated on variables no lane set or on models this machine lacked;
// each now runs in the measurement run, gated on its model's presence, or was removed (opt-in-tests/run-or-remove). A
// fixture pins the reading: a literal, a constant and an assigned gate are read, and a variable read as an option is
// not.
func TestEveryTestRunsSomewhere(t *testing.T) {
	t.Parallel()
	const fixture = "package p\n\nconst wide = \"WIDE\"\n\nfunc TestA(t *testing.T) {\n\tif os.Getenv(\"LITERAL\") != \"1\" {\n" +
		"\t\tt.Skip(\"set LITERAL\")\n\t}\n\tif os.Getenv(wide) == \"\" {\n\t\treturn\n\t}\n\tpath := os.Getenv(\"ASSIGNED\")\n" +
		"\tif path == \"\" {\n\t\ttestskip.NotApplicable(t, \"set ASSIGNED\")\n\t}\n\topen(Options{CUDA: os.Getenv(\"OPTION\") != \"\"})\n}\n"
	syntax := testcheck.Must(parser.ParseFile(token.NewFileSet(), "p_test.go", fixture, parser.SkipObjectResolution))(t)
	if got := gatedVariables(syntax); !slices.Equal(got, []string{"ASSIGNED", "LITERAL", "WIDE"}) {
		t.Fatalf("the fixture's gates read as %v, want the literal, the constant and the assigned one", got)
	}
	// A variable is set where a lane of the measurement run sets it, or where the repository's own code does: a lane of
	// the landing's, a test that starts itself as a child, as "NAME=value" or through Setenv.
	set := map[string]bool{}
	for _, lane := range landing.CycleLanes() {
		for _, entry := range lane.Environment {
			name, _, _ := strings.Cut(entry, "=")
			set[name] = true
		}
	}
	repo := testcheck.Must(filepath.Abs(filepath.Join("..", "..")))(t)
	tree := testcheck.Must(gosyntax.DiscoverGo(repo, "internal", "cmd"))(t)
	gates := map[string][]string{}
	for _, file := range tree.Files {
		syntax := testcheck.Must(file.Syntax())(t)
		for name := range setVariables(syntax) {
			set[name] = true
		}
		if strings.HasSuffix(file.Path, "_test.go") {
			for _, name := range gatedVariables(syntax) {
				gates[name] = append(gates[name], filepath.ToSlash(file.Path))
			}
		}
	}
	var nowhere []string
	for name, files := range gates {
		if !set[name] && !slices.Contains(awaitingPin, name) {
			nowhere = append(nowhere, name+": "+strings.Join(files, ", "))
		}
	}
	for _, name := range awaitingPin {
		if len(gates[name]) == 0 {
			nowhere = append(nowhere, name+" gates no test, so it leaves awaitingPin")
		}
	}
	slices.Sort(nowhere)
	if len(nowhere) != 0 {
		t.Fatalf("tests skip unless a variable no lane of the measurement run sets is set, so they run nowhere; gate each on "+
			"its model's presence, run it in a lane, or remove it, naming what it measured:\n%s", strings.Join(nowhere, "\n"))
	}
}
