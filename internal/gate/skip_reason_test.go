package gate

import (
	"go/ast"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/repoanalysis"
)

// skipReasonSites reports every integration skip that names the short-mode
// exclusion while its own guard reads the environment. Such a skip is
// classified only when the run is short, so in a complete run it leaves its
// package incomplete and the package earns no receipt at all.
// Every file is read, not only tests: a shared test gate such as the CUDA
// one lives in a non-test file of a support package, and its skip decides
// the verdict of every fixture that calls it.
func skipReasonSites(snapshot repoanalysis.SourceSnapshot) ([]string, error) {
	var sites []string
	for _, file := range snapshot.Files {
		syntax, err := file.Syntax()
		if err != nil {
			return nil, err
		}
		derived := environmentNames(syntax)
		ast.Inspect(syntax, func(node ast.Node) bool {
			statement, ok := node.(*ast.IfStmt)
			if !ok || !readsEnvironment(statement.Cond) && !decidesOnName(statement.Cond, derived) {
				return true
			}
			for _, inner := range statement.Body.List {
				if unclassifiedSkip(inner) {
					sites = append(sites, filepath.ToSlash(file.Path))
				}
			}
			return true
		})
	}
	return sites, nil
}

// environmentNames collects the identifiers a file assigns from the process
// environment. A guard that decides on one of them is the same gate as a
// guard that reads the environment inline, and is held to the same rule.
func environmentNames(syntax *ast.File) map[string]bool {
	names := map[string]bool{}
	ast.Inspect(syntax, func(node ast.Node) bool {
		assignment, ok := node.(*ast.AssignStmt)
		if !ok {
			return true
		}
		for index, value := range assignment.Rhs {
			if !readsEnvironment(value) || index >= len(assignment.Lhs) {
				continue
			}
			if name, ok := assignment.Lhs[index].(*ast.Ident); ok {
				names[name.Name] = true
			}
		}
		return true
	})
	return names
}

// mentions reports whether any node of an expression satisfies match, and
// stops at the first that does. Both ways a guard can name the environment
// are that question asked of a different node.
func mentions(expression ast.Expr, match func(ast.Node) bool) bool {
	found := false
	ast.Inspect(expression, func(node ast.Node) bool {
		found = found || match(node)
		return !found
	})
	return found
}

// decidesOnName reports a guard that decides on one of those identifiers.
func decidesOnName(expression ast.Expr, names map[string]bool) bool {
	return mentions(expression, func(node ast.Node) bool {
		name, ok := node.(*ast.Ident)
		return ok && names[name.Name]
	})
}

// readsEnvironment reports whether a guard decides on the process environment.
func readsEnvironment(expression ast.Expr) bool {
	return mentions(expression, func(node ast.Node) bool {
		selector, ok := node.(*ast.SelectorExpr)
		if !ok {
			return false
		}
		name, ok := selector.X.(*ast.Ident)
		return ok && name.Name == "os" && selector.Sel.Name == "Getenv"
	})
}

// unclassifiedSkip reports whether one statement skips without citing a
// reason a complete run classifies: the short-mode exclusion, which a
// complete run does not classify, or no classified reason at all.
func unclassifiedSkip(statement ast.Stmt) bool {
	expression, ok := statement.(*ast.ExprStmt)
	if !ok {
		return false
	}
	call, ok := expression.X.(*ast.CallExpr)
	if !ok {
		return false
	}
	selector, ok := call.Fun.(*ast.SelectorExpr)
	if !ok || selector.Sel.Name != "Skip" && selector.Sel.Name != "Skipf" {
		return false
	}
	classified := false
	for _, argument := range call.Args {
		ast.Inspect(argument, func(node ast.Node) bool {
			reason, ok := node.(*ast.SelectorExpr)
			if !ok {
				return true
			}
			name, ok := reason.X.(*ast.Ident)
			// The short-mode exclusion is classified only when the run is
			// short, so a guard that reads the environment and cites it
			// leaves its package incomplete in a complete run.
			classified = classified || ok && name.Name == "testskip" && reason.Sel.Name != "ShortIntegration"
			return !classified
		})
	}
	return !classified
}

// TestIntegrationSkipReasonNamesItsGate holds every integration skip to
// reporting the gate it actually answers to. A test held out by an
// environment variable is inapplicable wherever that variable is unset, in a
// short run and a complete one alike; calling it a short-mode exclusion
// leaves its package incomplete in a complete run, and an incomplete package
// records no verdict, so its receipt is never earned and the models it
// accepts never reach acceptance.
func TestIntegrationSkipReasonNamesItsGate(t *testing.T) {
	t.Parallel()
	repo, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := repoanalysis.DiscoverGo(repo, "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	sites, err := skipReasonSites(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	if len(sites) != 0 {
		t.Fatalf("environment-gated skips cite no reason a complete run classifies:\n%s", strings.Join(sites, "\n"))
	}
}

// TestSkippedFixtureClassification holds a shared test gate in a support
// package to the rule its fixtures are: an environment-gated skip in a
// non-test helper is reported unless it cites a classified reason, and the
// shared CUDA gate every device fixture calls cites one, so a complete run
// without the device classifies those fixtures instead of leaving their
// packages without a receipt.
func TestSkippedFixtureClassification(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	for name, source := range map[string]string{
		"go.mod": "module fixture\n\ngo 1.25\n",
		"internal/unclassified/gate.go": "package unclassified\n\nimport (\n\t\"os\"\n\t\"testing\"\n)\n\n" +
			"func Require(t testing.TB) {\n\tif os.Getenv(\"DEVICE\") == \"\" {\n\t\tt.Skip(\"set DEVICE=1\")\n\t}\n}\n",
		"internal/classified/gate.go": "package classified\n\nimport (\n\t\"os\"\n\t\"testing\"\n\n\t\"fixture/internal/testskip\"\n)\n\n" +
			"func Require(t testing.TB) {\n\tif os.Getenv(\"DEVICE\") == \"\" {\n\t\tt.Skip(testskip.Inapplicable + \": set DEVICE=1\")\n\t}\n}\n",
		"internal/testskip/testskip.go": "package testskip\n\nconst Inapplicable = \"inapplicable\"\n",
	} {
		path := filepath.Join(root, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(source), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	fixture, err := repoanalysis.DiscoverGo(root, "internal")
	if err != nil {
		t.Fatal(err)
	}
	sites, err := skipReasonSites(fixture)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(sites, []string{"internal/unclassified/gate.go"}) {
		t.Fatalf("support-package skips reported %v, want only the unclassified helper", sites)
	}
	repo, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	live, err := repoanalysis.DiscoverGo(repo, "internal/cuda/testutil")
	if err != nil {
		t.Fatal(err)
	}
	if sites, err := skipReasonSites(live); err != nil || len(sites) != 0 {
		t.Fatalf("the shared CUDA gate skips without a classified reason: %v %v", sites, err)
	}
}
