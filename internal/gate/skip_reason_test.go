package gate

import (
	"encoding/json"
	"go/ast"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/repoanalysis"
)

// skipReasonSites reports every integration skip that names the short-mode
// exclusion while its own guard reads the environment. Such a skip is
// classified only when the run is short, so in a complete run it leaves its
// package incomplete and the package earns no receipt at all.
func skipReasonSites(snapshot repoanalysis.SourceSnapshot, pinned []string) ([]string, error) {
	var sites []string
	for _, file := range snapshot.Files {
		if !file.Test {
			continue
		}
		// A file under a pinned media runtime path is left as it is: naming
		// its reason would move the runtime identity and cost a reviewed
		// reconciliation layer, which is a price to pay deliberately and not
		// as a side effect of a skip message. Row store-open-lineage-cost's
		// sibling finding records what those files still owe.
		if len(movedRuntimePaths([]string{file.Path}, pinned)) != 0 {
			continue
		}
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
	data, err := os.ReadFile(filepath.Join(repo, filepath.FromSlash(mediaMergedDocument)))
	if err != nil {
		t.Fatal(err)
	}
	var pins mediaRuntimePins
	if err := json.Unmarshal(data, &pins); err != nil {
		t.Fatal(err)
	}
	sites, err := skipReasonSites(snapshot, pins.RuntimePaths)
	if err != nil {
		t.Fatal(err)
	}
	if len(sites) != 0 {
		t.Fatalf("environment-gated skips cite no reason a complete run classifies:\n%s", strings.Join(sites, "\n"))
	}
}
