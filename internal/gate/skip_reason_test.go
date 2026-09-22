package gate

import (
	"go/ast"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/repoanalysis"
)

// skipReasonSites reports every integration skip that names the short-mode
// exclusion while its own guard reads the environment. Such a skip is
// classified only when the run is short, so in a complete run it leaves its
// package incomplete and the package earns no receipt at all.
func skipReasonSites(snapshot repoanalysis.SourceSnapshot) ([]string, error) {
	var sites []string
	for _, file := range snapshot.Files {
		if !file.Test {
			continue
		}
		syntax, err := file.Syntax()
		if err != nil {
			return nil, err
		}
		ast.Inspect(syntax, func(node ast.Node) bool {
			statement, ok := node.(*ast.IfStmt)
			if !ok || !readsEnvironment(statement.Cond) {
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

// readsEnvironment reports whether a guard decides on the process environment.
func readsEnvironment(expression ast.Expr) bool {
	found := false
	ast.Inspect(expression, func(node ast.Node) bool {
		selector, ok := node.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		name, ok := selector.X.(*ast.Ident)
		found = found || ok && name.Name == "os" && selector.Sel.Name == "Getenv"
		return !found
	})
	return found
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
