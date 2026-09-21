package repoanalysis

import (
	"go/ast"
	"path"
	"path/filepath"
	"testing"
)

// noChangeTests is how many production sites outside the artifact package
// still name artifact.ErrNoChange. They are the ones that
// branch on an unchanged repeat or pass the signal up through a publisher of
// their own; the row that gives them artifact.Publish's changed result lowers
// this, and nothing raises it.
const noChangeTests = 23

// TestIdempotentCommitHasOneOwner holds the idempotent commit to its owner.
// Committing a batch and treating an unchanged repeat as success is
// artifact.Publish; no production site outside artifact commits and tests
// the error against ErrNoChange in one statement any more, and the sites
// that still name ErrNoChange do not grow. An overlaid package that writes
// the statement out is held to being found, so the walk is not vacuous.
func TestIdempotentCommitHasOneOwner(t *testing.T) {
	snapshot, err := DiscoverGo(filepath.Join("..", ".."), "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	survey := func(snapshot SourceSnapshot) (written []string, tests int) {
		for _, file := range snapshot.Files {
			if file.Test || path.Dir(file.Path) == "internal/artifact" {
				continue
			}
			syntax, err := file.Syntax()
			if err != nil {
				t.Fatal(err)
			}
			ast.Inspect(syntax, func(node ast.Node) bool {
				switch typed := node.(type) {
				case *ast.SelectorExpr:
					if typed.Sel.Name == "ErrNoChange" {
						tests++
					}
				case *ast.IfStmt:
					if typed.Init != nil && names(typed.Init, "CommitBatch") && names(typed.Cond, "ErrNoChange") {
						written = append(written, file.Path)
					}
				}
				return true
			})
		}
		return written, tests
	}
	written, tests := survey(snapshot)
	if len(written) != 0 || tests > noChangeTests {
		t.Fatalf("idempotent commits written out by hand %v; %d sites name ErrNoChange, ceiling %d: use artifact.Publish", written, tests, noChangeTests)
	}
	if tests < noChangeTests {
		t.Fatalf("%d sites name ErrNoChange, below the recorded %d: lower noChangeTests to keep the gain", tests, noChangeTests)
	}

	const rogue = `package rogue

import (
	"context"
	"errors"

	"overgo/internal/artifact"
)

func publish(ctx context.Context, repository artifact.Repository, batch artifact.Batch) error {
	if _, err := artifact.CommitBatch(ctx, repository, batch); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return err
	}
	return nil
}
`
	overlaid, err := snapshot.Overlay(map[string][]byte{"internal/rogue/rogue.go": []byte(rogue)})
	if err != nil {
		t.Fatal(err)
	}
	if written, tests := survey(overlaid); len(written) != 1 || tests != noChangeTests+1 {
		t.Fatalf("the walk found %v and %d tests in a package that writes the statement out once", written, tests)
	}
}

// names reports whether the node mentions the selector or identifier name.
func names(node ast.Node, name string) bool {
	found := false
	ast.Inspect(node, func(inner ast.Node) bool {
		if ident, ok := inner.(*ast.Ident); ok && ident.Name == name {
			found = true
		}
		return !found
	})
	return found
}
