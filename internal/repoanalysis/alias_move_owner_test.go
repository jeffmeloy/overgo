package repoanalysis

import (
	"go/ast"
	"path"
	"path/filepath"
	"testing"
)

// TestAliasMovesHaveOneOwner holds the alias move to its owner. Naming the
// target an alias held is the store's compare-and-swap, and it was written
// out by hand at thirty sites; artifact.AliasMove, MoveAlias and AliasRemoval
// own it now, so production code outside the artifact package neither sets a
// binding's Previous in a literal nor assigns it afterwards. An overlaid
// package that does both is held to being found twice, so the walk is not
// vacuous.
func TestAliasMovesHaveOneOwner(t *testing.T) {
	snapshot, err := DiscoverGo(filepath.Join("..", ".."), "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	handwritten := func(snapshot SourceSnapshot) []string {
		var sites []string
		for _, file := range snapshot.Files {
			if file.Test || path.Dir(file.Path) == "internal/artifact" {
				continue
			}
			syntax, err := file.Syntax()
			if err != nil {
				t.Fatal(err)
			}
			ast.Inspect(syntax, func(node ast.Node) bool {
				var named ast.Expr
				switch typed := node.(type) {
				case *ast.KeyValueExpr:
					named = typed.Key
				case *ast.AssignStmt:
					if selector, ok := typed.Lhs[0].(*ast.SelectorExpr); ok {
						named = selector.Sel
					}
				}
				if field, ok := named.(*ast.Ident); ok && field.Name == "Previous" && callsIDPointer(node) {
					sites = append(sites, file.Path)
				}
				return true
			})
		}
		return sites
	}
	if sites := handwritten(snapshot); len(sites) != 0 {
		t.Fatalf("alias moves written out by hand, use artifact.AliasMove, MoveAlias or AliasRemoval: %v", sites)
	}

	const rogue = `package rogue

import "overgo/internal/artifact"

func move(name string, target, held artifact.ID) []artifact.AliasBinding {
	first := artifact.AliasBinding{Name: name, Target: target, Previous: artifact.IDPointer(held)}
	second := artifact.AliasBinding{Name: name, Target: target}
	second.Previous = artifact.IDPointer(held)
	return []artifact.AliasBinding{first, second}
}
`
	overlaid, err := snapshot.Overlay(map[string][]byte{"internal/rogue/rogue.go": []byte(rogue)})
	if err != nil {
		t.Fatal(err)
	}
	if sites := handwritten(overlaid); len(sites) != 2 {
		t.Fatalf("the walk found %v in a package that writes the move out twice", sites)
	}
}

// callsIDPointer reports whether the node's value is artifact.IDPointer(...).
func callsIDPointer(node ast.Node) bool {
	var value ast.Expr
	switch typed := node.(type) {
	case *ast.KeyValueExpr:
		value = typed.Value
	case *ast.AssignStmt:
		value = typed.Rhs[0]
	}
	call, ok := value.(*ast.CallExpr)
	if !ok {
		return false
	}
	selector, ok := call.Fun.(*ast.SelectorExpr)
	return ok && selector.Sel.Name == "IDPointer"
}
